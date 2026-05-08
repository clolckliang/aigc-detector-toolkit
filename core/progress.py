"""Unified progress helpers with Rich/tqdm/no-op fallbacks."""
from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterable, Optional


_CURRENT_MANAGER: ContextVar[Optional["ProgressManager"]] = ContextVar("progress_manager", default=None)


def _format_stats(fields: dict) -> str:
    parts = []
    labels = {
        "changed": "改",
        "skipped": "跳过",
        "rounds": "轮",
        "failed": "失败",
        "ai_ratio": "AI",
        "eta": "剩余",
    }
    for key in ("changed", "skipped", "rounds", "failed", "ai_ratio", "eta"):
        value = fields.get(key)
        if value is None:
            continue
        if key == "ai_ratio" and isinstance(value, (int, float)):
            value = f"{value:.1f}%"
        parts.append(f"{labels[key]}={value}")
    return " ".join(parts)


class NullProgress:
    """Progress-compatible no-op handle."""

    def update(self, n: int = 1):
        return None

    def set_postfix(self, *args, **kwargs):
        return None

    def close(self):
        return None


class ProgressHandle:
    """Small compatibility wrapper for tqdm-like progress bars."""

    def __init__(self, manager: "ProgressManager", task_id):
        self.manager = manager
        self.task_id = task_id

    def update(self, n: int = 1):
        self.manager.advance(self.task_id, n)

    def set_postfix(self, *args, **kwargs):
        fields = {}
        if args and isinstance(args[0], dict):
            fields.update(args[0])
        fields.update(kwargs)
        self.manager.update(self.task_id, **fields)

    def close(self):
        self.manager.complete(self.task_id)


class ProgressManager:
    """
    Rich-backed progress manager.

    Falls back to no-op when Rich is unavailable, output is non-interactive, or disabled.
    Existing progress_iter/progress_bar calls automatically attach to the active manager.
    """

    def __init__(self, enabled: bool = True, transient: bool = True, force: bool = False):
        self.enabled = enabled
        self.transient = transient
        self.force = force
        self._progress = None
        self._token = None
        self._rich_enabled = False
        self._task_fields = {}

    def __enter__(self) -> "ProgressManager":
        self._rich_enabled = self._should_use_rich()
        if self._rich_enabled:
            from rich.progress import (
                BarColumn,
                Progress,
                TaskProgressColumn,
                TextColumn,
                TimeElapsedColumn,
            )
            from rich.table import Column

            self._progress = Progress(
                TextColumn("[progress.description]{task.description}", table_column=Column(ratio=2, no_wrap=True)),
                BarColumn(bar_width=None, table_column=Column(ratio=3)),
                TaskProgressColumn(),
                TextColumn("{task.fields[status]}", table_column=Column(width=16, no_wrap=True)),
                TextColumn("{task.fields[stats]}", table_column=Column(width=28, no_wrap=True)),
                TimeElapsedColumn(),
                transient=self.transient,
                expand=True,
            )
            self._progress.start()
        self._token = _CURRENT_MANAGER.set(self)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._token is not None:
            _CURRENT_MANAGER.reset(self._token)
            self._token = None
        if self._progress is not None:
            self._progress.stop()
            self._progress = None
        self._task_fields.clear()

    def _should_use_rich(self) -> bool:
        if not self.enabled:
            return False
        if os.environ.get("CI"):
            return False
        if not self.force and not sys.stderr.isatty():
            return False
        try:
            import rich  # noqa: F401
        except Exception:
            return False
        return True

    @property
    def active(self) -> bool:
        return self._rich_enabled and self._progress is not None

    def add_task(self, description: str, total: Optional[int] = None, unit: str = "段", status: str = ""):
        if not self.active:
            return None
        task_id = self._progress.add_task(
            description,
            total=total,
            status=status,
            stats="",
            unit=unit,
            changed=None,
            skipped=None,
            rounds=None,
            failed=None,
            ai_ratio=None,
            eta=None,
        )
        self._task_fields[task_id] = {
            "status": status,
            "stats": "",
            "changed": None,
            "skipped": None,
            "rounds": None,
            "failed": None,
            "ai_ratio": None,
            "eta": None,
        }
        return task_id

    def update(self, task_id, completed: Optional[int] = None, total: Optional[int] = None,
               status: Optional[str] = None, **fields):
        if not self.active or task_id is None or task_id not in self._task_fields:
            return
        update_kwargs = {}
        if completed is not None:
            update_kwargs["completed"] = completed
        if total is not None:
            update_kwargs["total"] = total
        if status is not None:
            update_kwargs["status"] = status

        task_fields = dict(self._task_fields.get(task_id, {}))
        task_fields.update({k: v for k, v in fields.items() if v is not None})
        if status is not None:
            task_fields["status"] = status
        update_kwargs.update(task_fields)
        update_kwargs["stats"] = _format_stats(task_fields)
        task_fields["stats"] = update_kwargs["stats"]
        self._task_fields[task_id] = task_fields
        self._progress.update(task_id, **update_kwargs)

    def advance(self, task_id, n: int = 1, **fields):
        if not self.active or task_id is None or task_id not in self._task_fields:
            return
        self.update(task_id, **fields)
        self._progress.advance(task_id, n)
        task = next((t for t in self._progress.tasks if t.id == task_id), None)
        if task and task.total is not None and task.completed >= task.total:
            self.complete(task_id)

    def complete(self, task_id, status: str = "完成"):
        if not self.active or task_id is None or task_id not in self._task_fields:
            return
        self.update(task_id, status=status)
        self._progress.remove_task(task_id)
        self._task_fields.pop(task_id, None)

    def iter(self, iterable: Iterable, total: Optional[int] = None, desc: str = "", unit: str = "段"):
        task_id = self.add_task(desc, total=total, unit=unit)
        for item in iterable:
            yield item
            self.advance(task_id, 1)
        self.complete(task_id)


def current_progress() -> Optional[ProgressManager]:
    return _CURRENT_MANAGER.get()


def progress_iter(iterable: Iterable, total: Optional[int] = None, desc: str = "", unit: str = "段"):
    manager = current_progress()
    if manager is not None:
        if manager.active:
            return manager.iter(iterable, total=total, desc=desc, unit=unit)
        return iterable
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, total=total, desc=desc, unit=unit, leave=False)
    except Exception:
        return iterable


@contextmanager
def progress_bar(total: Optional[int] = None, desc: str = "", unit: str = "段"):
    manager = current_progress()
    if manager is not None:
        if manager.active:
            task_id = manager.add_task(desc, total=total, unit=unit)
            yield ProgressHandle(manager, task_id)
            manager.complete(task_id)
        else:
            yield NullProgress()
        return

    try:
        from tqdm.auto import tqdm

        bar = tqdm(total=total, desc=desc, unit=unit, leave=False)
    except Exception:
        bar = NullProgress()

    try:
        yield bar
    finally:
        bar.close()
