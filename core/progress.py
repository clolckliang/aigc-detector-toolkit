"""Progress helpers with a no-op fallback when tqdm is unavailable."""
from typing import Iterable, Optional


def progress_iter(iterable: Iterable, total: Optional[int] = None, desc: str = ""):
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, total=total, desc=desc, unit="段", leave=False)
    except Exception:
        return iterable
