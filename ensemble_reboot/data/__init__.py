"""Dataset loaders. Downloaded files live in the gitignored data/ directory;
only loaders, item-id manifests, and sha256 hashes are committed."""

from .gpqa import load_gpqa_diamond
from .hle import load_hle_items
from .lcb import load_lcb_items
# math_sets kept for backwards compatibility (disabled in pilot as of 2026-10-27)
from .math_sets import load_math_items

__all__ = ["load_gpqa_diamond", "load_hle_items", "load_lcb_items", "load_math_items"]
