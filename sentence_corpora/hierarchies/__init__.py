"""Hierarchy-specific corpus wrappers for sentence-corpora package.

This module provides convenience wrappers around the base Corpus class
for common hierarchy configurations (2-level and 3-level). These classes
use composition rather than inheritance to avoid dataclass issues.
"""

from __future__ import annotations

from .two_level import TwoLevelCorpus
from .three_level import ThreeLevelCorpus

__all__ = ["TwoLevelCorpus", "ThreeLevelCorpus"]
