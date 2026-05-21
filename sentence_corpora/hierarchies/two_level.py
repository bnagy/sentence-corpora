"""Two-level hierarchy implementation (work/author) for sentence-corpora package.

This module provides a convenience wrapper around the base Corpus class
for two-level hierarchies (work/author). It uses composition to add
convenience methods without inheritance issues.
"""

from __future__ import annotations

import numpy as np

from ..corpus import Corpus
from ..sampling import BalancedSampler
from ..sentence import Sentence


class TwoLevelCorpus:
    """Corpus with two-level hierarchy (work/author).

    This class wraps a base Corpus and provides two-level specific
    convenience methods. It uses composition rather than inheritance
    to avoid dataclass issues.

    Args:
        sentences: List of sentences with 'work' and 'author' metadata.
    """

    def __init__(self, sentences: list[Sentence]) -> None:
        self._corpus = Corpus(sentences)

    def __len__(self) -> int:
        return len(self._corpus)

    def __iter__(self):
        return iter(self._corpus)

    def __getitem__(self, index: int) -> Sentence:
        return self._corpus[index]

    def __repr__(self) -> str:
        return f"TwoLevelCorpus({len(self._corpus)} sentences)"

    def get_levels(self) -> list[str]:
        """Return the hierarchy levels available in this corpus."""
        return ["work", "author"]

    def get_unique_values(self, level: str) -> list[str]:
        """Get unique values for a specific hierarchy level."""
        return self._corpus.get_unique_values(level)

    def filter_by_level(self, level: str, value: str) -> TwoLevelCorpus:
        """Filter corpus by a specific level and value."""
        filtered = self._corpus.filter_by_level(level, value)
        return TwoLevelCorpus(filtered._sentences)

    def by_work(self, work: str) -> TwoLevelCorpus:
        """Return a sub-corpus filtered to a single work."""
        return self.filter_by_level("work", work)

    def by_author(self, author: str) -> TwoLevelCorpus:
        """Return a sub-corpus filtered to a single author."""
        return self.filter_by_level("author", author)

    def works(self) -> list[str]:
        """Return list of unique works."""
        return self.get_unique_values("work")

    def authors(self) -> list[str]:
        """Return list of unique authors."""
        return self.get_unique_values("author")

    def sample_balanced(
        self, total: int, rng: np.random.Generator
    ) -> tuple[list[Sentence], dict]:
        """Sample sentences balanced across works and authors.

        Args:
            total: Total number of sentences to sample.
            rng: NumPy random generator.

        Returns:
            Tuple of (sampled_sentences, breakdown_dict).
        """
        grouped = BalancedSampler.group_by_levels(
            self._corpus._sentences, ["work", "author"]
        )
        samples, breakdown = BalancedSampler.sample_balanced(
            grouped, ["work", "author"], total, rng, return_sentences=True
        )
        return samples, breakdown
