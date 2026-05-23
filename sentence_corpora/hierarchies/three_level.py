"""Three-level hierarchy implementation (work/author/translator) for sentence-corpora package.

This module provides a convenience wrapper around the base Corpus class
for three-level hierarchies (work/author/translator). It uses composition
to add convenience methods without inheritance issues.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..corpus import Corpus
from ..sampling import BalancedSampler
from ..sentence import Sentence


class ThreeLevelCorpus:
    """Corpus with three-level hierarchy (work/author/translator).

    This class wraps a base Corpus and provides three-level specific
    convenience methods. It uses composition rather than inheritance
    to avoid dataclass issues.

    Args:
        sentences: List of sentences with 'work', 'author', and
            'translator' metadata.
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
        return f"ThreeLevelCorpus({len(self._corpus)} sentences)"

    def get_levels(self) -> list[str]:
        """Return the hierarchy levels available in this corpus.

        Ordered from lowest (work) to highest (translator), matching
        the convention used by :class:`TwoLevelCorpus`.
        """
        return ["work", "author", "translator"]

    def get_unique_values(self, level: str) -> list[str]:
        """Get unique values for a specific hierarchy level."""
        return self._corpus.get_unique_values(level)

    def filter_by_level(self, level: str, value: str) -> ThreeLevelCorpus:
        """Filter corpus by a specific level and value."""
        filtered = self._corpus.filter_by_level(level, value)
        return ThreeLevelCorpus(filtered._sentences)

    def by_translator(self, translator: str) -> ThreeLevelCorpus:
        """Return a sub-corpus filtered to a single translator."""
        return self.filter_by_level("translator", translator)

    def by_author(self, author: str) -> ThreeLevelCorpus:
        """Return a sub-corpus filtered to a single author."""
        return self.filter_by_level("author", author)

    def by_work(self, work: str) -> ThreeLevelCorpus:
        """Return a sub-corpus filtered to a single work."""
        return self.filter_by_level("work", work)

    def translators(self) -> list[str]:
        """Return list of unique translators."""
        return self.get_unique_values("translator")

    def authors(self) -> list[str]:
        """Return list of unique authors."""
        return self.get_unique_values("author")

    def works(self) -> list[str]:
        """Return list of unique works."""
        return self.get_unique_values("work")

    def sample_balanced(
        self,
        target_tokens: int,
        rng: np.random.Generator,
        exclude_translator: str | None = None,
    ) -> tuple[list[Sentence], dict]:
        """Sample sentences balanced across translators, authors, and works by token count.

        Args:
            target_tokens: Minimum number of tokens to sample.
            rng: NumPy random generator.
            exclude_translator: Optional translator name to exclude.

        Returns:
            Tuple of (sampled_sentences, breakdown_dict). Breakdown values
            are token counts.
        """
        sentences = self._corpus._sentences
        if exclude_translator is not None:
            sentences = [s for s in sentences if s.translator != exclude_translator]

        grouped = BalancedSampler.group_by_levels(
            sentences, ["translator", "author", "work"]
        )
        samples, breakdown = BalancedSampler.sample_balanced(
            grouped,
            ["translator", "author", "work"],
            target_tokens,
            rng,
            return_sentences=True,
        )
        return samples, breakdown

    def stats(self) -> None:
        """Print a table of works, sentences, and tokens per translator."""
        from tabulate import tabulate

        rows: dict[str, dict[str, Any]] = {}
        for sentence in self._corpus:
            t = str(sentence.translator)
            if t not in rows:
                rows[t] = {"works": set(), "sentences": 0, "tokens": 0}
            rows[t]["works"].add(str(sentence.work))
            rows[t]["sentences"] += 1
            rows[t]["tokens"] += len(sentence.text.split())

        table = [
            [t, len(rows[t]["works"]), rows[t]["sentences"], rows[t]["tokens"]]
            for t in sorted(rows)
        ]
        table.append(
            [
                "TOTAL",
                len({w for r in rows.values() for w in r["works"]}),
                sum(r["sentences"] for r in rows.values()),
                sum(r["tokens"] for r in rows.values()),
            ]
        )
        print(
            tabulate(
                table,
                headers=["Translator", "Works", "Sentences", "Tokens"],
                tablefmt="simple",
                intfmt=",",
            )
        )

    def to_pickle(self, path: str) -> None:
        """Save this corpus to a pickle file."""
        self._corpus.to_pickle(path)

    @classmethod
    def from_pickle(cls, path: str) -> ThreeLevelCorpus:
        """Load a corpus from a pickle file."""
        corpus = Corpus.from_pickle(path)
        return cls(corpus._sentences)

    def chunk_works(self, min_tokens: int) -> ThreeLevelCorpus:
        """Chunk works into smaller pieces with at least min_tokens each."""
        chunked = self._corpus.chunk_works(min_tokens)
        return ThreeLevelCorpus(chunked._sentences)
