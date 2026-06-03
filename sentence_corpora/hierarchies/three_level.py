"""Three-level hierarchy implementation for sentence-corpora package.

This module provides a convenience wrapper around the base Corpus class
for three-level hierarchies. It uses composition to add convenience
methods without inheritance issues.

The three levels are configurable, ordered from highest to lowest in the
has-many nesting (e.g., ``("translator", "author", "work")`` or
``("meter", "author", "work")``). If *levels* is not passed explicitly,
they are auto-detected from the first sentence's metadata keys.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from ..corpus import Corpus
from ..sampling import BalancedSampler
from ..sentence import Sentence


class ThreeLevelCorpus:
    """Corpus with three-level hierarchy.

    This class wraps a base Corpus and provides three-level specific
    convenience methods. It uses composition rather than inheritance
    to avoid dataclass issues.

    Args:
        sentences: List of sentences with metadata keys matching *levels*.
        levels: Tuple of three level names, from highest to lowest in the
            has-many nesting (e.g., ``("translator", "author", "work")``).
            If not provided, levels are auto-detected from the first
            sentence's metadata keys. A warning is emitted. Explicit is
            recommended.
    """

    def __init__(
        self,
        sentences: list[Sentence],
        levels: tuple[str, str, str] | None = None,
    ) -> None:
        self._corpus = Corpus(sentences)
        if levels is not None:
            self._levels = levels
        elif sentences:
            detected = tuple(sentences[0].metadata.keys())
            if len(detected) != 3:
                raise ValueError(
                    f"Expected exactly 3 metadata keys for ThreeLevelCorpus, "
                    f"got {len(detected)}: {detected}. Pass levels explicitly."
                )
            self._levels = detected  # type: ignore[assignment]
            warnings.warn(
                f"ThreeLevelCorpus: auto-detected levels {detected} from "
                f"sentence metadata keys (assumed highest-to-lowest). "
                f"Pass levels= explicitly to suppress this warning.",
                stacklevel=2,
            )
        else:
            self._levels = ("translator", "author", "work")

    @property
    def levels(self) -> tuple[str, str, str]:
        """The three level names, from highest to lowest."""
        return self._levels

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

        Ordered from lowest to highest.
        """
        return list(self._levels)

    def get_unique_values(self, level: str) -> list[str]:
        """Get unique values for a specific hierarchy level."""
        return self._corpus.get_unique_values(level)

    def filter_by_level(self, level: str, value: str) -> ThreeLevelCorpus:
        """Filter corpus by a specific level and value."""
        filtered = self._corpus.filter_by_level(level, value)
        return ThreeLevelCorpus(filtered._sentences, levels=self._levels)

    def sample_balanced(
        self,
        target_tokens: int,
        rng: np.random.Generator,
        exclude: tuple[str, str] | None = None,
    ) -> tuple[list[Sentence], dict]:
        """Sample sentences balanced across all three levels by token count.

        Args:
            target_tokens: Minimum number of tokens to sample.
            rng: NumPy random generator.
            exclude: Optional ``(level, value)`` tuple to exclude from sampling.

        Returns:
            Tuple of (sampled_sentences, breakdown_dict). Breakdown values
            are token counts.
        """
        sentences = self._corpus._sentences
        if exclude is not None:
            exclude_level, exclude_value = exclude
            sentences = [
                s
                for s in sentences
                if s.metadata.get(exclude_level) != exclude_value
            ]

        level_order = list(self._levels)
        grouped = BalancedSampler.group_by_levels(sentences, level_order)
        samples, breakdown = BalancedSampler.sample_balanced(
            grouped,
            level_order,
            target_tokens,
            rng,
            return_sentences=True,
        )
        return samples, breakdown

    def stats(self) -> None:
        """Print a table grouped by the top-level entity."""
        from tabulate import tabulate

        top_level = self._levels[0]
        bottom_level = self._levels[2]
        rows: dict[str, dict[str, Any]] = {}
        for sentence in self._corpus:
            key = str(sentence.metadata.get(top_level, ""))
            if key not in rows:
                rows[key] = {"works": set(), "sentences": 0, "tokens": 0}
            rows[key]["works"].add(str(sentence.metadata.get(bottom_level, "")))
            rows[key]["sentences"] += 1
            rows[key]["tokens"] += len(sentence.text.split())

        def _mean(tokens: int, sentences: int) -> str:
            return f"{tokens / sentences:.1f}" if sentences else "—"

        table = [
            [
                key,
                len(rows[key]["works"]),
                rows[key]["sentences"],
                rows[key]["tokens"],
                _mean(rows[key]["tokens"], rows[key]["sentences"]),
            ]
            for key in sorted(rows)
        ]
        total_s = sum(r["sentences"] for r in rows.values())
        total_t = sum(r["tokens"] for r in rows.values())
        table.append(
            [
                "TOTAL",
                len({w for r in rows.values() for w in r["works"]}),
                total_s,
                total_t,
                _mean(total_t, total_s),
            ]
        )
        print(
            tabulate(
                table,
                headers=[
                    top_level.capitalize(),
                    "Works",
                    "Sentences",
                    "Tokens",
                    "Mean Len",
                ],
                tablefmt="simple",
                intfmt=",",
            )
        )

    def to_pickle(self, path: str) -> None:
        """Save this corpus to a pickle file."""
        self._corpus.to_pickle(path)

    @classmethod
    def from_pickle(
        cls,
        path: str,
        levels: tuple[str, str, str] | None = None,
    ) -> ThreeLevelCorpus:
        """Load a corpus from a pickle file.

        Args:
            path: Path to the pickle file.
            levels: Level names to use. If not provided, levels are
                auto-detected from the first sentence's metadata keys.
                A warning is emitted. Explicit is recommended.
        """
        obj = Corpus.from_pickle(path)
        if isinstance(obj, cls):
            return obj
        return cls(obj._sentences, levels=levels)

    def chunk_works(self, min_tokens: int) -> ThreeLevelCorpus:
        """Chunk works into smaller pieces with at least min_tokens each."""
        chunked = self._corpus.chunk_works(min_tokens)
        return ThreeLevelCorpus(chunked._sentences, levels=self._levels)
