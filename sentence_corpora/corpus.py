"""Corpus class for sentence-corpora package."""

from __future__ import annotations

import pickle
import random
from collections.abc import Iterator

from .sampling.balanced_sampler import _greedy_accumulate
from .sentence import Sentence


class Corpus:
    """A collection of sentences with hierarchical metadata.

    This class manages a list of Sentence objects and provides methods
    for filtering, sampling, and querying the corpus based on hierarchy
    levels stored in the sentence metadata.

    Args:
        sentences: List of Sentence objects.
    """

    def __init__(self, sentences: list[Sentence]) -> None:
        self._sentences = list(sentences)

    def __len__(self) -> int:
        return len(self._sentences)

    def __iter__(self) -> Iterator[Sentence]:
        return iter(self._sentences)

    def __getitem__(self, index: int) -> Sentence:
        return self._sentences[index]

    def __repr__(self) -> str:
        return f"Corpus({len(self._sentences)} sentences)"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Corpus):
            return NotImplemented
        return self._sentences == other._sentences

    def sample(self, n: int) -> list[Sentence]:
        """Return *n* sentences sampled without replacement."""
        return random.sample(self._sentences, n)

    def get_levels(self) -> list[str]:
        """Return the hierarchy levels available in this corpus."""
        if not self._sentences:
            return []
        return list(self._sentences[0].metadata.keys())

    def get_unique_values(self, level: str) -> list[str]:
        """Get unique values for a specific hierarchy level."""
        values: set[str] = set()
        for s in self._sentences:
            v = s.metadata[level]
            if isinstance(v, str):
                values.add(v)
        return sorted(values)

    def filter_by_level(self, level: str, value: str) -> Corpus:
        """Filter corpus by a specific level and value."""
        filtered = [s for s in self._sentences if s.metadata.get(level) == value]
        return Corpus(filtered)

    def to_pickle(self, path: str) -> None:
        """Save this Corpus to a pickle file."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_pickle(cls, path: str) -> Corpus:
        """Load a Corpus from a pickle file."""
        with open(path, "rb") as f:
            return pickle.load(f)

    def chunk_works(self, min_tokens: int) -> Corpus:
        """Chunk works into smaller pieces with at least min_tokens each.

        Args:
            min_tokens: Minimum number of tokens per chunk.

        Returns:
            A new Corpus with chunked works named {work}_{chunk_num}.

        Raises:
            ValueError: If any work has fewer tokens than min_tokens.
        """
        if not self._sentences:
            return Corpus([])

        levels = self.get_levels()
        if not levels:
            return Corpus(self._sentences)

        work_level = levels[0]  # First level is the work level

        # Group sentences by the full hierarchy tuple to avoid collisions.
        # For two-level corpora this is (work, author); for three-level it's
        # (work, author, translator). Chunks must never mix sentences from
        # different hierarchy entities.
        by_work: dict[tuple, list[Sentence]] = {}
        for s in self._sentences:
            work = s.metadata.get(work_level)
            if not isinstance(work, str):
                continue
            # Build grouping key from all hierarchy levels except work
            key_parts = [work]
            for level in levels[1:]:
                key_parts.append(str(s.metadata.get(level, "")))
            key = tuple(key_parts)
            if key not in by_work:
                by_work[key] = []
            by_work[key].append(s)

        new_sentences = []
        for key, sentences in by_work.items():
            work = key[0]

            # Defensive check: verify all sentences in this group share the
            # same hierarchy metadata. This should always pass since we
            # grouped by the full tuple, but guards against future bugs.
            if len(key) > 1:
                for level_idx, level in enumerate(levels[1:], 1):
                    expected = key[level_idx]
                    for s in sentences:
                        actual = str(s.metadata.get(level, ""))
                        if actual != expected:
                            raise AssertionError(
                                f"Chunking integrity violation: sentences with "
                                f"different '{level}' values ({expected!r} vs "
                                f"{actual!r}) were mixed into the same chunk "
                                f"for work {work!r}"
                            )

            total_tokens = sum(len(s.text.split()) for s in sentences)
            if total_tokens < min_tokens:
                raise ValueError(
                    f"Work '{work}' has {total_tokens} tokens, "
                    f"which is less than min_tokens={min_tokens}"
                )

            # Use dynamic target chunking: after each chunk, recalculate
            # the target size based on remaining tokens and remaining chunks.
            # This keeps chunks as even as possible while ensuring each
            # non-last chunk has at least min_tokens.
            num_chunks = total_tokens // min_tokens
            if num_chunks == 0:
                num_chunks = 1

            remaining = list(sentences)
            chunks_remaining = num_chunks
            chunk_idx = 0

            while remaining:
                is_last = chunks_remaining <= 1
                if is_last:
                    # Last chunk gets all remaining sentences
                    selected = remaining
                    remaining = []
                else:
                    # Recalculate target based on what's left
                    tokens_remaining = sum(len(s.text.split()) for s in remaining)
                    target = tokens_remaining // chunks_remaining
                    if target < min_tokens:
                        target = min_tokens
                    selected, _ = _greedy_accumulate(remaining, target, shuffle=False)
                    remaining = remaining[len(selected) :]

                chunk_idx += 1
                for s in selected:
                    new_s = Sentence(
                        text=s.text,
                        metadata={
                            **s.metadata,
                            work_level: f"{work}_{chunk_idx}",
                        },
                    )
                    new_sentences.append(new_s)

                chunks_remaining -= 1

        return Corpus(new_sentences)
