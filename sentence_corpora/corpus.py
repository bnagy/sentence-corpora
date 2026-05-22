"""Corpus class for sentence-corpora package."""

from __future__ import annotations

import pickle
import random
from collections.abc import Iterator

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

        # Group sentences by work
        by_work: dict[str, list[Sentence]] = {}
        for s in self._sentences:
            work = s.metadata.get(work_level)
            if isinstance(work, str):
                if work not in by_work:
                    by_work[work] = []
                by_work[work].append(s)

        new_sentences = []
        for work, sentences in by_work.items():
            total_tokens = sum(len(s.text.split()) for s in sentences)
            if total_tokens < min_tokens:
                raise ValueError(
                    f"Work '{work}' has {total_tokens} tokens, "
                    f"which is less than min_tokens={min_tokens}"
                )

            # Calculate number of chunks: we want to minimize chunks while ensuring
            # each has at least min_tokens
            # num_chunks = floor(total_tokens / min_tokens)
            num_chunks = total_tokens // min_tokens

            # Calculate target tokens per chunk (evenly distributed)
            base_tokens = total_tokens // num_chunks
            remainder = total_tokens % num_chunks

            # Build chunks by iterating through sentences
            chunk_idx = 0
            current_tokens = 0
            target = base_tokens + (1 if chunk_idx < remainder else 0)

            for s in sentences:
                new_s = Sentence(
                    text=s.text,
                    metadata={**s.metadata, work_level: f"{work}_{chunk_idx + 1}"},
                )
                new_sentences.append(new_s)
                current_tokens += len(s.text.split())

                # Move to next chunk when we've reached the target
                # But only if we're not on the last chunk yet
                if current_tokens >= target and chunk_idx < num_chunks - 1:
                    chunk_idx += 1
                    current_tokens = 0
                    target = base_tokens + (1 if chunk_idx < remainder else 0)

        return Corpus(new_sentences)
