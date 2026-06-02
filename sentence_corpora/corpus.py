"""Corpus class for sentence-corpora package."""

from __future__ import annotations

import pickle
import random
from collections.abc import Iterator

from .sentence import Sentence


def _balanced_loss(sizes: list[int], min_tokens: int) -> float:
    """Compute balanced loss: average of evenness and min-size losses.

    evenness_loss = mean squared deviation from actual mean chunk size
    min_size_loss = mean squared deviation from min_tokens (only below-min chunks)
    """
    if not sizes:
        return 0.0
    n = len(sizes)
    mean_size = sum(sizes) / n
    evenness = sum((sz - mean_size) ** 2 for sz in sizes) / n
    min_loss = sum((min_tokens - sz) ** 2 for sz in sizes if sz < min_tokens) / n
    return 0.5 * evenness + 0.5 * min_loss


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

        Uses a two-phase approach:
        1. Greedy accumulation with backward merge: forward accumulate sentences
           until >= min_tokens, finalize chunk, repeat. The last chunk (remainder)
           is merged backward into the previous chunk, guaranteeing all chunks
           >= min_tokens.
        2. Ripple optimization: iterate right-to-left over chunks, trying to move
           the first sentence of each chunk backward. Accept moves that reduce
           the balanced loss (evenness + min-size) without dropping any chunk
           below floor (0.8 * min_tokens).

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
        by_work: dict[tuple, list[Sentence]] = {}
        for s in self._sentences:
            work = s.metadata.get(work_level)
            if not isinstance(work, str):
                continue
            key_parts = [work]
            for level in levels[1:]:
                key_parts.append(str(s.metadata.get(level, "")))
            key = tuple(key_parts)
            if key not in by_work:
                by_work[key] = []
            by_work[key].append(s)

        new_sentences: list[Sentence] = []
        for key, sentences in by_work.items():
            work = key[0]

            # Defensive check: verify all sentences in this group share the
            # same hierarchy metadata.
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

            # Phase 1: Greedy accumulation with backward merge
            chunks = self._greedy_backward_merge(sentences, min_tokens)

            # Phase 2: Ripple optimization
            self._ripple_optimize(chunks, min_tokens)

            # Build output sentences with chunk metadata
            for chunk_idx, chunk in enumerate(chunks, 1):
                for s in chunk:
                    new_s = Sentence(
                        text=s.text,
                        metadata={
                            **s.metadata,
                            work_level: f"{work}_{chunk_idx}",
                        },
                    )
                    new_sentences.append(new_s)

        return Corpus(new_sentences)

    @staticmethod
    def _greedy_backward_merge(
        sentences: list[Sentence], min_tokens: int
    ) -> list[list[Sentence]]:
        """Forward accumulation with backward merge of the last chunk.

        Accumulates sentences until sum >= min_tokens, then finalizes the chunk.
        The final remainder (if any) is merged into the previous chunk.
        This guarantees every chunk has >= min_tokens.
        """
        chunks: list[list[Sentence]] = []
        current: list[Sentence] = []
        acc = 0
        for s in sentences:
            current.append(s)
            acc += len(s.text.split())
            if acc >= min_tokens:
                chunks.append(current)
                current = []
                acc = 0
        if current:
            if chunks:
                chunks[-1].extend(current)
            else:
                chunks.append(current)
        return chunks

    @staticmethod
    def _ripple_optimize(
        chunks: list[list[Sentence]], min_tokens: int
    ) -> None:
        """Optimize chunk sizes by rippling sentences backward.

        Iterates right-to-left over chunks. For each chunk, tries moving its
        first sentence backward to the previous chunk. Accepts the move if it
        reduces the balanced loss and the source chunk stays >= floor
        (0.8 * min_tokens). After accepting, tries again from the same chunk.
        Repeats full passes until no moves are accepted.

        Modifies chunks in place.
        """
        floor = int(min_tokens * 0.8)
        if len(chunks) <= 1:
            return

        sizes = [sum(len(c.text.split()) for c in chunk) for chunk in chunks]

        for _ in range(len(chunks) * 20):
            moved_any = False
            i = len(chunks) - 1
            while i >= 1:
                if len(chunks[i]) <= 1:
                    i -= 1
                    continue

                sent_sz = len(chunks[i][0].text.split())
                new_src = sizes[i] - sent_sz

                if new_src < floor:
                    i -= 1
                    continue

                # Compute balanced loss before and after the move
                old_loss = _balanced_loss(sizes, min_tokens)
                test_sizes = list(sizes)
                test_sizes[i] = new_src
                test_sizes[i - 1] += sent_sz
                new_loss = _balanced_loss(test_sizes, min_tokens)

                if new_loss < old_loss - 1e-9:
                    moved = chunks[i].pop(0)
                    chunks[i - 1].append(moved)
                    sizes[i] = new_src
                    sizes[i - 1] += sent_sz
                    moved_any = True
                    # Don't decrement i -- try to ripple more from same chunk
                else:
                    i -= 1

            if not moved_any:
                break
