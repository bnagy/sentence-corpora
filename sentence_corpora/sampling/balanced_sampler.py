"""Balanced sampling algorithms for hierarchical corpora.

Provides token-based balanced sampling: allocation across hierarchy groups
is proportional to token count, and sentences are greedily accumulated
until the token target is met. Since only complete sentences are sampled,
the actual token count is always >= the requested target.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from ..sentence import Sentence


def _sentence_tokens(s: Sentence) -> int:
    """Return the number of tokens in a sentence."""
    return len(s.text.split())


def _greedy_accumulate(
    sentences: list[Sentence],
    target_tokens: int,
    *,
    shuffle: bool = False,
    rng: np.random.Generator | None = None,
) -> tuple[list[Sentence], int]:
    """Greedily accumulate sentences until a token target is met.

    Iterates through sentences (optionally shuffled), accumulating token
    counts until the target is reached or exceeded. Since only complete
    sentences are ever selected, the actual token count is always >= target
    (unless the pool is exhausted).

    This is the shared greedy accumulation primitive used by both
    :meth:`BalancedSampler.sample_balanced` (randomized) and
    :meth:`Corpus.chunk_works` (deterministic).

    Args:
        sentences: Pool of sentences to draw from.
        target_tokens: Minimum tokens to accumulate.
        shuffle: If True, shuffle sentence order before accumulating.
        rng: Random generator (required if *shuffle* is True).

    Returns:
        Tuple of ``(selected_sentences, actual_token_count)``.
        If the pool is empty or target_tokens <= 0, returns ``([], 0)``.
        If the pool has fewer tokens than target, all sentences are returned.
    """
    if not sentences or target_tokens <= 0:
        return [], 0

    total_available = sum(_sentence_tokens(s) for s in sentences)
    if total_available <= target_tokens:
        return list(sentences), total_available

    if shuffle:
        assert rng is not None
        indices = np.arange(len(sentences))
        rng.shuffle(indices)
        ordered = [sentences[i] for i in indices]
    else:
        ordered = sentences

    selected: list[Sentence] = []
    accumulated = 0
    for s in ordered:
        selected.append(s)
        accumulated += _sentence_tokens(s)
        if accumulated >= target_tokens:
            break

    return selected, accumulated


class BalancedSampler:
    """Generalized balanced sampler that adapts to hierarchy depth.

    Sampling is token-based: ``target_tokens`` specifies the minimum number
    of tokens to sample. Since only complete sentences are ever selected,
    the actual token count is always >= target. Allocation across groups
    is proportional to each group's share of total tokens.
    """

    @staticmethod
    def group_by_levels(sentences: list[Sentence], levels: list[str]) -> dict[str, Any]:
        """Recursively group sentences by hierarchy levels.

        Args:
            sentences: List of sentences to group.
            levels: List of hierarchy levels in order
                (e.g., ``['translator', 'author', 'work']``).

        Returns:
            Nested dictionary structure matching the hierarchy levels.
        """
        if not levels:
            return {}  # type: ignore[return-value]

        level = levels[0]
        rest_levels = levels[1:]

        grouped: dict[str, list[Sentence]] = defaultdict(list)
        for sentence in sentences:
            key = getattr(sentence, level)
            grouped[key].append(sentence)

        if rest_levels:
            result = {}
            for key, group in grouped.items():
                result[key] = BalancedSampler.group_by_levels(group, rest_levels)
            return result
        else:
            return dict(grouped)

    @staticmethod
    def sample_balanced(
        grouped_sentences: Any,
        levels: list[str],
        target_tokens: int,
        rng: np.random.Generator,
        return_sentences: bool = True,
    ) -> tuple[list, dict]:
        """Perform balanced sampling across hierarchy levels by token count.

        At each leaf, sentences are shuffled and greedily accumulated until
        the allocated token target is met or exceeded. Since only complete
        sentences are sampled, the actual token count is always >= target.

        Args:
            grouped_sentences: Nested dictionary from :meth:`group_by_levels`.
            levels: List of hierarchy levels being sampled.
            target_tokens: Minimum number of tokens to sample.
            rng: NumPy random generator for reproducibility.
            return_sentences: If True, return Sentence objects;
                if False, return token tuples.

        Returns:
            Tuple of ``(sampled_items, breakdown_dict)``. Breakdown values
            are token counts (not sentence counts).
        """
        if not levels or target_tokens <= 0:
            return [], {}

        rest_levels = levels[1:]

        allocations = BalancedSampler._allocate_tokens_evenly(
            grouped_sentences, target_tokens
        )

        all_samples: list = []
        breakdown: dict = {}

        for key, allocated_tokens in allocations.items():
            current_level_data = grouped_sentences[key]

            if rest_levels and isinstance(current_level_data, dict):
                sub_samples, sub_breakdown = BalancedSampler.sample_balanced(
                    current_level_data,
                    rest_levels,
                    allocated_tokens,
                    rng,
                    return_sentences,
                )
                all_samples.extend(sub_samples)
                breakdown[key] = sub_breakdown
            else:
                if isinstance(current_level_data, list):
                    selected, token_count = BalancedSampler._sample_sentences_greedy(
                        current_level_data, allocated_tokens, rng
                    )
                    if return_sentences:
                        all_samples.extend(selected)
                    else:
                        all_samples.extend([tuple(s.text.split()) for s in selected])
                    breakdown[key] = token_count
                else:
                    # Single sentence (edge case)
                    if return_sentences:
                        all_samples.append(current_level_data)
                    else:
                        all_samples.append(tuple(current_level_data.text.split()))
                    breakdown[key] = _sentence_tokens(current_level_data)

        return all_samples, breakdown

    @staticmethod
    def _sample_sentences_greedy(
        sentences: list[Sentence],
        target_tokens: int,
        rng: np.random.Generator,
    ) -> tuple[list[Sentence], int]:
        """Shuffle sentences and greedily accumulate until target is met.

        Delegates to :func:`_greedy_accumulate` with ``shuffle=True``.

        Args:
            sentences: Pool of sentences to sample from.
            target_tokens: Minimum tokens to accumulate.
            rng: NumPy random generator.

        Returns:
            Tuple of ``(selected_sentences, actual_token_count)``.
        """
        return _greedy_accumulate(sentences, target_tokens, shuffle=True, rng=rng)

    @staticmethod
    def _allocate_tokens_evenly(groups: dict, target_tokens: int) -> dict[str, int]:
        """Allocate token targets proportionally across groups.

        Each group receives a share of the target proportional to its
        fraction of total tokens. The remainder (from integer division)
        is distributed one token each to the first N groups.

        Args:
            groups: Dictionary mapping group names to their contents
                (nested dicts or lists of sentences).
            target_tokens: Total tokens to allocate.

        Returns:
            Dictionary mapping group names to allocated token targets.
        """
        group_names = sorted(groups.keys())
        total_tokens = sum(
            BalancedSampler._get_group_tokens(groups[g]) for g in group_names
        )

        if total_tokens == 0:
            return {g: 0 for g in group_names}

        allocated: dict[str, int] = {}
        remaining = target_tokens

        for idx, group_name in enumerate(group_names):
            group_tokens = BalancedSampler._get_group_tokens(groups[group_name])
            # Proportional share, rounded down
            share = (target_tokens * group_tokens) // total_tokens
            allocated[group_name] = share
            remaining -= share

        # Distribute remainder to first N groups
        for idx in range(min(remaining, len(group_names))):
            allocated[group_names[idx]] += 1

        return allocated

    @staticmethod
    def _get_group_tokens(group_data: Any) -> int:
        """Calculate the total token count of a group.

        Args:
            group_data: A list of Sentences, a nested dict, or a single Sentence.

        Returns:
            Total number of tokens in the group.
        """
        if isinstance(group_data, list):
            return sum(_sentence_tokens(s) for s in group_data)
        elif isinstance(group_data, dict):
            return sum(
                BalancedSampler._get_group_tokens(v) for v in group_data.values()
            )
        elif isinstance(group_data, Sentence):
            return _sentence_tokens(group_data)
        else:
            return 0
