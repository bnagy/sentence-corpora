"""Balanced sampling algorithms for hierarchical corpora."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from ..sentence import Sentence


class BalancedSampler:
    """Generalized balanced sampler that adapts to hierarchy depth.

    This class provides balanced sampling algorithms that work with
    any number of hierarchy levels (2-3 levels supported).
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
        total_samples: int,
        rng: np.random.Generator,
        return_sentences: bool = True,
    ) -> tuple[list, dict]:
        """Perform balanced sampling across hierarchy levels.

        Args:
            grouped_sentences: Nested dictionary from :meth:`group_by_levels`.
            levels: List of hierarchy levels being sampled.
            total_samples: Total number of samples to draw.
            rng: NumPy random generator for reproducibility.
            return_sentences: If True, return Sentence objects;
                if False, return token tuples.

        Returns:
            Tuple of ``(sampled_items, breakdown_dict)``.
        """
        if not levels:
            return [], {}

        rest_levels = levels[1:]

        allocations = BalancedSampler._allocate_samples_evenly(
            grouped_sentences, total_samples
        )

        all_samples: list = []
        breakdown: dict = {}

        for key, allocated_count in allocations.items():
            current_level_data = grouped_sentences[key]

            if rest_levels and isinstance(current_level_data, dict):
                sub_samples, sub_breakdown = BalancedSampler.sample_balanced(
                    current_level_data,
                    rest_levels,
                    allocated_count,
                    rng,
                    return_sentences,
                )
                all_samples.extend(sub_samples)
                breakdown[key] = sub_breakdown
            else:
                if isinstance(current_level_data, list):
                    if allocated_count >= len(current_level_data):
                        selected = current_level_data
                    else:
                        indices = rng.choice(
                            len(current_level_data),
                            size=allocated_count,
                            replace=False,
                        )
                        selected = [current_level_data[i] for i in indices]

                    if return_sentences:
                        all_samples.extend(selected)
                    else:
                        all_samples.extend([tuple(s.text.split()) for s in selected])

                    breakdown[key] = len(selected)
                else:
                    if return_sentences:
                        all_samples.append(current_level_data)
                    else:
                        all_samples.append(tuple(current_level_data.text.split()))
                    breakdown[key] = 1

        return all_samples, breakdown

    @staticmethod
    def _allocate_samples_evenly(groups: dict, total_samples: int) -> dict[str, int]:
        """Allocate samples evenly across groups, redistributing shortfalls.

        Args:
            groups: Dictionary mapping group names to their contents.
            total_samples: Total number of samples to allocate.

        Returns:
            Dictionary mapping group names to allocated sample counts.
        """
        group_names = sorted(groups.keys())
        allocated: dict[str, int] = {}
        remaining_quota = total_samples
        remaining_groups = list(group_names)

        while remaining_groups:
            n_remaining = len(remaining_groups)
            per_group = remaining_quota // n_remaining
            remainder = remaining_quota % n_remaining

            still_need_allocation = []
            for idx, group_name in enumerate(remaining_groups):
                group_size = BalancedSampler._get_group_size(groups[group_name])
                quota = per_group + (1 if idx < remainder else 0)

                if quota >= group_size:
                    allocated[group_name] = group_size
                    remaining_quota -= group_size
                else:
                    still_need_allocation.append(group_name)

            if len(still_need_allocation) == len(remaining_groups):
                for idx, group_name in enumerate(still_need_allocation):
                    quota = per_group + (1 if idx < remainder else 0)
                    allocated[group_name] = quota
                break

            remaining_groups = still_need_allocation

        return allocated

    @staticmethod
    def _get_group_size(group_data: Any) -> int:
        """Calculate the size of a group, handling nested structures.

        Args:
            group_data: A list, dict, or single sentence.

        Returns:
            The number of sentences in the group.
        """
        if isinstance(group_data, list):
            return len(group_data)
        elif isinstance(group_data, dict):
            return sum(BalancedSampler._get_group_size(v) for v in group_data.values())
        else:
            return 1
