"""Tests for BalancedSampler._allocate_tokens_evenly and _get_group_tokens."""

from __future__ import annotations

from sentence_corpora import Sentence
from sentence_corpora.sampling import BalancedSampler


class TestGetGroupTokens:
    """Tests for BalancedSampler._get_group_tokens."""

    def test_list(self) -> None:
        sentences = [
            Sentence(text="a b c", metadata={}),
            Sentence(text="d e", metadata={}),
        ]
        assert BalancedSampler._get_group_tokens(sentences) == 5

    def test_dict(self) -> None:
        data = {
            "a": [Sentence(text="x y", metadata={})],
            "b": [Sentence(text="z", metadata={})],
        }
        assert BalancedSampler._get_group_tokens(data) == 3

    def test_nested_dict(self) -> None:
        data = {
            "a": {
                "x": [Sentence(text="x y", metadata={})],
                "y": [Sentence(text="z", metadata={})],
            },
            "b": [Sentence(text="w", metadata={})],
        }
        assert BalancedSampler._get_group_tokens(data) == 4

    def test_single_sentence(self) -> None:
        s = Sentence(text="hello world", metadata={})
        assert BalancedSampler._get_group_tokens(s) == 2

    def test_empty_list(self) -> None:
        assert BalancedSampler._get_group_tokens([]) == 0

    def test_empty_dict(self) -> None:
        assert BalancedSampler._get_group_tokens({}) == 0

    def test_unknown_type(self) -> None:
        assert BalancedSampler._get_group_tokens(42) == 0  # type: ignore[arg-type]


class TestAllocateTokensEvenly:
    """Tests for BalancedSampler._allocate_tokens_evenly."""

    def test_even_split_equal_groups(self) -> None:
        """Equal token groups get equal allocation."""
        groups = {
            "a": [Sentence(text="x y", metadata={})],
            "b": [Sentence(text="x y", metadata={})],
        }
        result = BalancedSampler._allocate_tokens_evenly(groups, 4)
        assert result["a"] == 2
        assert result["b"] == 2

    def test_proportional_split(self) -> None:
        """Groups get allocation proportional to their token share, capped at available."""
        groups = {
            "a": [Sentence(text="x y z", metadata={})],  # 3 tokens
            "b": [Sentence(text="x", metadata={})],  # 1 token
        }
        result = BalancedSampler._allocate_tokens_evenly(groups, 8)
        # Total available is 4, so allocation is capped
        assert sum(result.values()) == 4
        assert result["a"] == 3
        assert result["b"] == 1

    def test_remainder_distribution(self) -> None:
        """When target exceeds total, allocation is capped at available."""
        groups = {
            "a": [Sentence(text="x", metadata={})],  # 1 token
            "b": [Sentence(text="x", metadata={})],  # 1 token
            "c": [Sentence(text="x", metadata={})],  # 1 token
        }
        result = BalancedSampler._allocate_tokens_evenly(groups, 5)
        # Total available is 3, so each group gets its 1 token
        assert result["a"] == 1
        assert result["b"] == 1
        assert result["c"] == 1
        assert sum(result.values()) == 3

    def test_request_more_than_total(self) -> None:
        """When target exceeds total available, allocation is capped at available."""
        groups = {
            "a": [Sentence(text="x y", metadata={})],  # 2 tokens
            "b": [Sentence(text="x", metadata={})],  # 1 token
        }
        result = BalancedSampler._allocate_tokens_evenly(groups, 1000)
        # Total available is 3, so allocation is capped
        assert sum(result.values()) == 3
        assert result["a"] == 2
        assert result["b"] == 1

    def test_nested_groups(self) -> None:
        """Nested dict groups: allocation based on recursive token count."""
        groups = {
            "a": {
                "x": [Sentence(text="x y z", metadata={})],
                "y": [Sentence(text="x", metadata={})],
            },  # 4 tokens total
            "b": [Sentence(text="x y", metadata={})],  # 2 tokens
        }
        result = BalancedSampler._allocate_tokens_evenly(groups, 6)
        assert result["a"] == 4
        assert result["b"] == 2

    def test_single_group(self) -> None:
        """Single group gets all tokens, capped at available."""
        groups = {"a": [Sentence(text="x y z", metadata={})]}
        result = BalancedSampler._allocate_tokens_evenly(groups, 10)
        # Only 3 tokens available, so capped
        assert result["a"] == 3

    def test_zero_total_tokens(self) -> None:
        """All groups empty → zero allocation."""
        groups: dict[str, list[Sentence]] = {"a": [], "b": []}
        result = BalancedSampler._allocate_tokens_evenly(groups, 10)
        assert result["a"] == 0
        assert result["b"] == 0

    def test_sum_equals_target(self) -> None:
        """Sum of allocations equals target when corpus has enough tokens."""
        groups = {
            "a": [Sentence(text="x y z", metadata={})],  # 3
            "b": [Sentence(text="x y", metadata={})],  # 2
            "c": [Sentence(text="x", metadata={})],  # 1
        }
        target = 6  # Exactly matches total available
        result = BalancedSampler._allocate_tokens_evenly(groups, target)
        assert sum(result.values()) == target

    def test_zero_target(self) -> None:
        """Zero target → all groups get 0."""
        groups = {
            "a": [Sentence(text="x y", metadata={})],
            "b": [Sentence(text="z", metadata={})],
        }
        result = BalancedSampler._allocate_tokens_evenly(groups, 0)
        assert result["a"] == 0
        assert result["b"] == 0
