"""Tests for BalancedSampler class."""

from __future__ import annotations

import numpy as np

from sentence_corpora import Sentence
from sentence_corpora.sampling import BalancedSampler


def _sentences() -> list[Sentence]:
    """Return a standard set of test sentences."""
    return [
        Sentence(
            text="t1 t2",
            metadata={"work": "w1", "author": "a1", "translator": "tr1"},
        ),
        Sentence(
            text="t3 t4 t5",
            metadata={"work": "w1", "author": "a1", "translator": "tr1"},
        ),
        Sentence(
            text="t6",
            metadata={"work": "w2", "author": "a2", "translator": "tr1"},
        ),
        Sentence(
            text="t7 t8",
            metadata={"work": "w2", "author": "a2", "translator": "tr2"},
        ),
        Sentence(
            text="t9 t10 t11",
            metadata={"work": "w3", "author": "a3", "translator": "tr2"},
        ),
    ]


class TestGroupByLevels:
    """Tests for BalancedSampler.group_by_levels."""

    def test_single_level(self) -> None:
        sentences = _sentences()
        grouped = BalancedSampler.group_by_levels(sentences, ["translator"])
        assert set(grouped.keys()) == {"tr1", "tr2"}
        assert len(grouped["tr1"]) == 3
        assert len(grouped["tr2"]) == 2

    def test_two_levels(self) -> None:
        sentences = _sentences()
        grouped = BalancedSampler.group_by_levels(sentences, ["translator", "work"])
        assert set(grouped.keys()) == {"tr1", "tr2"}
        assert set(grouped["tr1"].keys()) == {"w1", "w2"}
        assert len(grouped["tr1"]["w1"]) == 2
        assert len(grouped["tr1"]["w2"]) == 1

    def test_three_levels(self) -> None:
        sentences = _sentences()
        grouped = BalancedSampler.group_by_levels(
            sentences, ["translator", "author", "work"]
        )
        assert set(grouped.keys()) == {"tr1", "tr2"}
        assert set(grouped["tr1"].keys()) == {"a1", "a2"}
        assert set(grouped["tr1"]["a1"].keys()) == {"w1"}
        assert len(grouped["tr1"]["a1"]["w1"]) == 2

    def test_empty_levels_returns_empty_dict(self) -> None:
        sentences = _sentences()
        result = BalancedSampler.group_by_levels(sentences, [])
        assert result == {}

    def test_empty_sentence_list(self) -> None:
        grouped = BalancedSampler.group_by_levels([], ["translator"])
        assert grouped == {}


class TestSampleBalanced:
    """Tests for BalancedSampler.sample_balanced."""

    def test_basic_sampling(self) -> None:
        sentences = _sentences()
        grouped = BalancedSampler.group_by_levels(sentences, ["translator", "work"])
        rng = np.random.default_rng(42)
        samples, breakdown = BalancedSampler.sample_balanced(
            grouped, ["translator", "work"], 4, rng
        )
        assert len(samples) == 4
        assert all(isinstance(s, Sentence) for s in samples)

    def test_request_more_than_available(self) -> None:
        sentences = _sentences()
        grouped = BalancedSampler.group_by_levels(sentences, ["translator"])
        rng = np.random.default_rng(42)
        samples, breakdown = BalancedSampler.sample_balanced(
            grouped, ["translator"], 100, rng
        )
        assert len(samples) == len(sentences)

    def test_zero_samples(self) -> None:
        sentences = _sentences()
        grouped = BalancedSampler.group_by_levels(sentences, ["translator"])
        rng = np.random.default_rng(42)
        samples, breakdown = BalancedSampler.sample_balanced(
            grouped, ["translator"], 0, rng
        )
        assert len(samples) == 0

    def test_return_token_tuples(self) -> None:
        sentences = _sentences()
        grouped = BalancedSampler.group_by_levels(sentences, ["translator"])
        rng = np.random.default_rng(42)
        samples, breakdown = BalancedSampler.sample_balanced(
            grouped, ["translator"], 3, rng, return_sentences=False
        )
        assert len(samples) == 3
        assert all(isinstance(s, tuple) for s in samples)

    def test_breakdown_structure(self) -> None:
        sentences = _sentences()
        grouped = BalancedSampler.group_by_levels(sentences, ["translator", "work"])
        rng = np.random.default_rng(42)
        samples, breakdown = BalancedSampler.sample_balanced(
            grouped, ["translator", "work"], 4, rng
        )
        assert "tr1" in breakdown
        assert "tr2" in breakdown

    def test_single_sentence(self) -> None:
        sentences = [
            Sentence(
                text="only one",
                metadata={"work": "w1", "author": "a1", "translator": "tr1"},
            )
        ]
        grouped = BalancedSampler.group_by_levels(
            sentences, ["translator", "author", "work"]
        )
        rng = np.random.default_rng(42)
        samples, breakdown = BalancedSampler.sample_balanced(
            grouped, ["translator", "author", "work"], 1, rng
        )
        assert len(samples) == 1
        assert samples[0].text == "only one"


class TestAllocateSamplesEvenly:
    """Tests for BalancedSampler._allocate_samples_evenly."""

    def test_even_split(self) -> None:
        groups = {"a": [1, 2, 3], "b": [4, 5, 6]}
        result = BalancedSampler._allocate_samples_evenly(groups, 4)
        assert result["a"] == 2
        assert result["b"] == 2

    def test_uneven_split(self) -> None:
        groups = {"a": [1, 2, 3], "b": [4, 5, 6]}
        result = BalancedSampler._allocate_samples_evenly(groups, 5)
        assert sum(result.values()) == 5

    def test_group_smaller_than_quota(self) -> None:
        groups = {"a": [1], "b": [4, 5, 6, 7, 8]}
        result = BalancedSampler._allocate_samples_evenly(groups, 4)
        assert result["a"] == 1
        assert result["b"] == 3

    def test_request_more_than_total(self) -> None:
        groups = {"a": [1, 2], "b": [3, 4]}
        result = BalancedSampler._allocate_samples_evenly(groups, 100)
        assert result["a"] == 2
        assert result["b"] == 2

    def test_nested_groups(self) -> None:
        groups = {
            "a": {"x": [1, 2], "y": [3, 4]},
            "b": [5, 6, 7, 8],
        }
        result = BalancedSampler._allocate_samples_evenly(groups, 4)
        assert result["a"] == 2
        assert result["b"] == 2


class TestGetGroupSize:
    """Tests for BalancedSampler._get_group_size."""

    def test_list(self) -> None:
        assert BalancedSampler._get_group_size([1, 2, 3]) == 3

    def test_dict(self) -> None:
        assert BalancedSampler._get_group_size({"a": [1, 2], "b": [3]}) == 3

    def test_nested_dict(self) -> None:
        data = {"a": {"x": [1, 2], "y": [3]}, "b": [4]}
        assert BalancedSampler._get_group_size(data) == 4

    def test_single_item(self) -> None:
        assert BalancedSampler._get_group_size("single") == 1

    def test_empty_list(self) -> None:
        assert BalancedSampler._get_group_size([]) == 0
