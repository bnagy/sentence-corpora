"""Tests for LambdaGAV class — init, utilities, and protocol."""

from __future__ import annotations

from sentence_corpora.lambdag.av import LambdaGAV


class TestLambdaGAVInit:
    """Tests for LambdaGAV initialization."""

    def test_defaults(self) -> None:
        av = LambdaGAV()
        assert av.order == 4
        assert av.num_references == 10
        assert av.seed == 42

    def test_custom(self) -> None:
        av = LambdaGAV(order=3, num_references=5, seed=123)
        assert av.order == 3
        assert av.num_references == 5
        assert av.seed == 123


class TestLlrToPercentProbability:
    """Tests for llr_to_percent_probability."""

    def test_neutral_prior(self) -> None:
        result = LambdaGAV.llr_to_percent_probability(0.0)
        assert result == 50.0

    def test_positive_llr(self) -> None:
        result = LambdaGAV.llr_to_percent_probability(1.0)
        assert result > 50.0

    def test_negative_llr(self) -> None:
        result = LambdaGAV.llr_to_percent_probability(-1.0)
        assert result < 50.0

    def test_pre_test_prob_one(self) -> None:
        result = LambdaGAV.llr_to_percent_probability(10.0, pre_test_prob=1.0)
        assert result == 100.0

    def test_pre_test_prob_zero(self) -> None:
        result = LambdaGAV.llr_to_percent_probability(10.0, pre_test_prob=0.0)
        assert result == 0.0


class TestFlattenBreakdown:
    """Tests for _flatten_breakdown."""

    def test_flatten_three_level(self) -> None:
        """Flatten a translator→author→work breakdown."""
        breakdown = {
            "t1": {"a1": {"w1": 5, "w2": 3}},
            "t2": {"a2": {"w3": 7}},
        }
        result = LambdaGAV._flatten_breakdown(breakdown)
        assert result == {"w1": 5, "w2": 3, "w3": 7}

    def test_flatten_two_level(self) -> None:
        """Flatten an author→work breakdown."""
        breakdown = {
            "a1": {"w1": 5, "w2": 3},
            "a2": {"w3": 7},
        }
        result = LambdaGAV._flatten_breakdown(breakdown)
        assert result == {"w1": 5, "w2": 3, "w3": 7}

    def test_flatten_already_flat(self) -> None:
        """Flat dict passes through unchanged."""
        breakdown = {"w1": 5, "w2": 3}
        result = LambdaGAV._flatten_breakdown(breakdown)
        assert result == {"w1": 5, "w2": 3}

    def test_flatten_empty(self) -> None:
        result = LambdaGAV._flatten_breakdown({})
        assert result == {}
