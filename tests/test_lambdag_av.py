"""Tests for LambdaGAV class."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sentence_corpora import Sentence
from sentence_corpora.hierarchies import ThreeLevelCorpus
from sentence_corpora.lambdag.av import LambdaGAV


def _make_sentences(
    n: int, translator: str = "t1", author: str = "a1", work: str = "w1"
) -> list[Sentence]:
    """Create n sentences with given metadata."""
    return [
        Sentence(
            text=f"word{i} word{i + 1}",
            metadata={"work": work, "author": author, "translator": translator},
        )
        for i in range(n)
    ]


def _make_corpus() -> ThreeLevelCorpus:
    """Create a test corpus with two translators."""
    sentences = (
        _make_sentences(10, translator="known", author="a1", work="w1")
        + _make_sentences(10, translator="known", author="a2", work="w2")
        + _make_sentences(10, translator="other", author="a3", work="w3")
    )
    return ThreeLevelCorpus(sentences)


def _make_unknown_corpus() -> ThreeLevelCorpus:
    """Create a small unknown corpus."""
    return ThreeLevelCorpus(
        _make_sentences(5, translator="unknown", author="a1", work="w1")
    )


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


class TestFlattenBreakdownWorks:
    """Tests for _flatten_breakdown_works."""

    def test_flatten(self) -> None:
        breakdown = {
            "t1": {"a1": {"w1": 5, "w2": 3}},
            "t2": {"a2": {"w3": 7}},
        }
        result = LambdaGAV._flatten_breakdown_works(breakdown)
        assert result == {"w1": 5, "w2": 3, "w3": 7}


class TestRunSingleAvProblem:
    """Tests for run_single_av_problem with mocked LambdaGMethod."""

    @patch("sentence_corpora.lambdag.av.LambdaGMethod")
    def test_basic_run(self, mock_method_cls: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_instance.lambdag_score.return_value = 1.5
        mock_method_cls.return_value = mock_instance

        ref_corpus = _make_corpus()
        unknown_corpus = _make_unknown_corpus()

        av = LambdaGAV()
        result = av.run_single_av_problem(
            known_translator="known",
            unknown_corpus=unknown_corpus,
            reference_corpus=ref_corpus,
            known_size=5,
            reference_size=5,
            seed=42,
        )

        assert "error" not in result
        assert result["score"] == 1.5
        assert result["known_translator"] == "known"
        assert result["known_sentences_used"] == 5
        assert result["unknown_sentences_used"] == 5

    @patch("sentence_corpora.lambdag.av.LambdaGMethod")
    def test_unknown_translator(self, mock_method_cls: MagicMock) -> None:
        ref_corpus = _make_corpus()
        unknown_corpus = _make_unknown_corpus()

        av = LambdaGAV()
        result = av.run_single_av_problem(
            known_translator="nonexistent",
            unknown_corpus=unknown_corpus,
            reference_corpus=ref_corpus,
            known_size=5,
            reference_size=5,
            seed=42,
        )

        assert "error" in result
        assert "nonexistent" in str(result["error"])

    @patch("sentence_corpora.lambdag.av.LambdaGMethod")
    def test_result_keys(self, mock_method_cls: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_instance.lambdag_score.return_value = 2.0
        mock_method_cls.return_value = mock_instance

        ref_corpus = _make_corpus()
        unknown_corpus = _make_unknown_corpus()

        av = LambdaGAV()
        result = av.run_single_av_problem(
            known_translator="known",
            unknown_corpus=unknown_corpus,
            reference_corpus=ref_corpus,
            known_size=5,
            reference_size=5,
            seed=42,
        )

        expected_keys = {
            "score",
            "known_translator",
            "known_sentences_used",
            "reference_sentences_used",
            "unknown_sentences_used",
            "n_works_sampled",
            "n_ref_translators",
            "known_breakdown",
            "ref_breakdown",
            "corrected_score_sqrt",
            "corrected_score_hapax",
            "post_test_likelihood",
            "unknown_tokens_used",
            "hapax_tokens",
            "hapax_ratio",
        }
        assert set(result.keys()) == expected_keys

    @patch("sentence_corpora.lambdag.av.LambdaGMethod")
    def test_custom_order_and_references(self, mock_method_cls: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_instance.lambdag_score.return_value = 0.5
        mock_method_cls.return_value = mock_instance

        ref_corpus = _make_corpus()
        unknown_corpus = _make_unknown_corpus()

        av = LambdaGAV()
        av.run_single_av_problem(
            known_translator="known",
            unknown_corpus=unknown_corpus,
            reference_corpus=ref_corpus,
            known_size=5,
            reference_size=5,
            order=3,
            num_references=5,
            seed=42,
        )

        mock_method_cls.assert_called_once_with(
            basis="tokens",
            order=3,
            smoothing="kneser_ney",
            lowercasing=False,
            sentenize=False,
            num_references=5,
        )

    @patch("sentence_corpora.lambdag.av.LambdaGMethod")
    def test_rng_overrides_seed(self, mock_method_cls: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_instance.lambdag_score.return_value = 1.0
        mock_method_cls.return_value = mock_instance

        ref_corpus = _make_corpus()
        unknown_corpus = _make_unknown_corpus()

        av = LambdaGAV(seed=42)
        custom_rng = np.random.default_rng(99)
        av.run_single_av_problem(
            known_translator="known",
            unknown_corpus=unknown_corpus,
            reference_corpus=ref_corpus,
            known_size=5,
            reference_size=5,
            rng=custom_rng,
        )

        mock_instance.random_gen = custom_rng
