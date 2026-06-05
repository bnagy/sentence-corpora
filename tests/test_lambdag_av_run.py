"""Tests for LambdaGAV.run_single_av_problem with mocked LambdaGMethod."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from sentence_corpora import Sentence
from sentence_corpora.hierarchies import ThreeLevelCorpus, TwoLevelCorpus
from sentence_corpora.lambdag.av import LambdaGAV


# ---------------------------------------------------------------------------
# Helpers — three-level (translator/author/work)
# ---------------------------------------------------------------------------


def _make_sentences_3level(
    n: int, translator: str = "t1", author: str = "a1", work: str = "w1"
) -> list[Sentence]:
    """Create n sentences with three-level metadata."""
    return [
        Sentence(
            text=f"word{i} word{i + 1}",
            metadata={"work": work, "author": author, "translator": translator},
        )
        for i in range(n)
    ]


def _make_corpus_3level() -> ThreeLevelCorpus:
    """Create a three-level test corpus with two translators."""
    sentences = (
        _make_sentences_3level(10, translator="known", author="a1", work="w1")
        + _make_sentences_3level(10, translator="known", author="a2", work="w2")
        + _make_sentences_3level(10, translator="other", author="a3", work="w3")
    )
    return ThreeLevelCorpus(sentences, levels=("translator", "author", "work"))


def _make_q_corpus_3level() -> ThreeLevelCorpus:
    """Create a small three-level q_corpus."""
    return ThreeLevelCorpus(
        _make_sentences_3level(5, translator="unknown", author="a1", work="w1"),
        levels=("translator", "author", "work"),
    )


# ---------------------------------------------------------------------------
# Helpers — two-level (work/author)
# ---------------------------------------------------------------------------


def _make_sentences_2level(
    n: int, author: str = "a1", work: str = "w1"
) -> list[Sentence]:
    """Create n sentences with two-level metadata."""
    return [
        Sentence(
            text=f"word{i} word{i + 1}",
            metadata={"work": work, "author": author},
        )
        for i in range(n)
    ]


def _make_corpus_2level() -> TwoLevelCorpus:
    """Create a two-level test corpus with two authors."""
    sentences = (
        _make_sentences_2level(10, author="known", work="w1")
        + _make_sentences_2level(10, author="known", work="w2")
        + _make_sentences_2level(10, author="other", work="w3")
    )
    return TwoLevelCorpus(sentences, levels=("author", "work"))


def _make_q_corpus_2level() -> TwoLevelCorpus:
    """Create a small two-level q_corpus."""
    return TwoLevelCorpus(
        _make_sentences_2level(5, author="unknown", work="w1"),
        levels=("author", "work"),
    )


class TestRunSingleAvProblemThreeLevel:
    """Tests for run_single_av_problem with ThreeLevelCorpus (translator verification)."""

    @patch("sentence_corpora.lambdag.av.LambdaGMethod")
    def test_basic_run(self, mock_method_cls: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_instance.lambdag_score.return_value = 1.5
        mock_method_cls.return_value = mock_instance

        kr_corpus = _make_corpus_3level()
        q_corpus = _make_q_corpus_3level()

        av = LambdaGAV()
        result = av.run_single_av_problem(
            known_entity="known",
            q_corpus=q_corpus,
            kr_corpus=kr_corpus,
            # Request enough tokens for ~5 sentences (each is 2 tokens)
            known_tokens=12,
            reference_tokens=12,
            seed=42,
        )

        assert "error" not in result
        assert result["score"] == 1.5
        assert result["known_entity"] == "known"
        assert result["verification_level"] == "translator"
        # 12 tokens requested, 2 tokens per sentence = ~6 sentences
        assert result["known_sentences_used"] >= 5
        assert result["unknown_sentences_used"] == 5

    @patch("sentence_corpora.lambdag.av.LambdaGMethod")
    def test_unknown_entity(self, mock_method_cls: MagicMock) -> None:
        kr_corpus = _make_corpus_3level()
        q_corpus = _make_q_corpus_3level()

        av = LambdaGAV()
        result = av.run_single_av_problem(
            known_entity="nonexistent",
            q_corpus=q_corpus,
            kr_corpus=kr_corpus,
            known_tokens=5,
            reference_tokens=5,
            seed=42,
        )

        assert "error" in result
        assert "nonexistent" in str(result["error"])
        assert "translator" in str(result["error"])

    @patch("sentence_corpora.lambdag.av.LambdaGMethod")
    def test_result_keys(self, mock_method_cls: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_instance.lambdag_score.return_value = 2.0
        mock_method_cls.return_value = mock_instance

        kr_corpus = _make_corpus_3level()
        q_corpus = _make_q_corpus_3level()

        av = LambdaGAV()
        result = av.run_single_av_problem(
            known_entity="known",
            q_corpus=q_corpus,
            kr_corpus=kr_corpus,
            known_tokens=5,
            reference_tokens=5,
            seed=42,
        )

        expected_keys = {
            "score",
            "known_entity",
            "verification_level",
            "known_sentences_used",
            "reference_sentences_used",
            "unknown_sentences_used",
            "n_works_sampled",
            "n_ref_entities",
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

        kr_corpus = _make_corpus_3level()
        q_corpus = _make_q_corpus_3level()

        av = LambdaGAV()
        av.run_single_av_problem(
            known_entity="known",
            q_corpus=q_corpus,
            kr_corpus=kr_corpus,
            known_tokens=5,
            reference_tokens=5,
            order=3,
            num_references=5,
            seed=42,
        )

        mock_method_cls.assert_called_once()
        call_kwargs = mock_method_cls.call_args.kwargs
        assert call_kwargs["order"] == 3
        assert call_kwargs["num_references"] == 5

    @patch("sentence_corpora.lambdag.av.LambdaGMethod")
    def test_n_ref_entities(self, mock_method_cls: MagicMock) -> None:
        """n_ref_entities should count reference entities at the verification level."""
        mock_instance = MagicMock()
        mock_instance.lambdag_score.return_value = 1.0
        mock_method_cls.return_value = mock_instance

        kr_corpus = _make_corpus_3level()
        q_corpus = _make_q_corpus_3level()

        av = LambdaGAV()
        result = av.run_single_av_problem(
            known_entity="known",
            q_corpus=q_corpus,
            kr_corpus=kr_corpus,
            known_tokens=5,
            reference_tokens=5,
            seed=42,
        )

        # Reference corpus excludes "known", so only "other" remains
        assert result["n_ref_entities"] == 1


class TestRunSingleAvProblemTwoLevel:
    """Tests for run_single_av_problem with TwoLevelCorpus (author verification)."""

    @patch("sentence_corpora.lambdag.av.LambdaGMethod")
    def test_basic_run(self, mock_method_cls: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_instance.lambdag_score.return_value = 2.0
        mock_method_cls.return_value = mock_instance

        kr_corpus = _make_corpus_2level()
        q_corpus = _make_q_corpus_2level()

        av = LambdaGAV()
        result = av.run_single_av_problem(
            known_entity="known",
            q_corpus=q_corpus,
            kr_corpus=kr_corpus,
            # Request enough tokens for ~5 sentences (each is 2 tokens)
            known_tokens=12,
            reference_tokens=12,
            seed=42,
        )

        assert "error" not in result
        assert result["score"] == 2.0
        assert result["known_entity"] == "known"
        assert result["verification_level"] == "author"
        # 12 tokens requested, 2 tokens per sentence = ~6 sentences
        assert result["known_sentences_used"] >= 5
        assert result["unknown_sentences_used"] == 5

    @patch("sentence_corpora.lambdag.av.LambdaGMethod")
    def test_unknown_entity(self, mock_method_cls: MagicMock) -> None:
        kr_corpus = _make_corpus_2level()
        q_corpus = _make_q_corpus_2level()

        av = LambdaGAV()
        result = av.run_single_av_problem(
            known_entity="nonexistent",
            q_corpus=q_corpus,
            kr_corpus=kr_corpus,
            known_tokens=5,
            reference_tokens=5,
            seed=42,
        )

        assert "error" in result
        assert "nonexistent" in str(result["error"])
        assert "author" in str(result["error"])

    @patch("sentence_corpora.lambdag.av.LambdaGMethod")
    def test_result_keys(self, mock_method_cls: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_instance.lambdag_score.return_value = 1.0
        mock_method_cls.return_value = mock_instance

        kr_corpus = _make_corpus_2level()
        q_corpus = _make_q_corpus_2level()

        av = LambdaGAV()
        result = av.run_single_av_problem(
            known_entity="known",
            q_corpus=q_corpus,
            kr_corpus=kr_corpus,
            known_tokens=5,
            reference_tokens=5,
            seed=42,
        )

        expected_keys = {
            "score",
            "known_entity",
            "verification_level",
            "known_sentences_used",
            "reference_sentences_used",
            "unknown_sentences_used",
            "n_works_sampled",
            "n_ref_entities",
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
    def test_n_ref_entities(self, mock_method_cls: MagicMock) -> None:
        """n_ref_entities should count reference entities at the verification level (author)."""
        mock_instance = MagicMock()
        mock_instance.lambdag_score.return_value = 1.0
        mock_method_cls.return_value = mock_instance

        kr_corpus = _make_corpus_2level()
        q_corpus = _make_q_corpus_2level()

        av = LambdaGAV()
        result = av.run_single_av_problem(
            known_entity="known",
            q_corpus=q_corpus,
            kr_corpus=kr_corpus,
            known_tokens=5,
            reference_tokens=5,
            seed=42,
        )

        # Reference corpus excludes "known", so only "other" remains
        assert result["n_ref_entities"] == 1

    @patch("sentence_corpora.lambdag.av.LambdaGMethod")
    def test_corrected_scores_with_zero_tokens(
        self, mock_method_cls: MagicMock
    ) -> None:
        """Corrected scores should handle empty q_corpus gracefully."""
        mock_instance = MagicMock()
        mock_instance.lambdag_score.return_value = 0.0
        mock_method_cls.return_value = mock_instance

        kr_corpus = _make_corpus_2level()
        q_corpus = TwoLevelCorpus([], levels=("author", "work"))

        av = LambdaGAV()
        result = av.run_single_av_problem(
            known_entity="known",
            q_corpus=q_corpus,
            kr_corpus=kr_corpus,
            known_tokens=5,
            reference_tokens=5,
            seed=42,
        )

        assert result["score"] == 0.0
        assert result["corrected_score_sqrt"] == 0.0
        assert result["corrected_score_hapax"] == 0.0
        assert result["hapax_ratio"] == 0.0

    @patch("sentence_corpora.lambdag.av.LambdaGMethod")
    def test_rng_overrides_seed(self, mock_method_cls: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_instance.lambdag_score.return_value = 1.0
        mock_method_cls.return_value = mock_instance

        kr_corpus = _make_corpus_2level()
        q_corpus = _make_q_corpus_2level()

        av = LambdaGAV(seed=42)
        custom_rng = np.random.default_rng(99)
        result = av.run_single_av_problem(
            known_entity="known",
            q_corpus=q_corpus,
            kr_corpus=kr_corpus,
            known_tokens=5,
            reference_tokens=5,
            rng=custom_rng,
        )

        assert "error" not in result
        assert mock_instance.random_gen is custom_rng
