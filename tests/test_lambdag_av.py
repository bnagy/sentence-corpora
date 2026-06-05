"""Tests for LambdaGAV class."""

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


class TestEndToEndSampling:
    """End-to-end tests using real LambdaGMethod with small corpora.

    These tests catch regressions in sample_balanced where the number of
    sampled sentences drops dramatically. The root cause is that
    _allocate_tokens_evenly allocates proportionally across groups without
    capping at available tokens, so when the corpus is small relative to
    the target, most groups can't fulfill their allocation.
    """

    @staticmethod
    def _make_realistic_sentences(
        n: int, translator: str, author: str, work: str
    ) -> list[Sentence]:
        """Create sentences with realistic Latin-like POS-tagged text."""
        base_texts = [
            "NOUN VERB ADJ NOUN ADP NOUN",
            "PRON VERB NOUN ADJ CONJ NOUN",
            "DET NOUN VERB ADV ADP DET NOUN",
            "NOUN NOUN VERB ADJ CONJ VERB",
            "ADP DET NOUN VERB NOUN ADJ",
        ]
        sentences = []
        for i in range(n):
            text = base_texts[i % len(base_texts)]
            sentences.append(
                Sentence(
                    text=text,
                    metadata={
                        "translator": translator,
                        "author": author,
                        "work": work,
                    },
                )
            )
        return sentences

    def test_sample_balanced_accumulates_requested_tokens(self) -> None:
        """sample_balanced should accumulate at least target_tokens tokens.

        Regression: _allocate_tokens_evenly allocates proportionally across
        groups without capping at available tokens. When corpus is small
        relative to target, groups can't fulfill their allocation, resulting
        in far fewer tokens than requested.
        """
        sentences = []
        for t_idx, translator in enumerate(["t1", "t2", "t3"]):
            for w_idx in range(3):
                work = f"work_{w_idx}"
                author = f"author_{t_idx}"
                sentences.extend(
                    self._make_realistic_sentences(
                        50, translator, author, work
                    )
                )
        corpus = ThreeLevelCorpus(
            sentences, levels=("translator", "author", "work")
        )

        rng = np.random.default_rng(42)

        # Sample with a token target. The corpus has ~450 sentences × 6 tokens
        # = ~2700 tokens total. Requesting 2000 tokens should yield a large
        # fraction of the corpus.
        selected, breakdown = corpus.sample_balanced(tokens=2000, rng=rng)

        total_tokens = sum(len(s.text.split()) for s in selected)

        # Should have accumulated at least the requested tokens
        # (unless corpus is smaller than target)
        corpus_total = sum(len(s.text.split()) for s in corpus)
        assert total_tokens <= corpus_total, "Sampled more tokens than available"

        # The key regression check: if the corpus has enough tokens,
        # sample_balanced should return at least target_tokens worth.
        if corpus_total >= 2000:
            assert total_tokens >= 2000, (
                f"sample_balanced returned only {total_tokens} tokens, "
                f"expected >= 2000. Selected {len(selected)} sentences "
                f"from corpus of {len(corpus)} sentences ({corpus_total} tokens)."
            )

    def test_sample_balanced_with_large_target_returns_all(self) -> None:
        """When target exceeds available tokens, all sentences should be returned."""
        sentences = self._make_realistic_sentences(10, "t1", "a1", "w1")
        corpus = ThreeLevelCorpus(
            sentences, levels=("translator", "author", "work")
        )

        rng = np.random.default_rng(42)
        selected, breakdown = corpus.sample_balanced(tokens=999999, rng=rng)

        assert len(selected) == 10
        assert sum(len(s.text.split()) for s in selected) == sum(
            len(s.text.split()) for s in corpus
        )

    def test_three_level_correct_run_produces_positive_score(self) -> None:
        """A correct AV run (known translator matches question) should produce
        a positive LambdaG score."""
        sentences = []
        for t_idx, translator in enumerate(["known", "ref_a", "ref_b"]):
            for w_idx in range(3):
                work = f"work_{w_idx}"
                author = f"author_{t_idx}"
                sentences.extend(
                    self._make_realistic_sentences(
                        20, translator, author, work
                    )
                )
        kr_corpus = ThreeLevelCorpus(
            sentences, levels=("translator", "author", "work")
        )

        # Question from the "known" translator
        q_sentences = self._make_realistic_sentences(
            5, "known", "author_0", "work_0"
        )
        q_corpus = ThreeLevelCorpus(
            q_sentences, levels=("translator", "author", "work")
        )

        av = LambdaGAV()
        result = av.run_single_av_problem(
            known_entity="known",
            q_corpus=q_corpus,
            kr_corpus=kr_corpus,
            known_tokens=2000,
            reference_tokens=2000,
            seed=42,
        )

        assert "error" not in result
        # Correct match should produce a positive score
        assert result["score"] > 0.0, (
            f"Expected positive score for correct match, got {result['score']}"
        )


class TestExactScoreRegression:
    """Exact-score regression tests with pinned expected values.

    These tests use real sentences from the nile train corpus (with generic
    metadata) and pin the exact LambdaG score. Any change to the sampling
    algorithm, LambdaGMethod parameters, or corpus construction will cause a
    failure. This catches regressions like the sample_balanced token
    allocation bug where scores changed dramatically.
    """

    @staticmethod
    def _make_fixture_corpus() -> tuple[ThreeLevelCorpus, ThreeLevelCorpus]:
        """Create a deterministic fixture from real nile train corpus sentences.

        Uses 450 real sentences with generic metadata (to avoid copyright):
        - 3 translators x 3 works x 50 sentences each
        - q_corpus: 5 sentences from an unknown translator

        Sentences are selected with a fixed seed (99) from the nile train_corpus.
        """
        import pickle
        with open("/Users/ben/code/nile/train_corpus.pkl", "rb") as f:
            train = pickle.load(f)

        all_sentences = list(train)
        rng = np.random.default_rng(99)
        rng.shuffle(all_sentences)

        translator_names = ["translator_A", "translator_B", "translator_C"]
        work_names = ["work_1", "work_2", "work_3"]
        sentences_per_work = 50

        kr_sentences = []
        idx = 0
        for t_name in translator_names:
            for w_name in work_names:
                for _ in range(sentences_per_work):
                    s = all_sentences[idx]
                    kr_sentences.append(
                        Sentence(
                            text=s.text,
                            metadata={
                                "translator": t_name,
                                "author": f"author_{t_name}",
                                "work": w_name,
                            },
                        )
                    )
                    idx += 1

        kr_corpus = ThreeLevelCorpus(
            kr_sentences, levels=("translator", "author", "work")
        )

        q_sentences = []
        for i in range(5):
            s = all_sentences[idx + i]
            q_sentences.append(
                Sentence(
                    text=s.text,
                    metadata={
                        "translator": "unknown",
                        "author": "author_unknown",
                        "work": "work_unknown",
                    },
                )
            )
        q_corpus = ThreeLevelCorpus(
            q_sentences, levels=("translator", "author", "work")
        )

        return kr_corpus, q_corpus

    def test_exact_score_correct_match(self) -> None:
        """Pin the exact LambdaG score for a correct translator match.

        Fixture: 450 real sentences, 3 translators x 3 works x 50 sentences
        Question: 5 real sentences from unknown translator
        Parameters: known_tokens=500, reference_tokens=5000, seed=42

        Expected score: 13.243315 (pinned on 2026-06-04)
        """
        kr_corpus, q_corpus = self._make_fixture_corpus()

        av = LambdaGAV()
        result = av.run_single_av_problem(
            known_entity="translator_A",
            q_corpus=q_corpus,
            kr_corpus=kr_corpus,
            known_tokens=500,
            reference_tokens=5000,
            seed=42,
        )

        assert "error" not in result
        expected_score = 13.243315
        assert abs(result["score"] - expected_score) < 0.001, (
            f"Score regression: expected {expected_score}, got {result['score']}. "
            f"This indicates a change in sample_balanced allocation or LambdaGMethod."
        )

    def test_exact_score_incorrect_match(self) -> None:
        """Pin the exact LambdaG score for an incorrect translator match.

        Same fixture but known_entity="translator_B" while question is from
        unknown translator. Should produce a lower score since the known
        translator does not match the question.

        Expected score: 0.465066 (pinned on 2026-06-04)
        """
        kr_corpus, q_corpus = self._make_fixture_corpus()

        av = LambdaGAV()
        result = av.run_single_av_problem(
            known_entity="translator_B",
            q_corpus=q_corpus,
            kr_corpus=kr_corpus,
            known_tokens=500,
            reference_tokens=5000,
            seed=42,
        )

        assert "error" not in result
        expected_score = 0.465066
        assert abs(result["score"] - expected_score) < 0.001, (
            f"Score regression: expected {expected_score}, got {result['score']}. "
            f"This indicates a change in sample_balanced allocation or LambdaGMethod."
        )
