"""End-to-end and regression tests for LambdaGAV.

These tests use real LambdaGMethod (not mocked) to catch regressions
in sampling, scoring, and corpus construction.
"""

from __future__ import annotations

import numpy as np

from sentence_corpora import Sentence
from sentence_corpora.hierarchies import ThreeLevelCorpus
from sentence_corpora.lambdag.av import LambdaGAV


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
