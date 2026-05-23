"""Tests for BalancedSampler._sample_sentences_greedy."""

from __future__ import annotations

import numpy as np

from sentence_corpora import Sentence
from sentence_corpora.sampling import BalancedSampler

from .sampling_fixture import sentence_tokens, total_tokens


class TestSampleSentencesGreedy:
    """Tests for BalancedSampler._sample_sentences_greedy."""

    def test_empty_list(self) -> None:
        rng = np.random.default_rng(42)
        selected, tokens = BalancedSampler._sample_sentences_greedy([], 10, rng)
        assert selected == []
        assert tokens == 0

    def test_target_zero(self) -> None:
        sentences = [Sentence(text="a b c", metadata={})]
        rng = np.random.default_rng(42)
        selected, tokens = BalancedSampler._sample_sentences_greedy(sentences, 0, rng)
        assert selected == []
        assert tokens == 0

    def test_all_sentences_returned_when_not_enough(self) -> None:
        sentences = [Sentence(text="a b", metadata={})]  # 2 tokens
        rng = np.random.default_rng(42)
        selected, tokens = BalancedSampler._sample_sentences_greedy(sentences, 10, rng)
        assert len(selected) == 1
        assert tokens == 2

    def test_tokens_ge_target(self) -> None:
        sentences = [
            Sentence(text="a b", metadata={}),
            Sentence(text="c d e", metadata={}),
            Sentence(text="f", metadata={}),
        ]
        rng = np.random.default_rng(42)
        target = 3
        selected, tokens = BalancedSampler._sample_sentences_greedy(
            sentences, target, rng
        )
        assert tokens >= target

    def test_deterministic(self) -> None:
        sentences = [
            Sentence(text="a b c", metadata={}),
            Sentence(text="d e", metadata={}),
            Sentence(text="f g h i", metadata={}),
        ]
        rng1 = np.random.default_rng(42)
        sel1, tok1 = BalancedSampler._sample_sentences_greedy(sentences, 4, rng1)
        rng2 = np.random.default_rng(42)
        sel2, tok2 = BalancedSampler._sample_sentences_greedy(sentences, 4, rng2)
        assert [s.text for s in sel1] == [s.text for s in sel2]
        assert tok1 == tok2

    def test_overshoot_bounded(self) -> None:
        """Overshoot should be less than the largest sentence."""
        sentences = [
            Sentence(text="a b c d e f g", metadata={}),  # 7 tokens
            Sentence(text="h i", metadata={}),  # 2 tokens
            Sentence(text="j k l m n o p q", metadata={}),  # 8 tokens
        ]
        rng = np.random.default_rng(42)
        target = 5
        selected, tokens = BalancedSampler._sample_sentences_greedy(
            sentences, target, rng
        )
        max_sent = max(sentence_tokens(s) for s in sentences)  # 8
        assert tokens >= target
        assert tokens < target + max_sent

    def test_single_sentence_below_target(self) -> None:
        """Single sentence with fewer tokens than target returns that sentence."""
        sentences = [Sentence(text="x y", metadata={})]  # 2 tokens
        rng = np.random.default_rng(42)
        selected, tokens = BalancedSampler._sample_sentences_greedy(sentences, 5, rng)
        assert len(selected) == 1
        assert tokens == 2

    def test_exact_match_possible(self) -> None:
        """When sentences exactly match target, no overshoot."""
        sentences = [
            Sentence(text="a b", metadata={}),  # 2
            Sentence(text="c d", metadata={}),  # 2
        ]
        rng = np.random.default_rng(42)
        selected, tokens = BalancedSampler._sample_sentences_greedy(sentences, 4, rng)
        assert tokens == 4
        assert len(selected) == 2

    def test_varying_lengths_realistic(self) -> None:
        """Test with realistically varying sentence lengths (6–16 tokens)."""
        sentences = [
            Sentence(
                text=" ".join(chr(ord("a") + (i % 26)) for i in range(n)), metadata={}
            )
            for n in [8, 12, 7, 15, 10, 9, 11, 6, 14, 16, 13]
        ]
        rng = np.random.default_rng(42)
        target = 30
        selected, tokens = BalancedSampler._sample_sentences_greedy(
            sentences, target, rng
        )
        assert tokens >= target
        # Verify all selected sentences are from the original pool
        original_texts = {s.text for s in sentences}
        for s in selected:
            assert s.text in original_texts
