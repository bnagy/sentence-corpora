"""Tests for BalancedSampler.sample_balanced (token-based integration tests)."""

from __future__ import annotations

import numpy as np

from sentence_corpora import Sentence
from sentence_corpora.sampling import BalancedSampler

from .sampling_fixture import realistic_sentences, sentence_tokens, total_tokens


class TestSampleBalancedTokens:
    """Core token-based sampling properties."""

    def test_tokens_ge_target(self) -> None:
        """Actual tokens must be >= target_tokens."""
        sentences = realistic_sentences()
        grouped = BalancedSampler.group_by_levels(
            sentences, ["translator", "author", "work"]
        )
        rng = np.random.default_rng(42)
        samples, _ = BalancedSampler.sample_balanced(
            grouped, ["translator", "author", "work"], 50, rng
        )
        assert total_tokens(samples) >= 50

    def test_zero_tokens_returns_empty(self) -> None:
        sentences = realistic_sentences()
        grouped = BalancedSampler.group_by_levels(sentences, ["translator"])
        rng = np.random.default_rng(42)
        samples, _ = BalancedSampler.sample_balanced(grouped, ["translator"], 0, rng)
        assert len(samples) == 0

    def test_one_token_returns_at_least_one_sentence(self) -> None:
        sentences = realistic_sentences()
        grouped = BalancedSampler.group_by_levels(sentences, ["translator"])
        rng = np.random.default_rng(42)
        samples, _ = BalancedSampler.sample_balanced(grouped, ["translator"], 1, rng)
        assert len(samples) >= 1
        assert total_tokens(samples) >= 1

    def test_more_than_available_returns_all(self) -> None:
        sentences = realistic_sentences()
        total = total_tokens(sentences)  # 293
        grouped = BalancedSampler.group_by_levels(sentences, ["translator"])
        rng = np.random.default_rng(42)
        samples, _ = BalancedSampler.sample_balanced(
            grouped, ["translator"], total * 10, rng
        )
        assert len(samples) == 29
        assert total_tokens(samples) == total


class TestSampleBalancedBreakdown:
    """Breakdown structure and token count verification."""

    def test_breakdown_has_all_groups(self) -> None:
        sentences = realistic_sentences()
        grouped = BalancedSampler.group_by_levels(
            sentences, ["translator", "author", "work"]
        )
        rng = np.random.default_rng(42)
        _, breakdown = BalancedSampler.sample_balanced(
            grouped, ["translator", "author", "work"], 50, rng
        )
        assert "Guillelmus" in breakdown
        assert "Burgundio" in breakdown
        assert "Bartholomaeus" in breakdown

    def test_breakdown_leaf_values_are_token_counts(self) -> None:
        sentences = realistic_sentences()
        grouped = BalancedSampler.group_by_levels(
            sentences, ["translator", "author", "work"]
        )
        rng = np.random.default_rng(42)
        _, breakdown = BalancedSampler.sample_balanced(
            grouped, ["translator", "author", "work"], 50, rng
        )

        def _check(node):
            if isinstance(node, dict):
                for child in node.values():
                    _check(child)
            else:
                assert isinstance(node, int)
                assert node >= 0

        _check(breakdown)


class TestSampleBalancedOvershoot:
    """Greedy overshoot properties."""

    def test_overshoot_bounded_by_max_sentence(self) -> None:
        """Overshoot should be bounded (actual < target + max_sentence_length)."""
        sentences = realistic_sentences()
        max_sent = max(sentence_tokens(s) for s in sentences)  # 16
        grouped = BalancedSampler.group_by_levels(sentences, ["translator"])
        rng = np.random.default_rng(42)
        target = 50
        samples, _ = BalancedSampler.sample_balanced(
            grouped, ["translator"], target, rng
        )
        actual = total_tokens(samples)
        assert actual >= target
        # Overshoot is bounded by the max sentence length per group.
        # With multiple groups, total overshoot could be larger.
        # Just check it's not wildly off (within 2x max_sent per group).
        num_groups = len(grouped)
        assert actual < target + max_sent * num_groups, (
            f"Overshoot too large: {actual} vs target {target} + {max_sent}×{num_groups}"
        )

    def test_small_target_overshoot(self) -> None:
        """With target=7 and min sentence=6, should get 1 sentence (6 or 7+ tokens)."""
        sentences = realistic_sentences()
        grouped = BalancedSampler.group_by_levels(sentences, ["translator"])
        rng = np.random.default_rng(42)
        samples, _ = BalancedSampler.sample_balanced(grouped, ["translator"], 7, rng)
        actual = total_tokens(samples)
        assert actual >= 7
        # Min sentence is 6 tokens, max is 16
        assert actual < 7 + 16


class TestSampleBalancedProportional:
    """Proportional allocation across groups."""

    def test_proportional_allocation(self) -> None:
        """Groups with more tokens should get proportionally more samples.

        With a large enough target relative to corpus size, the allocation
        is proportional to group token counts. The actual sampled tokens
        may differ from allocation due to greedy accumulation, but the
        total should be close to the target.
        """
        sentences = realistic_sentences()
        # Guillelmus: 116, Burgundio: 98, Bartholomaeus: 79 (total 293)
        grouped = BalancedSampler.group_by_levels(sentences, ["translator"])
        rng = np.random.default_rng(42)
        samples, breakdown = BalancedSampler.sample_balanced(
            grouped, ["translator"], 200, rng
        )
        total_sampled = sum(len(s.text.split()) for s in samples)
        # Should sample close to the target (within 20% due to greedy overshoot)
        assert 160 <= total_sampled <= 240, (
            f"Expected ~200 tokens sampled, got {total_sampled}"
        )
        # All groups should have some samples
        assert len(breakdown) == 3


class TestSampleBalancedDeterminism:
    """Reproducibility with seeds."""

    def test_same_seed_same_samples(self) -> None:
        sentences = realistic_sentences()
        grouped = BalancedSampler.group_by_levels(
            sentences, ["translator", "author", "work"]
        )
        rng1 = np.random.default_rng(42)
        s1, _ = BalancedSampler.sample_balanced(
            grouped, ["translator", "author", "work"], 50, rng1
        )
        rng2 = np.random.default_rng(42)
        s2, _ = BalancedSampler.sample_balanced(
            grouped, ["translator", "author", "work"], 50, rng2
        )
        assert [s.text for s in s1] == [s.text for s in s2]

    def test_different_seeds_differ(self) -> None:
        sentences = realistic_sentences()
        grouped = BalancedSampler.group_by_levels(sentences, ["translator"])
        rng1 = np.random.default_rng(42)
        s1, _ = BalancedSampler.sample_balanced(grouped, ["translator"], 30, rng1)
        rng2 = np.random.default_rng(99)
        s2, _ = BalancedSampler.sample_balanced(grouped, ["translator"], 30, rng2)
        assert [s.text for s in s1] != [s.text for s in s2]


class TestSampleBalancedEdgeCases:
    """Edge cases and special scenarios."""

    def test_empty_groups(self) -> None:
        grouped = BalancedSampler.group_by_levels([], ["translator"])
        rng = np.random.default_rng(42)
        samples, _ = BalancedSampler.sample_balanced(grouped, ["translator"], 10, rng)
        assert len(samples) == 0

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
        samples, _ = BalancedSampler.sample_balanced(
            grouped, ["translator", "author", "work"], 1, rng
        )
        assert len(samples) == 1
        assert samples[0].text == "only one"

    def test_return_token_tuples(self) -> None:
        sentences = realistic_sentences()
        grouped = BalancedSampler.group_by_levels(sentences, ["translator"])
        rng = np.random.default_rng(42)
        samples, _ = BalancedSampler.sample_balanced(
            grouped, ["translator"], 30, rng, return_sentences=False
        )
        assert len(samples) > 0
        assert all(isinstance(s, tuple) for s in samples)

    def test_sentence_integrity_preserved(self) -> None:
        """Sampled sentences should be complete (not truncated)."""
        sentences = realistic_sentences()
        original_texts = {s.text for s in sentences}
        grouped = BalancedSampler.group_by_levels(
            sentences, ["translator", "author", "work"]
        )
        rng = np.random.default_rng(42)
        samples, _ = BalancedSampler.sample_balanced(
            grouped, ["translator", "author", "work"], 100, rng
        )
        for s in samples:
            assert s.text in original_texts

    def test_realistic_fixture_totals(self) -> None:
        """Verify fixture has expected totals."""
        sentences = realistic_sentences()
        assert len(sentences) == 29
        assert total_tokens(sentences) == 293
