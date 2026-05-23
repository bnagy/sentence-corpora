"""Tests for BalancedSampler.group_by_levels."""

from __future__ import annotations

from sentence_corpora import Sentence
from sentence_corpora.sampling import BalancedSampler

from .sampling_fixture import realistic_sentences


class TestGroupByLevels:
    """Tests for BalancedSampler.group_by_levels."""

    def test_single_level(self) -> None:
        sentences = realistic_sentences()
        grouped = BalancedSampler.group_by_levels(sentences, ["translator"])
        assert set(grouped.keys()) == {"Guillelmus", "Burgundio", "Bartholomaeus"}
        assert len(grouped["Guillelmus"]) == 12
        assert len(grouped["Burgundio"]) == 9
        assert len(grouped["Bartholomaeus"]) == 8

    def test_two_levels(self) -> None:
        sentences = realistic_sentences()
        grouped = BalancedSampler.group_by_levels(sentences, ["translator", "author"])
        assert set(grouped.keys()) == {"Guillelmus", "Burgundio", "Bartholomaeus"}
        assert set(grouped["Guillelmus"].keys()) == {"Aristoteles", "Plato"}
        assert len(grouped["Guillelmus"]["Aristoteles"]) == 8
        assert len(grouped["Guillelmus"]["Plato"]) == 4

    def test_three_levels(self) -> None:
        sentences = realistic_sentences()
        grouped = BalancedSampler.group_by_levels(
            sentences, ["translator", "author", "work"]
        )
        guillelmus = grouped["Guillelmus"]
        assert set(guillelmus["Aristoteles"].keys()) == {"Analytica", "Physica"}
        assert len(guillelmus["Aristoteles"]["Analytica"]) == 5
        assert len(guillelmus["Aristoteles"]["Physica"]) == 3
        assert set(guillelmus["Plato"].keys()) == {"Timaeus"}
        assert len(guillelmus["Plato"]["Timaeus"]) == 4

    def test_empty_levels_returns_empty_dict(self) -> None:
        sentences = realistic_sentences()
        result = BalancedSampler.group_by_levels(sentences, [])
        assert result == {}

    def test_empty_sentence_list(self) -> None:
        grouped = BalancedSampler.group_by_levels([], ["translator"])
        assert grouped == {}
