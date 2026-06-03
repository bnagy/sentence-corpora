"""Tests for hierarchy-specific classes in sentence-corpora package."""

from __future__ import annotations

import numpy as np

from sentence_corpora import Sentence
from sentence_corpora.hierarchies import ThreeLevelCorpus, TwoLevelCorpus


class TestTwoLevelCorpus:
    """Tests for TwoLevelCorpus class."""

    def test_creation(self):
        """Test creating a TwoLevelCorpus."""
        sentences = [
            Sentence(text="Text 1", metadata={"work": "work1", "author": "author1"}),
            Sentence(text="Text 2", metadata={"work": "work2", "author": "author1"}),
            Sentence(text="Text 3", metadata={"work": "work1", "author": "author2"}),
        ]
        corpus = TwoLevelCorpus(sentences, levels=("author", "work"))
        assert len(corpus) == 3

    def test_get_levels(self):
        """Test getting hierarchy levels."""
        sentences = [
            Sentence(text="Text 1", metadata={"work": "work1", "author": "author1"}),
        ]
        corpus = TwoLevelCorpus(sentences, levels=("author", "work"))
        assert corpus.get_levels() == ["author", "work"]

    def test_get_unique_values(self):
        """Test getting unique values for a level."""
        sentences = [
            Sentence(text="Text 1", metadata={"work": "work1", "author": "author1"}),
            Sentence(text="Text 2", metadata={"work": "work2", "author": "author1"}),
            Sentence(text="Text 3", metadata={"work": "work1", "author": "author2"}),
        ]
        corpus = TwoLevelCorpus(sentences, levels=("author", "work"))
        assert corpus.get_unique_values("work") == ["work1", "work2"]
        assert corpus.get_unique_values("author") == ["author1", "author2"]

    def test_filter_by_level(self):
        """Test filtering corpus by level."""
        sentences = [
            Sentence(text="Text 1", metadata={"work": "work1", "author": "author1"}),
            Sentence(text="Text 2", metadata={"work": "work2", "author": "author1"}),
            Sentence(text="Text 3", metadata={"work": "work1", "author": "author2"}),
        ]
        corpus = TwoLevelCorpus(sentences, levels=("author", "work"))

        filtered = corpus.filter_by_level("work", "work1")
        assert len(filtered) == 2
        assert all(s.work == "work1" for s in filtered)

    def test_get_unique_values_translator(self):
        """Test getting unique translator values."""
        sentences = [
            Sentence(
                text="Text 1",
                metadata={"work": "work1", "author": "author1", "translator": "trans1"},
            ),
            Sentence(
                text="Text 2",
                metadata={"work": "work1", "author": "author1", "translator": "trans2"},
            ),
        ]
        corpus = ThreeLevelCorpus(sentences, levels=("translator", "author", "work"))
        assert corpus.get_unique_values("translator") == ["trans1", "trans2"]

    def test_sample_balanced(self):
        """Test balanced sampling."""
        sentences = [
            Sentence(
                text="Text 1",
                metadata={"work": "work1", "author": "author1", "translator": "trans1"},
            ),
            Sentence(
                text="Text 2",
                metadata={"work": "work1", "author": "author1", "translator": "trans1"},
            ),
            Sentence(
                text="Text 3",
                metadata={"work": "work2", "author": "author2", "translator": "trans1"},
            ),
            Sentence(
                text="Text 4",
                metadata={"work": "work2", "author": "author2", "translator": "trans1"},
            ),
        ]

        corpus = ThreeLevelCorpus(sentences, levels=("translator", "author", "work"))
        rng = np.random.default_rng(42)

        samples, breakdown = corpus.sample_balanced(2, rng)

        assert len(samples) == 2
        assert isinstance(samples[0], Sentence)
        assert "trans1" in breakdown

    def test_stats(self, capsys):
        """Test stats method prints correct table."""
        sentences = [
            Sentence(
                text="word1 word2 word3",
                metadata={"work": "work1", "author": "author1", "translator": "trans1"},
            ),
            Sentence(
                text="word1 word2",
                metadata={"work": "work2", "author": "author1", "translator": "trans1"},
            ),
            Sentence(
                text="word1 word2 word3 word4",
                metadata={"work": "work1", "author": "author2", "translator": "trans2"},
            ),
        ]
        corpus = ThreeLevelCorpus(sentences, levels=("translator", "author", "work"))
        corpus.stats()
        captured = capsys.readouterr()
        assert "trans1" in captured.out
        assert "trans2" in captured.out
        assert "TOTAL" in captured.out

    def test_to_from_pickle(self, tmp_path):
        """Test pickling and unpickling ThreeLevelCorpus."""
        sentences = [
            Sentence(
                text="Text 1",
                metadata={"work": "work1", "author": "author1", "translator": "trans1"},
            ),
            Sentence(
                text="Text 2",
                metadata={"work": "work1", "author": "author1", "translator": "trans1"},
            ),
            Sentence(
                text="Text 3",
                metadata={"work": "work2", "author": "author2", "translator": "trans2"},
            ),
        ]
        corpus = ThreeLevelCorpus(sentences, levels=("translator", "author", "work"))

        path = tmp_path / "corpus.pkl"
        corpus.to_pickle(str(path))

        loaded = ThreeLevelCorpus.from_pickle(str(path))
        assert isinstance(loaded, ThreeLevelCorpus)
        assert len(loaded) == 3
        assert loaded[0].text == "Text 1"
        # Verify the loaded corpus is fully functional
        assert loaded.get_unique_values("translator") == ["trans1", "trans2"]
        assert loaded.get_unique_values("work") == ["work1", "work2"]
        # Verify sample_balanced works on loaded corpus
        rng = np.random.default_rng(42)
        samples, breakdown = loaded.sample_balanced(4, rng)
        assert len(samples) > 0

    def test_repr(self):
        """Test repr output."""
        sentences = [
            Sentence(
                text="Text 1",
                metadata={"work": "work1", "author": "author1", "translator": "trans1"},
            ),
        ]
        corpus = ThreeLevelCorpus(sentences, levels=("translator", "author", "work"))
        assert repr(corpus) == "ThreeLevelCorpus(1 sentences)"

    def test_two_level_repr(self):
        """Test TwoLevelCorpus repr output."""
        sentences = [
            Sentence(text="Text 1", metadata={"work": "work1", "author": "author1"}),
        ]
        corpus = TwoLevelCorpus(sentences, levels=("author", "work"))
        assert repr(corpus) == "TwoLevelCorpus(1 sentences)"

    def test_empty_three_level_corpus(self):
        """Test empty ThreeLevelCorpus."""
        corpus = ThreeLevelCorpus([], levels=("translator", "author", "work"))
        assert len(corpus) == 0
        assert corpus.get_unique_values("translator") == []
        assert corpus.get_unique_values("author") == []
        assert corpus.get_unique_values("work") == []

    def test_empty_two_level_corpus(self):
        """Test empty TwoLevelCorpus."""
        corpus = TwoLevelCorpus([], levels=("author", "work"))
        assert len(corpus) == 0
        assert corpus.get_unique_values("work") == []
        assert corpus.get_unique_values("author") == []
