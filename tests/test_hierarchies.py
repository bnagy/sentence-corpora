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

        samples, breakdown = corpus.sample_balanced(sentences=2, rng=rng)

        # Balanced sampling may return slightly more than requested due to
        # greedy accumulation at leaf level
        assert 2 <= len(samples) <= 4
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
        samples, breakdown = loaded.sample_balanced(sentences=4, rng=rng)
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

    def test_empty_corpus_with_explicit_levels(self):
        """Test empty corpus works when levels are passed explicitly."""
        corpus = ThreeLevelCorpus([], levels=("translator", "author", "work"))
        assert len(corpus) == 0
        assert corpus.levels == ("translator", "author", "work")
        assert corpus.get_unique_values("translator") == []
        assert corpus.get_unique_values("author") == []
        assert corpus.get_unique_values("work") == []

    def test_filter_by_level_preserves_levels(self):
        """Test filter_by_level preserves levels in new instance."""
        sentences = [
            Sentence(text="Text 1", metadata={"work": "w1", "author": "a1"}),
            Sentence(text="Text 2", metadata={"work": "w2", "author": "a1"}),
            Sentence(text="Text 3", metadata={"work": "w1", "author": "a2"}),
        ]
        corpus = TwoLevelCorpus(sentences, levels=("author", "work"))
        filtered = corpus.filter_by_level("author", "a1")
        assert filtered.levels == ("author", "work")
        assert len(filtered) == 2

    def test_sample_balanced_with_exclude(self):
        """Test balanced sampling with exclude parameter."""
        sentences = [
            Sentence(text="w1 w2", metadata={"work": "w1", "author": "a1"}),
            Sentence(text="w3 w4", metadata={"work": "w2", "author": "a1"}),
            Sentence(text="w5 w6", metadata={"work": "w3", "author": "a2"}),
            Sentence(text="w7 w8", metadata={"work": "w4", "author": "a2"}),
        ]
        corpus = TwoLevelCorpus(sentences, levels=("author", "work"))
        rng = np.random.default_rng(42)
        samples, breakdown = corpus.sample_balanced(sentences=2, rng=rng, exclude=("author", "a1"))
        assert all(s.metadata["author"] != "a1" for s in samples)

    def test_stats_with_dynamic_levels(self, capsys):
        """Test stats uses dynamic level names in header."""
        sentences = [
            Sentence(text="word1 word2 word3", metadata={"work": "w1", "author": "a1"}),
            Sentence(text="word1 word2", metadata={"work": "w2", "author": "a1"}),
            Sentence(text="word1 word2 word3 word4", metadata={"work": "w1", "author": "a2"}),
        ]
        corpus = TwoLevelCorpus(sentences, levels=("author", "work"))
        corpus.stats()
        captured = capsys.readouterr()
        assert "Author" in captured.out
        assert "a1" in captured.out
        assert "a2" in captured.out
        assert "TOTAL" in captured.out

    def test_to_from_pickle_with_levels(self, tmp_path):
        """Test pickling and unpickling preserves levels."""
        sentences = [
            Sentence(text="Text 1", metadata={"work": "w1", "author": "a1"}),
            Sentence(text="Text 2", metadata={"work": "w2", "author": "a2"}),
        ]
        corpus = TwoLevelCorpus(sentences, levels=("author", "work"))
        path = tmp_path / "corpus.pkl"
        corpus.to_pickle(str(path))
        loaded = TwoLevelCorpus.from_pickle(str(path), levels=("author", "work"))
        assert loaded.levels == ("author", "work")
        assert len(loaded) == 2

    def test_from_pickle_auto_detects_with_warning(self, tmp_path):
        """Test from_pickle without levels auto-detects and warns."""
        import warnings
        sentences = [
            Sentence(text="Text 1", metadata={"author": "a1", "work": "w1"}),
        ]
        corpus = TwoLevelCorpus(sentences, levels=("author", "work"))
        path = tmp_path / "corpus.pkl"
        corpus.to_pickle(str(path))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            TwoLevelCorpus.from_pickle(str(path))
            assert len(w) == 1
            assert "auto-detected" in str(w[0].message)

    def test_chunk_preserves_levels(self):
        """Test chunk preserves levels."""
        sentences = [
            Sentence(text="word " * 500, metadata={"work": "w1", "author": "a1"}),
            Sentence(text="word " * 500, metadata={"work": "w1", "author": "a1"}),
            Sentence(text="word " * 500, metadata={"work": "w1", "author": "a1"}),
            Sentence(text="word " * 500, metadata={"work": "w1", "author": "a1"}),
            Sentence(text="word " * 500, metadata={"work": "w1", "author": "a1"}),
            Sentence(text="word " * 500, metadata={"work": "w1", "author": "a1"}),
            Sentence(text="word " * 500, metadata={"work": "w1", "author": "a1"}),
        ]
        corpus = TwoLevelCorpus(sentences, levels=("author", "work"))
        chunked = corpus.chunk(1000)
        assert chunked.levels == ("author", "work")
        assert len(chunked) == 7

    def test_custom_levels_end_to_end(self):
        """Test TwoLevelCorpus with custom level names."""
        sentences = [
            Sentence(text="t1", metadata={"section": "s1", "chapter": "c1"}),
            Sentence(text="t2", metadata={"section": "s1", "chapter": "c2"}),
            Sentence(text="t3", metadata={"section": "s2", "chapter": "c1"}),
        ]
        corpus = TwoLevelCorpus(sentences, levels=("section", "chapter"))
        assert corpus.get_levels() == ["section", "chapter"]
        assert corpus.get_unique_values("section") == ["s1", "s2"]
        assert corpus.get_unique_values("chapter") == ["c1", "c2"]
        filtered = corpus.filter_by_level("section", "s1")
        assert len(filtered) == 2
        assert filtered.levels == ("section", "chapter")


class TestThreeLevelCorpusAutoDetect:
    """Tests for ThreeLevelCorpus auto-detection and levels property."""

    def test_levels_property_explicit(self):
        """Test levels property returns explicitly set levels."""
        sentences = [
            Sentence(text="Text 1", metadata={"work": "w1", "author": "a1", "translator": "t1"}),
        ]
        corpus = ThreeLevelCorpus(sentences, levels=("translator", "author", "work"))
        assert corpus.levels == ("translator", "author", "work")

    def test_levels_property_auto_detected(self):
        """Test levels property returns auto-detected levels."""
        import warnings
        sentences = [
            Sentence(text="Text 1", metadata={"translator": "t1", "author": "a1", "work": "w1"}),
        ]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            corpus = ThreeLevelCorpus(sentences)
            assert len(w) == 1
            assert "auto-detected" in str(w[0].message)
            assert corpus.levels == ("translator", "author", "work")

    def test_get_levels_returns_list(self):
        """Test get_levels returns a list copy."""
        sentences = [
            Sentence(text="Text 1", metadata={"work": "w1", "author": "a1", "translator": "t1"}),
        ]
        corpus = ThreeLevelCorpus(sentences, levels=("translator", "author", "work"))
        assert corpus.get_levels() == ["translator", "author", "work"]

    def test_auto_detect_wrong_key_count(self):
        """Test auto-detect raises ValueError for wrong number of keys."""
        import pytest
        sentences = [
            Sentence(text="Text 1", metadata={"work": "w1", "author": "a1"}),
        ]
        with pytest.raises(ValueError, match="Expected exactly 3 metadata keys"):
            ThreeLevelCorpus(sentences)

    def test_empty_corpus_with_explicit_levels(self):
        """Test empty corpus works when levels are passed explicitly."""
        corpus = ThreeLevelCorpus([], levels=("translator", "author", "work"))
        assert len(corpus) == 0
        assert corpus.levels == ("translator", "author", "work")
        assert corpus.get_unique_values("translator") == []
        assert corpus.get_unique_values("author") == []
        assert corpus.get_unique_values("work") == []

    def test_filter_by_level_preserves_levels(self):
        """Test filter_by_level preserves levels in new instance."""
        sentences = [
            Sentence(text="Text 1", metadata={"work": "w1", "author": "a1", "translator": "t1"}),
            Sentence(text="Text 2", metadata={"work": "w2", "author": "a1", "translator": "t1"}),
            Sentence(text="Text 3", metadata={"work": "w1", "author": "a2", "translator": "t2"}),
        ]
        corpus = ThreeLevelCorpus(sentences, levels=("translator", "author", "work"))
        filtered = corpus.filter_by_level("translator", "t1")
        assert filtered.levels == ("translator", "author", "work")
        assert len(filtered) == 2

    def test_sample_balanced_with_exclude(self):
        """Test balanced sampling with exclude parameter."""
        sentences = [
            Sentence(text="w1 w2", metadata={"work": "w1", "author": "a1", "translator": "t1"}),
            Sentence(text="w3 w4", metadata={"work": "w2", "author": "a1", "translator": "t1"}),
            Sentence(text="w5 w6", metadata={"work": "w3", "author": "a2", "translator": "t2"}),
            Sentence(text="w7 w8", metadata={"work": "w4", "author": "a2", "translator": "t2"}),
        ]
        corpus = ThreeLevelCorpus(sentences, levels=("translator", "author", "work"))
        rng = np.random.default_rng(42)
        samples, breakdown = corpus.sample_balanced(sentences=2, rng=rng, exclude=("translator", "t1"))
        assert all(s.metadata["translator"] != "t1" for s in samples)

    def test_stats_with_dynamic_levels(self, capsys):
        """Test stats uses dynamic level names in header."""
        sentences = [
            Sentence(text="word1 word2 word3", metadata={"work": "w1", "author": "a1", "translator": "t1"}),
            Sentence(text="word1 word2", metadata={"work": "w2", "author": "a1", "translator": "t1"}),
            Sentence(text="word1 word2 word3 word4", metadata={"work": "w1", "author": "a2", "translator": "t2"}),
        ]
        corpus = ThreeLevelCorpus(sentences, levels=("translator", "author", "work"))
        corpus.stats()
        captured = capsys.readouterr()
        assert "Translator" in captured.out
        assert "t1" in captured.out
        assert "t2" in captured.out
        assert "TOTAL" in captured.out

    def test_to_from_pickle_with_levels(self, tmp_path):
        """Test pickling and unpickling preserves levels."""
        sentences = [
            Sentence(text="Text 1", metadata={"work": "w1", "author": "a1", "translator": "t1"}),
            Sentence(text="Text 2", metadata={"work": "w2", "author": "a2", "translator": "t2"}),
        ]
        corpus = ThreeLevelCorpus(sentences, levels=("translator", "author", "work"))
        path = tmp_path / "corpus.pkl"
        corpus.to_pickle(str(path))
        loaded = ThreeLevelCorpus.from_pickle(str(path), levels=("translator", "author", "work"))
        assert loaded.levels == ("translator", "author", "work")
        assert len(loaded) == 2

    def test_from_pickle_auto_detects_with_warning(self, tmp_path):
        """Test from_pickle without levels auto-detects and warns."""
        import warnings
        sentences = [
            Sentence(text="Text 1", metadata={"translator": "t1", "author": "a1", "work": "w1"}),
        ]
        corpus = ThreeLevelCorpus(sentences, levels=("translator", "author", "work"))
        path = tmp_path / "corpus.pkl"
        corpus.to_pickle(str(path))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ThreeLevelCorpus.from_pickle(str(path))
            assert len(w) == 1
            assert "auto-detected" in str(w[0].message)

    def test_chunk_preserves_levels(self):
        """Test chunk preserves levels."""
        sentences = [
            Sentence(text="word " * 500, metadata={"work": "w1", "author": "a1", "translator": "t1"}),
            Sentence(text="word " * 500, metadata={"work": "w1", "author": "a1", "translator": "t1"}),
            Sentence(text="word " * 500, metadata={"work": "w1", "author": "a1", "translator": "t1"}),
            Sentence(text="word " * 500, metadata={"work": "w1", "author": "a1", "translator": "t1"}),
            Sentence(text="word " * 500, metadata={"work": "w1", "author": "a1", "translator": "t1"}),
            Sentence(text="word " * 500, metadata={"work": "w1", "author": "a1", "translator": "t1"}),
            Sentence(text="word " * 500, metadata={"work": "w1", "author": "a1", "translator": "t1"}),
        ]
        corpus = ThreeLevelCorpus(sentences, levels=("translator", "author", "work"))
        chunked = corpus.chunk(1000)
        assert chunked.levels == ("translator", "author", "work")
        assert len(chunked) == 7

    def test_custom_levels_end_to_end(self):
        """Test ThreeLevelCorpus with custom level names (meter)."""
        sentences = [
            Sentence(text="t1", metadata={"meter": "dactylic", "author": "a1", "work": "w1"}),
            Sentence(text="t2", metadata={"meter": "iambic", "author": "a1", "work": "w2"}),
            Sentence(text="t3", metadata={"meter": "dactylic", "author": "a2", "work": "w1"}),
        ]
        corpus = ThreeLevelCorpus(sentences, levels=("meter", "author", "work"))
        assert corpus.get_levels() == ["meter", "author", "work"]
        assert corpus.get_unique_values("meter") == ["dactylic", "iambic"]
        filtered = corpus.filter_by_level("meter", "dactylic")
        assert len(filtered) == 2
        assert filtered.levels == ("meter", "author", "work")

        rng = np.random.default_rng(42)
        samples, breakdown = corpus.sample_balanced(sentences=2, rng=rng)
        assert len(samples) == 2

        samples, breakdown = corpus.sample_balanced(sentences=2, rng=rng, exclude=("meter", "dactylic"))
        assert len(samples) == 1
        assert samples[0].metadata["meter"] == "iambic"

    def test_exclude_by_middle_level(self):
        """Test exclude works for middle level in ThreeLevelCorpus."""
        sentences = [
            Sentence(text="w1 w2", metadata={"work": "w1", "author": "a1", "translator": "t1"}),
            Sentence(text="w3 w4", metadata={"work": "w2", "author": "a1", "translator": "t1"}),
            Sentence(text="w5 w6", metadata={"work": "w3", "author": "a2", "translator": "t1"}),
            Sentence(text="w7 w8", metadata={"work": "w4", "author": "a2", "translator": "t1"}),
        ]
        corpus = ThreeLevelCorpus(sentences, levels=("translator", "author", "work"))
        rng = np.random.default_rng(42)
        samples, breakdown = corpus.sample_balanced(sentences=2, rng=rng, exclude=("author", "a1"))
        assert all(s.metadata["author"] != "a1" for s in samples)

    def test_repr_with_levels(self):
        """Test repr output."""
        sentences = [
            Sentence(text="Text 1", metadata={"work": "w1", "author": "a1", "translator": "t1"}),
        ]
        corpus = ThreeLevelCorpus(sentences, levels=("translator", "author", "work"))
        assert repr(corpus) == "ThreeLevelCorpus(1 sentences)"

    def test_two_level_repr(self):
        """Test TwoLevelCorpus repr output."""
        sentences = [
            Sentence(text="Text 1", metadata={"work": "w1", "author": "a1"}),
        ]
        corpus = TwoLevelCorpus(sentences, levels=("author", "work"))
        assert repr(corpus) == "TwoLevelCorpus(1 sentences)"

    def test_get_unique_tuples(self):
        """Test getting unique tuples across all levels."""
        sentences = [
            Sentence(text="T1", metadata={"work": "w1", "author": "a1"}),
            Sentence(text="T2", metadata={"work": "w1", "author": "a1"}),
            Sentence(text="T3", metadata={"work": "w2", "author": "a1"}),
            Sentence(text="T4", metadata={"work": "w2", "author": "a2"}),
        ]
        corpus = TwoLevelCorpus(sentences, levels=("author", "work"))
        result = corpus.get_unique_tuples()
        assert result == [("a1", "w1"), ("a1", "w2"), ("a2", "w2")]

    def test_get_unique_tuples_empty_corpus(self):
        """Test get_unique_tuples on empty corpus returns empty list."""
        corpus = TwoLevelCorpus([], levels=("author", "work"))
        assert corpus.get_unique_tuples() == []

    def test_get_unique_tuples_single_sentence(self):
        """Test get_unique_tuples with a single sentence."""
        sentences = [
            Sentence(text="T1", metadata={"work": "w1", "author": "a1"}),
        ]
        corpus = TwoLevelCorpus(sentences, levels=("author", "work"))
        assert corpus.get_unique_tuples() == [("a1", "w1")]


class TestThreeLevelCorpusGetUniqueTuples:
    """Tests for ThreeLevelCorpus.get_unique_tuples."""

    def test_get_unique_tuples(self):
        """Test getting unique tuples across all three levels."""
        sentences = [
            Sentence(text="T1", metadata={"work": "w1", "author": "a1", "translator": "t1"}),
            Sentence(text="T2", metadata={"work": "w1", "author": "a1", "translator": "t1"}),
            Sentence(text="T3", metadata={"work": "w2", "author": "a1", "translator": "t1"}),
            Sentence(text="T4", metadata={"work": "w2", "author": "a2", "translator": "t2"}),
        ]
        corpus = ThreeLevelCorpus(sentences, levels=("translator", "author", "work"))
        result = corpus.get_unique_tuples()
        assert result == [
            ("t1", "a1", "w1"),
            ("t1", "a1", "w2"),
            ("t2", "a2", "w2"),
        ]

    def test_get_unique_tuples_empty_corpus(self):
        """Test get_unique_tuples on empty corpus returns empty list."""
        corpus = ThreeLevelCorpus([], levels=("translator", "author", "work"))
        assert corpus.get_unique_tuples() == []

    def test_get_unique_tuples_single_sentence(self):
        """Test get_unique_tuples with a single sentence."""
        sentences = [
            Sentence(text="T1", metadata={"work": "w1", "author": "a1", "translator": "t1"}),
        ]
        corpus = ThreeLevelCorpus(sentences, levels=("translator", "author", "work"))
        assert corpus.get_unique_tuples() == [("t1", "a1", "w1")]
