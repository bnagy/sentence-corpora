"""Tests for Sentence and Corpus classes in sentence-corpora package."""

from __future__ import annotations
import pytest
from sentence_corpora import Sentence, Corpus


class TestSentence:
    """Tests for Sentence class."""

    def test_creation(self):
        """Test creating a Sentence."""
        sentence = Sentence(
            text="Test sentence.", metadata={"work": "work1", "author": "author1"}
        )
        assert sentence.text == "Test sentence."
        assert sentence.metadata["work"] == "work1"
        assert sentence.metadata["author"] == "author1"

    def test_dot_notation_access(self):
        """Test accessing metadata via dot notation."""
        sentence = Sentence(
            text="Test sentence.", metadata={"work": "work1", "author": "author1"}
        )
        assert sentence.work == "work1"
        assert sentence.author == "author1"

    def test_frozen(self):
        """Test that Sentence is frozen."""
        sentence = Sentence(
            text="Test sentence.", metadata={"work": "work1", "author": "author1"}
        )
        with pytest.raises(AttributeError):
            sentence.text = "New text"  # pyright: ignore[reportAttributeAccessIssue]

    def test_missing_attribute(self):
        """Test accessing missing attribute raises AttributeError."""
        sentence = Sentence(
            text="Test sentence.", metadata={"work": "work1", "author": "author1"}
        )
        with pytest.raises(AttributeError):
            _ = sentence.translator


class TestCorpus:
    """Tests for Corpus class."""

    def test_creation(self):
        """Test creating a Corpus."""
        sentences = [
            Sentence(text="Text 1", metadata={"work": "work1", "author": "author1"}),
            Sentence(text="Text 2", metadata={"work": "work2", "author": "author1"}),
            Sentence(text="Text 3", metadata={"work": "work1", "author": "author2"}),
        ]
        corpus = Corpus(sentences)
        assert len(corpus) == 3

    def test_iteration(self):
        """Test iterating over corpus."""
        sentences = [
            Sentence(text="Text 1", metadata={"work": "work1", "author": "author1"}),
            Sentence(text="Text 2", metadata={"work": "work2", "author": "author1"}),
        ]
        corpus = Corpus(sentences)
        items = list(corpus)
        assert len(items) == 2

    def test_indexing(self):
        """Test indexing into corpus."""
        sentences = [
            Sentence(text="Text 1", metadata={"work": "work1", "author": "author1"}),
            Sentence(text="Text 2", metadata={"work": "work2", "author": "author1"}),
        ]
        corpus = Corpus(sentences)
        assert corpus[0].text == "Text 1"
        assert corpus[1].text == "Text 2"

    def test_get_levels(self):
        """Test getting hierarchy levels."""
        sentences = [
            Sentence(text="Text 1", metadata={"work": "work1", "author": "author1"}),
        ]
        corpus = Corpus(sentences)
        assert set(corpus.get_levels()) == {"work", "author"}

    def test_get_unique_values(self):
        """Test getting unique values for a level."""
        sentences = [
            Sentence(text="Text 1", metadata={"work": "work1", "author": "author1"}),
            Sentence(text="Text 2", metadata={"work": "work2", "author": "author1"}),
            Sentence(text="Text 3", metadata={"work": "work1", "author": "author2"}),
        ]
        corpus = Corpus(sentences)
        assert corpus.get_unique_values("work") == ["work1", "work2"]
        assert corpus.get_unique_values("author") == ["author1", "author2"]

    def test_filter_by_level(self):
        """Test filtering corpus by level."""
        sentences = [
            Sentence(text="Text 1", metadata={"work": "work1", "author": "author1"}),
            Sentence(text="Text 2", metadata={"work": "work2", "author": "author1"}),
            Sentence(text="Text 3", metadata={"work": "work1", "author": "author2"}),
        ]
        corpus = Corpus(sentences)

        filtered = corpus.filter_by_level("work", "work1")
        assert len(filtered) == 2
        assert all(s.work == "work1" for s in filtered)

    def test_sample(self):
        """Test sampling from corpus."""
        sentences = [
            Sentence(text="Text 1", metadata={"work": "work1", "author": "author1"}),
            Sentence(text="Text 2", metadata={"work": "work2", "author": "author1"}),
            Sentence(text="Text 3", metadata={"work": "work1", "author": "author2"}),
        ]
        corpus = Corpus(sentences)

        sampled = corpus.sample(2)
        assert len(sampled) == 2

    def test_to_from_pickle(self, tmp_path):
        """Test pickling and unpickling corpus."""
        sentences = [
            Sentence(text="Text 1", metadata={"work": "work1", "author": "author1"}),
        ]
        corpus = Corpus(sentences)

        path = tmp_path / "corpus.pkl"
        corpus.to_pickle(str(path))

        loaded = Corpus.from_pickle(str(path))
        assert len(loaded) == 1
        assert loaded[0].text == "Text 1"

    def test_eq_with_corpus(self):
        """Test equality between two corpora."""
        sentences = [
            Sentence(text="Text 1", metadata={"work": "work1", "author": "author1"}),
        ]
        corpus1 = Corpus(sentences)
        corpus2 = Corpus(sentences)
        assert corpus1 == corpus2

    def test_eq_with_non_corpus(self):
        """Test equality with non-Corpus returns NotImplemented."""
        sentences = [
            Sentence(text="Text 1", metadata={"work": "work1", "author": "author1"}),
        ]
        corpus = Corpus(sentences)
        assert corpus != "not a corpus"
        assert corpus != 42

    def test_eq_different_lengths(self):
        """Test inequality for corpora of different lengths."""
        corpus1 = Corpus(
            [
                Sentence(text="T1", metadata={"work": "w1", "author": "a1"}),
            ]
        )
        corpus2 = Corpus(
            [
                Sentence(text="T1", metadata={"work": "w1", "author": "a1"}),
                Sentence(text="T2", metadata={"work": "w2", "author": "a1"}),
            ]
        )
        assert corpus1 != corpus2

    def test_empty_corpus(self):
        """Test creating an empty corpus."""
        corpus = Corpus([])
        assert len(corpus) == 0
        assert corpus.get_levels() == []
        assert list(corpus) == []

    def test_filter_no_matches(self):
        """Test filtering with no matches returns empty corpus."""
        sentences = [
            Sentence(text="T1", metadata={"work": "w1", "author": "a1"}),
        ]
        corpus = Corpus(sentences)
        filtered = corpus.filter_by_level("work", "nonexistent")
        assert len(filtered) == 0

    def test_sample_zero(self):
        """Test sampling zero sentences."""
        sentences = [
            Sentence(text="T1", metadata={"work": "w1", "author": "a1"}),
        ]
        corpus = Corpus(sentences)
        assert corpus.sample(0) == []

    def test_repr(self):
        """Test repr output."""
        sentences = [
            Sentence(text="T1", metadata={"work": "w1", "author": "a1"}),
        ]
        corpus = Corpus(sentences)
        assert repr(corpus) == "Corpus(1 sentences)"

    def test_sentence_repr(self):
        """Test Sentence repr truncates long text."""
        long_text = "a" * 100
        sentence = Sentence(text=long_text, metadata={"work": "w1"})
        r = repr(sentence)
        assert "..." in r
        assert len(r) < len(long_text) + 50
