"""Tests for hierarchy-specific classes in sentence-corpora package."""

from __future__ import annotations

import numpy as np
import pytest

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
        corpus = TwoLevelCorpus(sentences)
        assert len(corpus) == 3

    def test_get_levels(self):
        """Test getting hierarchy levels."""
        sentences = [
            Sentence(text="Text 1", metadata={"work": "work1", "author": "author1"}),
        ]
        corpus = TwoLevelCorpus(sentences)
        assert corpus.get_levels() == ["work", "author"]

    def test_get_unique_values(self):
        """Test getting unique values for a level."""
        sentences = [
            Sentence(text="Text 1", metadata={"work": "work1", "author": "author1"}),
            Sentence(text="Text 2", metadata={"work": "work2", "author": "author1"}),
            Sentence(text="Text 3", metadata={"work": "work1", "author": "author2"}),
        ]
        corpus = TwoLevelCorpus(sentences)
        assert corpus.get_unique_values("work") == ["work1", "work2"]
        assert corpus.get_unique_values("author") == ["author1", "author2"]

    def test_filter_by_level(self):
        """Test filtering corpus by level."""
        sentences = [
            Sentence(text="Text 1", metadata={"work": "work1", "author": "author1"}),
            Sentence(text="Text 2", metadata={"work": "work2", "author": "author1"}),
            Sentence(text="Text 3", metadata={"work": "work1", "author": "author2"}),
        ]
        corpus = TwoLevelCorpus(sentences)

        filtered = corpus.filter_by_level("work", "work1")
        assert len(filtered) == 2
        assert all(s.work == "work1" for s in filtered)

    def test_by_work(self):
        """Test by_work method."""
        sentences = [
            Sentence(text="Text 1", metadata={"work": "work1", "author": "author1"}),
            Sentence(text="Text 2", metadata={"work": "work2", "author": "author1"}),
            Sentence(text="Text 3", metadata={"work": "work1", "author": "author2"}),
        ]
        corpus = TwoLevelCorpus(sentences)

        filtered = corpus.by_work("work1")
        assert len(filtered) == 2

    def test_by_author(self):
        """Test by_author method."""
        sentences = [
            Sentence(text="Text 1", metadata={"work": "work1", "author": "author1"}),
            Sentence(text="Text 2", metadata={"work": "work2", "author": "author1"}),
            Sentence(text="Text 3", metadata={"work": "work1", "author": "author2"}),
        ]
        corpus = TwoLevelCorpus(sentences)

        filtered = corpus.by_author("author1")
        assert len(filtered) == 2

    def test_works(self):
        """Test works method."""
        sentences = [
            Sentence(text="Text 1", metadata={"work": "work1", "author": "author1"}),
            Sentence(text="Text 2", metadata={"work": "work2", "author": "author1"}),
        ]
        corpus = TwoLevelCorpus(sentences)
        assert corpus.works() == ["work1", "work2"]

    def test_authors(self):
        """Test authors method."""
        sentences = [
            Sentence(text="Text 1", metadata={"work": "work1", "author": "author1"}),
            Sentence(text="Text 2", metadata={"work": "work1", "author": "author2"}),
        ]
        corpus = TwoLevelCorpus(sentences)
        assert corpus.authors() == ["author1", "author2"]

    def test_sample_balanced(self):
        """Test balanced sampling (token-based).

        Each sentence is 2 tokens. Requesting 8 tokens should yield
        approximately 4 sentences (may be 3-5 due to greedy overshoot).
        """
        sentences = [
            Sentence(text="Text 1", metadata={"work": "work1", "author": "author1"}),
            Sentence(text="Text 2", metadata={"work": "work1", "author": "author1"}),
            Sentence(text="Text 3", metadata={"work": "work1", "author": "author1"}),
            Sentence(text="Text 4", metadata={"work": "work2", "author": "author2"}),
            Sentence(text="Text 5", metadata={"work": "work2", "author": "author2"}),
            Sentence(text="Text 6", metadata={"work": "work2", "author": "author2"}),
        ]

        corpus = TwoLevelCorpus(sentences)
        rng = np.random.default_rng(42)

        # 8 tokens requested; each sentence is 2 tokens → expect ~4 sentences
        samples, breakdown = corpus.sample_balanced(8, rng)

        total_tokens = sum(len(s.text.split()) for s in samples)
        assert total_tokens >= 8
        assert isinstance(samples[0], Sentence)
        assert "work1" in breakdown
        assert "work2" in breakdown

        def _check_breakdown(node):
            if isinstance(node, dict):
                for child in node.values():
                    _check_breakdown(child)
            else:
                assert isinstance(node, int)
                assert node >= 0

        _check_breakdown(breakdown)

    def test_stats(self, capsys):
        """Test stats method prints correct table."""
        sentences = [
            Sentence(
                text="word1 word2 word3",
                metadata={"work": "work1", "author": "author1"},
            ),
            Sentence(
                text="word1 word2",
                metadata={"work": "work2", "author": "author1"},
            ),
            Sentence(
                text="word1 word2 word3 word4",
                metadata={"work": "work1", "author": "author2"},
            ),
        ]
        corpus = TwoLevelCorpus(sentences)
        corpus.stats()
        captured = capsys.readouterr()
        assert "author1" in captured.out
        assert "author2" in captured.out
        assert "TOTAL" in captured.out

    def test_to_from_pickle(self, tmp_path):
        """Test pickling and unpickling TwoLevelCorpus."""
        sentences = [
            Sentence(text="Text 1", metadata={"work": "work1", "author": "author1"}),
        ]
        corpus = TwoLevelCorpus(sentences)

        path = tmp_path / "corpus.pkl"
        corpus.to_pickle(str(path))

        loaded = TwoLevelCorpus.from_pickle(str(path))
        assert len(loaded) == 1
        assert loaded[0].text == "Text 1"

    def test_chunk_works(self):
        """Test chunking works into smaller pieces."""
        # 7 sentences * 500 tokens = 3500 tokens, min_tokens=1000
        # Greedy approach: 1500, 1500, 500 tokens (last chunk under min_tokens is OK)
        sentences = [
            Sentence(
                text="word " * 500, metadata={"work": "work1", "author": "author1"}
            ),
            Sentence(
                text="word " * 500, metadata={"work": "work1", "author": "author1"}
            ),
            Sentence(
                text="word " * 500, metadata={"work": "work1", "author": "author1"}
            ),
            Sentence(
                text="word " * 500, metadata={"work": "work1", "author": "author1"}
            ),
            Sentence(
                text="word " * 500, metadata={"work": "work1", "author": "author1"}
            ),
            Sentence(
                text="word " * 500, metadata={"work": "work1", "author": "author1"}
            ),
            Sentence(
                text="word " * 500, metadata={"work": "work1", "author": "author1"}
            ),
        ]
        corpus = TwoLevelCorpus(sentences)

        chunked = corpus.chunk_works(1000)
        assert len(chunked) == 7
        works = chunked.works()
        assert "work1_1" in works
        assert "work1_2" in works
        assert "work1_3" in works

    def test_chunk_works_too_small(self):
        """Test that chunk_works raises ValueError for works too small."""
        sentences = [
            Sentence(
                text="word " * 100, metadata={"work": "work1", "author": "author1"}
            ),
        ]
        corpus = TwoLevelCorpus(sentences)

        with pytest.raises(ValueError, match="has 100 tokens"):
            corpus.chunk_works(200)


class TestThreeLevelCorpus:
    """Tests for ThreeLevelCorpus class."""

    def test_creation(self):
        """Test creating a ThreeLevelCorpus."""
        sentences = [
            Sentence(
                text="Text 1",
                metadata={"work": "work1", "author": "author1", "translator": "trans1"},
            ),
            Sentence(
                text="Text 2",
                metadata={"work": "work2", "author": "author1", "translator": "trans1"},
            ),
            Sentence(
                text="Text 3",
                metadata={"work": "work1", "author": "author2", "translator": "trans1"},
            ),
        ]
        corpus = ThreeLevelCorpus(sentences)
        assert len(corpus) == 3

    def test_get_levels(self):
        """Test getting hierarchy levels."""
        sentences = [
            Sentence(
                text="Text 1",
                metadata={"work": "work1", "author": "author1", "translator": "trans1"},
            ),
        ]
        corpus = ThreeLevelCorpus(sentences)
        assert corpus.get_levels() == ["work", "author", "translator"]

    def test_get_unique_values(self):
        """Test getting unique values for a level."""
        sentences = [
            Sentence(
                text="Text 1",
                metadata={"work": "work1", "author": "author1", "translator": "trans1"},
            ),
            Sentence(
                text="Text 2",
                metadata={"work": "work2", "author": "author1", "translator": "trans1"},
            ),
            Sentence(
                text="Text 3",
                metadata={"work": "work1", "author": "author2", "translator": "trans2"},
            ),
        ]
        corpus = ThreeLevelCorpus(sentences)
        assert corpus.get_unique_values("translator") == ["trans1", "trans2"]
        assert corpus.get_unique_values("author") == ["author1", "author2"]
        assert corpus.get_unique_values("work") == ["work1", "work2"]

    def test_filter_by_level(self):
        """Test filtering corpus by level."""
        sentences = [
            Sentence(
                text="Text 1",
                metadata={"work": "work1", "author": "author1", "translator": "trans1"},
            ),
            Sentence(
                text="Text 2",
                metadata={"work": "work2", "author": "author1", "translator": "trans1"},
            ),
            Sentence(
                text="Text 3",
                metadata={"work": "work1", "author": "author2", "translator": "trans2"},
            ),
        ]
        corpus = ThreeLevelCorpus(sentences)

        filtered = corpus.filter_by_level("translator", "trans1")
        assert len(filtered) == 2
        assert all(s.translator == "trans1" for s in filtered)

    def test_by_translator(self):
        """Test by_translator method."""
        sentences = [
            Sentence(
                text="Text 1",
                metadata={"work": "work1", "author": "author1", "translator": "trans1"},
            ),
            Sentence(
                text="Text 2",
                metadata={"work": "work2", "author": "author1", "translator": "trans1"},
            ),
            Sentence(
                text="Text 3",
                metadata={"work": "work1", "author": "author2", "translator": "trans2"},
            ),
        ]
        corpus = ThreeLevelCorpus(sentences)

        filtered = corpus.by_translator("trans1")
        assert len(filtered) == 2

    def test_by_author(self):
        """Test by_author method."""
        sentences = [
            Sentence(
                text="Text 1",
                metadata={"work": "work1", "author": "author1", "translator": "trans1"},
            ),
            Sentence(
                text="Text 2",
                metadata={"work": "work2", "author": "author1", "translator": "trans1"},
            ),
            Sentence(
                text="Text 3",
                metadata={"work": "work1", "author": "author2", "translator": "trans2"},
            ),
        ]
        corpus = ThreeLevelCorpus(sentences)

        filtered = corpus.by_author("author1")
        assert len(filtered) == 2

    def test_by_work(self):
        """Test by_work method."""
        sentences = [
            Sentence(
                text="Text 1",
                metadata={"work": "work1", "author": "author1", "translator": "trans1"},
            ),
            Sentence(
                text="Text 2",
                metadata={"work": "work2", "author": "author1", "translator": "trans1"},
            ),
            Sentence(
                text="Text 3",
                metadata={"work": "work1", "author": "author2", "translator": "trans2"},
            ),
        ]
        corpus = ThreeLevelCorpus(sentences)

        filtered = corpus.by_work("work1")
        assert len(filtered) == 2

    def test_translators(self):
        """Test translators method."""
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
        corpus = ThreeLevelCorpus(sentences)
        assert corpus.translators() == ["trans1", "trans2"]

    def test_authors(self):
        """Test authors method."""
        sentences = [
            Sentence(
                text="Text 1",
                metadata={"work": "work1", "author": "author1", "translator": "trans1"},
            ),
            Sentence(
                text="Text 2",
                metadata={"work": "work1", "author": "author2", "translator": "trans1"},
            ),
        ]
        corpus = ThreeLevelCorpus(sentences)
        assert corpus.authors() == ["author1", "author2"]

    def test_works(self):
        """Test works method."""
        sentences = [
            Sentence(
                text="Text 1",
                metadata={"work": "work1", "author": "author1", "translator": "trans1"},
            ),
            Sentence(
                text="Text 2",
                metadata={"work": "work2", "author": "author1", "translator": "trans1"},
            ),
        ]
        corpus = ThreeLevelCorpus(sentences)
        assert corpus.works() == ["work1", "work2"]

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

        corpus = ThreeLevelCorpus(sentences)
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
        corpus = ThreeLevelCorpus(sentences)
        corpus.stats()
        captured = capsys.readouterr()
        assert "trans1" in captured.out
        assert "trans2" in captured.out
        assert "TOTAL" in captured.out

    def test_sample_balanced_exclude_translator(self):
        """Test balanced sampling with exclude_translator (token-based).

        Each sentence is 2 tokens. Requesting 4 tokens from trans2-only
        corpus (2 sentences = 4 tokens) should return both trans2 sentences.
        """
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
            Sentence(
                text="Text 4",
                metadata={"work": "work2", "author": "author2", "translator": "trans2"},
            ),
        ]
        corpus = ThreeLevelCorpus(sentences)
        rng = np.random.default_rng(42)

        # 4 tokens = 2 sentences (each is 2 tokens)
        samples, breakdown = corpus.sample_balanced(4, rng, exclude_translator="trans1")

        assert len(samples) == 2
        assert all(s.translator != "trans1" for s in samples)

    def test_chunk_works(self):
        """Test chunking works into smaller pieces."""
        # 7 sentences * 500 tokens = 3500 tokens, min_tokens=1000
        # Greedy approach: 1500, 1500, 500 tokens (last chunk under min_tokens is OK)
        sentences = [
            Sentence(
                text="word " * 500,
                metadata={"work": "work1", "author": "author1", "translator": "trans1"},
            ),
            Sentence(
                text="word " * 500,
                metadata={"work": "work1", "author": "author1", "translator": "trans1"},
            ),
            Sentence(
                text="word " * 500,
                metadata={"work": "work1", "author": "author1", "translator": "trans1"},
            ),
            Sentence(
                text="word " * 500,
                metadata={"work": "work1", "author": "author1", "translator": "trans1"},
            ),
            Sentence(
                text="word " * 500,
                metadata={"work": "work1", "author": "author1", "translator": "trans1"},
            ),
            Sentence(
                text="word " * 500,
                metadata={"work": "work1", "author": "author1", "translator": "trans1"},
            ),
            Sentence(
                text="word " * 500,
                metadata={"work": "work1", "author": "author1", "translator": "trans1"},
            ),
        ]
        corpus = ThreeLevelCorpus(sentences)

        chunked = corpus.chunk_works(1000)
        assert len(chunked) == 7
        works = chunked.works()
        assert "work1_1" in works
        assert "work1_2" in works
        assert "work1_3" in works

    def test_to_from_pickle(self, tmp_path):
        """Test pickling and unpickling ThreeLevelCorpus."""
        sentences = [
            Sentence(
                text="Text 1",
                metadata={"work": "work1", "author": "author1", "translator": "trans1"},
            ),
        ]
        corpus = ThreeLevelCorpus(sentences)

        path = tmp_path / "corpus.pkl"
        corpus.to_pickle(str(path))

        loaded = ThreeLevelCorpus.from_pickle(str(path))
        assert len(loaded) == 1
        assert loaded[0].text == "Text 1"

    def test_repr(self):
        """Test repr output."""
        sentences = [
            Sentence(
                text="Text 1",
                metadata={"work": "work1", "author": "author1", "translator": "trans1"},
            ),
        ]
        corpus = ThreeLevelCorpus(sentences)
        assert repr(corpus) == "ThreeLevelCorpus(1 sentences)"

    def test_two_level_repr(self):
        """Test TwoLevelCorpus repr output."""
        sentences = [
            Sentence(text="Text 1", metadata={"work": "work1", "author": "author1"}),
        ]
        corpus = TwoLevelCorpus(sentences)
        assert repr(corpus) == "TwoLevelCorpus(1 sentences)"

    def test_empty_three_level_corpus(self):
        """Test empty ThreeLevelCorpus."""
        corpus = ThreeLevelCorpus([])
        assert len(corpus) == 0
        assert corpus.translators() == []
        assert corpus.authors() == []
        assert corpus.works() == []

    def test_empty_two_level_corpus(self):
        """Test empty TwoLevelCorpus."""
        corpus = TwoLevelCorpus([])
        assert len(corpus) == 0
        assert corpus.works() == []
        assert corpus.authors() == []
