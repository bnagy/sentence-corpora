"""Tests for Sentence and Corpus classes in sentence-corpora package."""

from __future__ import annotations

from collections import Counter, defaultdict

import pytest

from sentence_corpora import Corpus, Sentence


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
            [Sentence(text="T1", metadata={"work": "w1", "author": "a1"})]
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

    # ---------------------------------------------------------------------------
    # chunk_works tests
    # ---------------------------------------------------------------------------

    def test_chunk_works_empty_corpus(self):
        """Test chunk_works on empty corpus."""
        corpus = Corpus([])
        chunked = corpus.chunk_works(100)
        assert len(chunked) == 0

    def test_chunk_works_no_levels(self):
        """Test chunk_works on corpus with sentences but no metadata levels."""
        sentences = [
            Sentence(text="word " * 500, metadata={}),
            Sentence(text="word " * 500, metadata={}),
        ]
        corpus = Corpus(sentences)
        chunked = corpus.chunk_works(100)
        assert len(chunked) == 2

    def test_chunk_works_all_chunks_above_min(self):
        """Every chunk has at least min_tokens (greedy_bmerge guarantee)."""
        sentences = [
            Sentence(text="word " * 20, metadata={"work": "w1"}) for _ in range(20)
        ]
        corpus = Corpus(sentences)
        chunked = corpus.chunk_works(50)

        chunk_tokens: dict[str, int] = defaultdict(int)
        for s in chunked:
            chunk_tokens[s.work] += len(s.text.split())
        for work, tokens in chunk_tokens.items():
            assert tokens >= 50, f"Chunk {work} has only {tokens} tokens (min=50)"

    def test_chunk_works_floor_constraint(self):
        """No chunk falls below floor (0.8 * min_tokens) after ripple."""
        sentences = [Sentence(text="s " * 10, metadata={"work": "w1"}) for _ in range(9)]
        sentences.append(Sentence(text="L " * 200, metadata={"work": "w1"}))
        corpus = Corpus(sentences)
        chunked = corpus.chunk_works(50)

        chunk_tokens: dict[str, int] = defaultdict(int)
        for s in chunked:
            chunk_tokens[s.work] += len(s.text.split())
        floor = 40
        for work, tokens in chunk_tokens.items():
            assert tokens >= floor, (
                f"Chunk {work} has {tokens} tokens, below floor={floor}"
            )

    def test_chunk_works_sentences_sequential(self):
        """Sentences remain in sequential order within and across chunks."""
        sentences = [
            Sentence(text="A " * 30, metadata={"work": "w1"}),
            Sentence(text="B " * 30, metadata={"work": "w1"}),
            Sentence(text="C " * 30, metadata={"work": "w1"}),
            Sentence(text="D " * 30, metadata={"work": "w1"}),
        ]
        corpus = Corpus(sentences)
        chunked = corpus.chunk_works(50)

        texts = [s.text for s in chunked]
        assert texts[0].startswith("A")
        assert texts[1].startswith("B")
        assert texts[2].startswith("C")
        assert texts[3].startswith("D")

    def test_chunk_works_no_split_sentences(self):
        """Sentences are not split across chunks."""
        sentences = [
            Sentence(text="word " * 30, metadata={"work": "w1"}),
            Sentence(text="word " * 30, metadata={"work": "w1"}),
            Sentence(text="word " * 30, metadata={"work": "w1"}),
            Sentence(text="word " * 30, metadata={"work": "w1"}),
        ]
        corpus = Corpus(sentences)
        chunked = corpus.chunk_works(50)

        work_counts = Counter(s.metadata["work"] for s in chunked)
        for count in work_counts.values():
            assert count == 2, f"Expected 2 sentences per chunk, got {count}"

    def test_chunk_works_no_author_collisions(self):
        """Chunks never mix sentences from different authors."""
        sentences = [
            Sentence(text="a " * 500, metadata={"work": "shared", "author": "A1"}),
            Sentence(text="b " * 500, metadata={"work": "shared", "author": "A1"}),
            Sentence(text="c " * 500, metadata={"work": "shared", "author": "A2"}),
            Sentence(text="d " * 500, metadata={"work": "shared", "author": "A2"}),
        ]
        corpus = Corpus(sentences)
        chunked = corpus.chunk_works(100)

        seen_pairs = set()
        for s in chunked:
            pair = (s.author, s.work)
            assert pair not in seen_pairs, f"Duplicate (author, work) pair: {pair}"
            seen_pairs.add(pair)

        chunk_sents: dict[tuple, list[Sentence]] = defaultdict(list)
        for s in chunked:
            chunk_sents[(s.author, s.work)].append(s)
        for (author, work_name), sents in chunk_sents.items():
            authors = set(s.author for s in sents)
            assert len(authors) == 1

        assert len(chunked) == 4

    def test_chunk_works_no_translator_collisions(self):
        """Chunks never mix sentences from different translators."""
        sentences = [
            Sentence(
                text="a " * 500,
                metadata={"work": "shared", "author": "Aristoteles", "translator": "T1"},
            ),
            Sentence(
                text="b " * 500,
                metadata={"work": "shared", "author": "Aristoteles", "translator": "T1"},
            ),
            Sentence(
                text="c " * 500,
                metadata={"work": "shared", "author": "Aristoteles", "translator": "T2"},
            ),
            Sentence(
                text="d " * 500,
                metadata={"work": "shared", "author": "Aristoteles", "translator": "T2"},
            ),
        ]
        corpus = Corpus(sentences)
        chunked = corpus.chunk_works(100)

        seen_tuples = set()
        for s in chunked:
            tup = (s.translator, s.author, s.work)
            assert tup not in seen_tuples
            seen_tuples.add(tup)

        chunk_sents: dict[tuple, list[Sentence]] = defaultdict(list)
        for s in chunked:
            chunk_sents[(s.translator, s.author, s.work)].append(s)
        for (translator, author, work_name), sents in chunk_sents.items():
            translators = set(s.translator for s in sents)
            assert len(translators) == 1

        assert len(chunked) == 4

    def test_chunk_works_single_sentence_above_min(self):
        """Edge case: single sentence with tokens >= min_tokens."""
        sentences = [
            Sentence(text="word " * 100, metadata={"work": "w1"}),
        ]
        corpus = Corpus(sentences)
        chunked = corpus.chunk_works(50)
        assert len(chunked) == 1
        assert len(chunked[0].text.split()) == 100

    def test_chunk_works_exact_multiple(self):
        """Edge case: total tokens is exact multiple of min_tokens."""
        sentences = [
            Sentence(text="word " * 50, metadata={"work": "w1"}) for _ in range(5)
        ]
        corpus = Corpus(sentences)
        chunked = corpus.chunk_works(50)
        assert len(chunked) == 5
        for s in chunked:
            assert len(s.text.split()) == 50

    def test_chunk_works_ripple_improves_evenness(self):
        """Ripple optimization produces reasonably even chunk sizes."""
        sentences = [Sentence(text="s " * 5, metadata={"work": "w1"}) for _ in range(20)]
        sentences.append(Sentence(text="L " * 100, metadata={"work": "w1"}))
        corpus = Corpus(sentences)
        chunked = corpus.chunk_works(50)

        chunk_tokens: dict[str, int] = defaultdict(int)
        for s in chunked:
            chunk_tokens[s.work] += len(s.text.split())
        sizes = list(chunk_tokens.values())
        mean = sum(sizes) / len(sizes)
        std = (sum((x - mean) ** 2 for x in sizes) / len(sizes)) ** 0.5
        assert std < 50, f"Chunk sizes too uneven: std={std:.1f}, sizes={sizes}"

    def test_chunk_works_small_remainder_merges_backward(self):
        """Small remainder is merged into the previous chunk."""
        sentences = [
            Sentence(text="A " * 60, metadata={"work": "w1"}),
            Sentence(text="B " * 60, metadata={"work": "w1"}),
            Sentence(text="C " * 10, metadata={"work": "w1"}),
        ]
        corpus = Corpus(sentences)
        chunked = corpus.chunk_works(50)

        work_counts = Counter(s.work for s in chunked)
        assert len(work_counts) == 2, f"Expected 2 chunks, got {len(work_counts)}"

        chunk_tokens: dict[str, int] = defaultdict(int)
        for s in chunked:
            chunk_tokens[s.work] += len(s.text.split())
        for work, tokens in chunk_tokens.items():
            assert tokens >= 50, f"Chunk {work} has only {tokens} tokens"
