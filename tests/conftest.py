"""Shared test fixtures for sentence-corpora."""

from __future__ import annotations

import numpy as np
import pytest

from sentence_corpora import Corpus, Sentence
from sentence_corpora.hierarchies import ThreeLevelCorpus, TwoLevelCorpus


@pytest.fixture
def two_level_sentences() -> list[Sentence]:
    """Return a list of sentences with work/author metadata."""
    return [
        Sentence(text="word1 word2", metadata={"work": "w1", "author": "a1"}),
        Sentence(text="word3 word4 word5", metadata={"work": "w1", "author": "a1"}),
        Sentence(text="word6", metadata={"work": "w2", "author": "a1"}),
        Sentence(text="word7 word8", metadata={"work": "w2", "author": "a2"}),
        Sentence(text="word9 word10 word11", metadata={"work": "w3", "author": "a2"}),
    ]


@pytest.fixture
def three_level_sentences() -> list[Sentence]:
    """Return a list of sentences with work/author/translator metadata."""
    return [
        Sentence(
            text="word1 word2",
            metadata={"work": "w1", "author": "a1", "translator": "t1"},
        ),
        Sentence(
            text="word3 word4 word5",
            metadata={"work": "w1", "author": "a1", "translator": "t1"},
        ),
        Sentence(
            text="word6",
            metadata={"work": "w2", "author": "a1", "translator": "t1"},
        ),
        Sentence(
            text="word7 word8",
            metadata={"work": "w2", "author": "a2", "translator": "t2"},
        ),
        Sentence(
            text="word9 word10 word11",
            metadata={"work": "w3", "author": "a2", "translator": "t2"},
        ),
    ]


@pytest.fixture
def two_level_corpus(two_level_sentences: list[Sentence]) -> TwoLevelCorpus:
    """Return a TwoLevelCorpus."""
    return TwoLevelCorpus(two_level_sentences)


@pytest.fixture
def three_level_corpus(three_level_sentences: list[Sentence]) -> ThreeLevelCorpus:
    """Return a ThreeLevelCorpus."""
    return ThreeLevelCorpus(three_level_sentences)


@pytest.fixture
def base_corpus(two_level_sentences: list[Sentence]) -> Corpus:
    """Return a base Corpus."""
    return Corpus(two_level_sentences)


@pytest.fixture
def rng() -> np.random.Generator:
    """Return a seeded NumPy random generator."""
    return np.random.default_rng(42)
