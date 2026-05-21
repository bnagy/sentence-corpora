"""sentence-corpora: Flexible sentence corpus handling with hierarchical sampling.

This package provides tools for managing sentence corpora with 2-3
hierarchy levels, with balanced sampling algorithms for stylometric analysis.

Example usage:
    # Create sentences with metadata
    from sentence_corpora import Sentence, Corpus

    sentences = [
        Sentence(text="Some text...", metadata={"work": "work1", "author": "author1"}),
        Sentence(text="More text...", metadata={"work": "work2", "author": "author1"}),
    ]
    corpus = Corpus(sentences)

    # For two-level hierarchies (e.g., classical texts with work/author)
    from sentence_corpora.hierarchies import TwoLevelCorpus

    sentences = [
        Sentence(text="Some text...", metadata={"work": "work1", "author": "author1"}),
    ]
    corpus = TwoLevelCorpus(sentences)

    # For three-level hierarchies (e.g., nile project with work/author/translator)
    from sentence_corpora.hierarchies import ThreeLevelCorpus

    sentences = [
        Sentence(text="Some text...", metadata={"work": "work1", "author": "author1", "translator": "trans1"}),
    ]
    corpus = ThreeLevelCorpus(sentences)

    # Using balanced sampling
    import numpy as np
    from sentence_corpora.sampling import BalancedSampler

    samples, breakdown = BalancedSampler.sample_balanced(
        grouped_sentences,
        levels=['author', 'work'],
        total_samples=1000,
        rng=np.random.default_rng(42)
    )
"""

from __future__ import annotations

__version__ = "0.1.0"

from .sentence import Sentence
from .corpus import Corpus
from .hierarchies import TwoLevelCorpus, ThreeLevelCorpus
from .sampling import BalancedSampler

__all__ = [
    "Sentence",
    "Corpus",
    "TwoLevelCorpus",
    "ThreeLevelCorpus",
    "BalancedSampler",
]
