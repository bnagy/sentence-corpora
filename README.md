# sentence-corpora

[![CI](https://github.com/bnagy/sentence-corpora/actions/workflows/ci.yml/badge.svg)](https://github.com/bnagy/sentence-corpora/actions/workflows/ci.yml)
[![Python 3.11–3.14](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Flexible sentence corpus handling with hierarchical sampling for stylometric analysis.

## Overview

The `sentence-corpora` package provides tools for managing sentence corpora with 2-3 hierarchy levels, with balanced sampling algorithms for stylometric analysis.

## Features

- **Flexible hierarchy support**: Handle 2-3 hierarchy levels (e.g., work/author, work/author/translator)
- **Balanced sampling**: Algorithms that distribute samples evenly across hierarchy levels
- **Composition-based design**: Hierarchy wrappers use composition to avoid dataclass inheritance issues

## Installation

```bash
pip install git+https://github.com/bnagy/sentence-corpora.git
```

## Usage

### Base Classes

```python
from sentence_corpora import Sentence, Corpus

sentences = [
    Sentence(text="Some text...", metadata={"work": "work1", "author": "author1"}),
    Sentence(text="More text...", metadata={"work": "work2", "author": "author1"}),
]
corpus = Corpus(sentences)
```

### Two-Level Hierarchy (work/author)

```python
from sentence_corpora import Sentence
from sentence_corpora.hierarchies import TwoLevelCorpus

sentences = [
    Sentence(text="Some text...", metadata={"work": "work1", "author": "author1"}),
    Sentence(text="More text...", metadata={"work": "work2", "author": "author1"}),
]
corpus = TwoLevelCorpus(sentences)
```

### Three-Level Hierarchy (work/author/translator)

```python
from sentence_corpora import Sentence
from sentence_corpora.hierarchies import ThreeLevelCorpus

sentences = [
    Sentence(text="Some text...", metadata={"work": "work1", "author": "author1", "translator": "trans1"}),
]
corpus = ThreeLevelCorpus(sentences)
```

### Balanced Sampling

```python
import numpy as np
from sentence_corpora.sampling import BalancedSampler

# Group sentences by hierarchy levels
grouped = BalancedSampler.group_by_levels(corpus, ['translator', 'author', 'work'])

# Sample balanced across all levels
samples, breakdown = BalancedSampler.sample_balanced(
    grouped,
    levels=['translator', 'author', 'work'],
    total_samples=1000,
    rng=np.random.default_rng(42)
)
```

### LambdaG Authorship Verification

```python
from sentence_corpora.lambdag import LambdaGAV

av = LambdaGAV()
result = av.run_single_av_problem(
    known_translator="Guillelmus de Morbeka",
    unknown_corpus=nile_corpus,
    reference_corpus=train_corpus,
    known_size=1000,
    reference_size=5000,
)
print(f"Score: {result['score']}")
```

## Package Structure

```
sentence_corpora/
├── __init__.py          # Base classes (Sentence, Corpus)
├── sentence.py          # Sentence dataclass
├── corpus.py            # Corpus container class
├── hierarchies/         # Hierarchy-specific wrappers
│   ├── two_level.py     # Two-level hierarchy (work/author)
│   └── three_level.py   # Three-level hierarchy (work/author/translator)
├── sampling/            # Sampling algorithms
│   └── balanced_sampler.py
└── lambdag/             # LambdaG authorship verification
    └── av.py            # Authorship verification with balanced sampling
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT License