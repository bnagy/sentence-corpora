"""Shared test fixture for sampling tests.

Provides a realistic corpus with 3 translators, 5 authors, 7 works,
and varying sentence lengths (6–16 tokens), mimicking real POS-tagged
Latin text.
"""

from __future__ import annotations

from sentence_corpora import Sentence


def realistic_sentences() -> list[Sentence]:
    """Return a realistic corpus with varying sentence lengths.

    Structure:
      Guillelmus (12 sentences, 116 tokens):
        Aristoteles:
          Analytica:   5 sentences × [8,12,7,15,10] = 52 tokens
          Physica:     3 sentences × [9,11,6]        = 26 tokens
        Plato:
          Timaeus:     4 sentences × [10,8,13,7]     = 38 tokens

      Burgundio (9 sentences, 98 tokens):
        Aristoteles:
          Ethica:      4 sentences × [14,9,11,8]     = 42 tokens
        Plato:
          Republica:   3 sentences × [7,12,10]       = 29 tokens
        Proclus:
          Theologia:   2 sentences × [16,11]          = 27 tokens

      Bartholomaeus (8 sentences, 79 tokens):
        Aristoteles:
          Metaphysica: 3 sentences × [13,10,15]      = 38 tokens
        Alexander:
          De anima:    5 sentences × [6,9,8,11,7]    = 41 tokens

    Grand total: 29 sentences, 293 tokens.
    """
    sentences = []

    def make(n: int, work: str, author: str, translator: str) -> Sentence:
        tokens = " ".join(chr(ord("a") + (i % 26)) for i in range(n))
        return Sentence(
            text=tokens,
            metadata={
                "work": work,
                "author": author,
                "translator": translator,
            },
        )

    # Guillelmus
    for n in [8, 12, 7, 15, 10]:
        sentences.append(make(n, "Analytica", "Aristoteles", "Guillelmus"))
    for n in [9, 11, 6]:
        sentences.append(make(n, "Physica", "Aristoteles", "Guillelmus"))
    for n in [10, 8, 13, 7]:
        sentences.append(make(n, "Timaeus", "Plato", "Guillelmus"))

    # Burgundio
    for n in [14, 9, 11, 8]:
        sentences.append(make(n, "Ethica", "Aristoteles", "Burgundio"))
    for n in [7, 12, 10]:
        sentences.append(make(n, "Republica", "Plato", "Burgundio"))
    for n in [16, 11]:
        sentences.append(make(n, "Theologia", "Proclus", "Burgundio"))

    # Bartholomaeus
    for n in [13, 10, 15]:
        sentences.append(make(n, "Metaphysica", "Aristoteles", "Bartholomaeus"))
    for n in [6, 9, 8, 11, 7]:
        sentences.append(make(n, "De anima", "Alexander", "Bartholomaeus"))

    return sentences


def sentence_tokens(s: Sentence) -> int:
    """Return the number of tokens in a sentence."""
    return len(s.text.split())


def total_tokens(sentences: list[Sentence]) -> int:
    """Return the total token count for a list of sentences."""
    return sum(sentence_tokens(s) for s in sentences)
