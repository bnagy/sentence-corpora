"""LambdaG Authorship Verification module.

Provides a simple interface for running translator verification problems
using the LambdaG method with balanced sentence sampling.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Optional

import numpy as np
from lambdag import LambdaGMethod

from ..hierarchies import ThreeLevelCorpus


class LambdaGAV:
    """Run LambdaG translator verification problems with balanced sampling.

    This class wraps the LambdaG method for verifying whether an unknown
    text was translated by a known translator. It handles balanced sampling
    of sentences across works, authors, and translators.

    Example:
        >>> av = LambdaGAV()
        >>> result = av.run_single_av_problem(
        ...     known_translator="Guillelmus de Morbeka",
        ...     unknown_corpus=nile_corpus,
        ...     reference_corpus=train_corpus,
        ...     known_size=1000,
        ...     reference_size=5000,
        ... )
        >>> print(f"Score: {result['score']}")
    """

    def __init__(
        self,
        order: int = 4,
        num_references: int = 10,
        seed: int = 42,
    ) -> None:
        """Initialize the LambdaG AV runner.

        Args:
            order: N-gram order for the language model.
            num_references: Number of references for LambdaGMethod.
                LambdaGMethod.lambdag_score will resample the references
                this many times and return the average LLR from comparing
                the (one) known language model with the (sampled) reference
                models.
            seed: Default random seed for reproducibility.
        """
        self.order = order
        self.num_references = num_references
        self.seed = seed

    @staticmethod
    def llr_to_percent_probability(llr: float, pre_test_prob: float = 0.5) -> float:
        """Convert a Log-Likelihood Ratio (LLR) to a percentage probability.

        Args:
            llr: The log-likelihood ratio (natural log base e).
            pre_test_prob: Prior probability (0.0 to 1.0).
                Defaults to 0.5 (neutral/50-50).

        Returns:
            Post-test probability as a percentage.
        """
        if pre_test_prob >= 1.0:
            return 100.0
        if pre_test_prob <= 0.0:
            return 0.0

        lr = math.exp(llr)
        o_pre = pre_test_prob / (1 - pre_test_prob)
        o_post = o_pre * lr
        p_post = o_post / (1 + o_post)

        return p_post * 100

    @staticmethod
    def _flatten_breakdown_works(breakdown: dict) -> dict[str, int]:
        """Flatten a translator→author→work breakdown to just work counts.

        Args:
            breakdown: Nested dict from sample_balanced.

        Returns:
            Flat dict mapping work names to sentence counts.
        """
        result: dict[str, int] = {}
        for translator_data in breakdown.values():
            for author_data in translator_data.values():
                result.update(author_data)
        return result

    def run_single_av_problem(
        self,
        known_translator: str,
        unknown_corpus: ThreeLevelCorpus,
        reference_corpus: ThreeLevelCorpus,
        known_size: int = 1000,
        reference_size: int = 5000,
        order: int | None = None,
        num_references: int | None = None,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> dict[str, object]:
        """Run a single translator verification AV problem.

        Steps:
        1. Sample *known_size* sentences from *known_translator*,
           balanced across their works.
        2. Sample *reference_size* sentences from all other translators,
           balanced across translators, authors, and works.
        3. Use all sentences from *unknown_corpus* as the unknown text.
        4. Compute the LambdaG score.

        Args:
            known_translator: Full translator name
                (e.g. "Guillelmus de Morbeka").
            unknown_corpus: Corpus of the unknown text (e.g. De Nilo).
            reference_corpus: Corpus with all translators' sentences.
            known_size: Sentences to sample from the known translator.
            reference_size: Sentences for the reference set.
            order: N-gram order (defaults to instance value).
            num_references: Passed to LambdaGMethod (defaults to instance
                value).
            seed: Random seed (defaults to instance value). Ignored if
                *rng* is provided.
            rng: Seeded numpy.random.Generator. If provided, overrides
                *seed* and is passed directly to LambdaGMethod's
                random_generator parameter.

        Returns:
            Dictionary with score, metadata, and sample breakdowns.
        """
        order = order if order is not None else self.order
        num_references = (
            num_references if num_references is not None else self.num_references
        )

        if rng is None:
            seed = seed if seed is not None else self.seed
            rng = np.random.default_rng(seed)

        # Step 1: known translator sentences
        known_corpus = reference_corpus.by_translator(known_translator)
        if len(known_corpus) == 0:
            return {"error": f"No sentences found for translator {known_translator!r}"}

        known_sentences_raw, known_breakdown_full = known_corpus.sample_balanced(
            known_size, rng
        )
        known_sentences = [tuple(s.text.split()) for s in known_sentences_raw]
        known_breakdown = self._flatten_breakdown_works(known_breakdown_full)

        # Step 2: reference sentences (exclude known translator)
        ref_sentences_list = [
            s for s in reference_corpus if s.translator != known_translator
        ]
        ref_corpus = ThreeLevelCorpus(ref_sentences_list)
        reference_sentences_raw, ref_breakdown = ref_corpus.sample_balanced(
            reference_size, rng
        )
        reference_sentences = [tuple(s.text.split()) for s in reference_sentences_raw]

        # Step 3: unknown sentences (use all)
        unknown_sentences = [tuple(s.text.split()) for s in unknown_corpus]
        unknown_token_count = sum(len(tokens) for tokens in unknown_sentences)

        # Step 4: compute LambdaG score
        method = LambdaGMethod(
            basis="tokens",
            order=order,
            smoothing="kneser_ney",
            lowercasing=False,
            sentenize=False,
            num_references=num_references,
        )
        method.random_gen = rng
        score = method.lambdag_score(
            known_sentences, unknown_sentences, reference_sentences
        )

        result_dict: dict[str, object] = {
            "score": score,
            "known_translator": known_translator,
            "known_sentences_used": len(known_sentences),
            "reference_sentences_used": len(reference_sentences),
            "unknown_sentences_used": len(unknown_sentences),
            "n_works_sampled": len(known_breakdown),
            "n_ref_translators": len(ref_breakdown),
            "known_breakdown": known_breakdown,
            "ref_breakdown": ref_breakdown,
        }

        # Corrected scores
        corrected_score_sqrt = score / math.sqrt(unknown_token_count)

        token_counts = Counter(
            token for tokens in unknown_sentences for token in tokens
        )
        hapax_count = sum(1 for count in token_counts.values() if count == 1)
        hapax_ratio = hapax_count / unknown_token_count
        corrected_score_hapax = score * hapax_ratio

        post_test_likelihood = self.llr_to_percent_probability(
            corrected_score_sqrt,
            1 / (result_dict["n_ref_translators"] + 1),  # type: ignore[operator]
        )

        result_dict["corrected_score_sqrt"] = corrected_score_sqrt
        result_dict["corrected_score_hapax"] = corrected_score_hapax
        result_dict["post_test_likelihood"] = post_test_likelihood
        result_dict["unknown_tokens_used"] = unknown_token_count
        result_dict["hapax_tokens"] = hapax_count
        result_dict["hapax_ratio"] = hapax_ratio

        return result_dict
