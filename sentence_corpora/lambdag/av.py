"""LambdaG Authorship Verification module.

Provides a simple interface for running authorship verification
problems using the LambdaG method with balanced sentence sampling.

Works with any corpus hierarchy (two-level or three-level). The verification
level (author, translator, meter, etc.) is inferred automatically from the corpus.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Protocol, runtime_checkable

import numpy as np
from lambdag import LambdaGMethod


@runtime_checkable
class HierarchicalCorpus(Protocol):
    """Protocol for corpus objects that LambdaGAV can work with.

    Any corpus providing these methods can be used with
    :meth:`LambdaGAV.run_single_av_problem`, regardless of hierarchy depth.
    """

    def get_levels(self) -> list[str]: ...
    def filter_by_level(self, level: str, value: str) -> "HierarchicalCorpus": ...
    def sample_balanced(
        self, total: int, rng: np.random.Generator
    ) -> tuple[list, dict]: ...
    def __len__(self) -> int: ...
    def __iter__(self): ...


class LambdaGAV:
    """Run LambdaG authorship verification with balanced sampling.

    Works with any corpus hierarchy (two-level or three-level). The
    verification level — the highest-level entity being tested — is
    inferred from ``corpus.get_levels()[-1]``.

    Examples:
        Three-level (e.g., translator verification)::

            >>> av = LambdaGAV()
            >>> result = av.run_single_av_problem(
            ...     known_entity="Guillelmus de Morbeka",
            ...     q_corpus=nile_corpus,
            ...     kr_corpus=train_corpus,
            ...     known_size=1000,
            ...     reference_size=5000,
            ... )

        Two-level (e.g., author verification)::

            >>> av = LambdaGAV()
            >>> result = av.run_single_av_problem(
            ...     known_entity="Aristotle",
            ...     q_corpus=unknown_work,
            ...     kr_corpus=train_corpus,
            ...     known_size=2000,
            ...     reference_size=10000,
            ... )
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
        # Convert LLR to likelihood ratio
        lr = math.exp(llr)
        # Prior odds
        o_pre = pre_test_prob / (1 - pre_test_prob)
        # Posterior odds
        o_post = o_pre * lr
        # Posterior probability
        p_post = o_post / (1 + o_post)

        return p_post * 100

    @staticmethod
    def _flatten_breakdown(breakdown: dict) -> dict[str, int]:
        """Recursively flatten a nested breakdown dict to leaf counts.

        Handles arbitrary nesting depth. At the leaves, values are integers
        (sentence counts). At intermediate levels, values are dicts.

        Args:
            breakdown: Nested dict from :meth:`sample_balanced`.

        Returns:
            Flat dict mapping leaf names to sentence counts.
        """
        result: dict[str, int] = {}
        for key, value in breakdown.items():
            if isinstance(value, dict):
                result.update(LambdaGAV._flatten_breakdown(value))
            else:
                result[key] = value
        return result

    def run_single_av_problem(
        self,
        known_entity: str,
        q_corpus: "HierarchicalCorpus",
        kr_corpus: "HierarchicalCorpus",
        known_size: int = 1000,
        reference_size: int = 5000,
        order: int | None = None,
        num_references: int | None = None,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> dict[str, object]:
        """Run a single AV problem.

        The verification level (e.g. "translator" or "author") is inferred
        from ``kr_corpus.get_levels()[-1]`` — the last (highest) level in
        the hierarchy.

        Steps:
        1. Extract the known corpus (k_corpus) from *kr_corpus* by
           filtering to *known_entity* at the verification level.
        2. Sample *known_size* sentences from k_corpus, balanced across
           lower hierarchy levels.
        3. Build the reference corpus by excluding *known_entity* from
           *kr_corpus*, then sample *reference_size* sentences balanced
           across all remaining entities.
        4. Use all sentences from *q_corpus* as the unknown text.
        5. Compute the LambdaG score.

        Args:
            known_entity: Name of the known entity (translator, author,
                etc.) being tested. Must match the value at the highest
                level in the corpus hierarchy.
            q_corpus: Corpus of the unknown/question text.
            kr_corpus: Combined known + reference corpus containing all
                entities' sentences.
            known_size: Minimum tokens to sample from the known entity.
            reference_size: Minimum tokens for the reference set.
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

        # Infer the verification level from the corpus hierarchy
        levels = kr_corpus.get_levels()
        if not levels:
            return {"error": "Corpus has no hierarchy levels"}
        verification_level = levels[0]

        # Step 1: extract known corpus and sample
        k_corpus = kr_corpus.filter_by_level(verification_level, known_entity)
        if len(k_corpus) == 0:
            return {
                "error": f"No sentences found for {verification_level} {known_entity!r}"
            }

        known_sentences_raw, known_breakdown_full = k_corpus.sample_balanced(
            known_size, rng
        )
        known_sentences = [tuple(s.text.split()) for s in known_sentences_raw]
        known_breakdown = self._flatten_breakdown(known_breakdown_full)

        # Step 2: build reference corpus (exclude known entity) and sample
        ref_sentences_list = [
            s for s in kr_corpus if getattr(s, verification_level) != known_entity
        ]
        ref_corpus = type(kr_corpus)(ref_sentences_list)  # type: ignore[call-arg]
        reference_sentences_raw, ref_breakdown = ref_corpus.sample_balanced(
            reference_size, rng
        )
        reference_sentences = [tuple(s.text.split()) for s in reference_sentences_raw]

        # Step 3: unknown sentences (use all of q_corpus)
        unknown_sentences = [tuple(s.text.split()) for s in q_corpus]
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

        n_ref_entities = len(ref_breakdown)

        result_dict: dict[str, object] = {
            "score": score,
            "known_entity": known_entity,
            "verification_level": verification_level,
            "known_sentences_used": len(known_sentences),
            "reference_sentences_used": len(reference_sentences),
            "unknown_sentences_used": len(unknown_sentences),
            "n_works_sampled": len(known_breakdown),
            "n_ref_entities": n_ref_entities,
            "known_breakdown": known_breakdown,
            "ref_breakdown": ref_breakdown,
        }

        # Corrected scores
        corrected_score_sqrt = (
            score / math.sqrt(unknown_token_count) if unknown_token_count > 0 else 0.0
        )

        token_counts = Counter(
            token for tokens in unknown_sentences for token in tokens
        )
        hapax_count = sum(1 for count in token_counts.values() if count == 1)
        hapax_ratio = (
            hapax_count / unknown_token_count if unknown_token_count > 0 else 0.0
        )
        corrected_score_hapax = score * hapax_ratio

        post_test_likelihood = self.llr_to_percent_probability(
            corrected_score_sqrt,
            1 / (n_ref_entities + 1),
        )

        result_dict["corrected_score_sqrt"] = corrected_score_sqrt
        result_dict["corrected_score_hapax"] = corrected_score_hapax
        result_dict["post_test_likelihood"] = post_test_likelihood
        result_dict["unknown_tokens_used"] = unknown_token_count
        result_dict["hapax_tokens"] = hapax_count
        result_dict["hapax_ratio"] = hapax_ratio

        return result_dict
