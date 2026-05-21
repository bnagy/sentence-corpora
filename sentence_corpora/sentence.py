"""Sentence class for sentence-corpora package."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Sentence:
    """A sentence with text and metadata.

    This class uses composition for metadata storage, avoiding dataclass
    inheritance issues. The metadata dictionary stores all hierarchical
    information, while dot-notation access is provided via :meth:`__getattr__`.

    Args:
        text: The raw text of the sentence.
        metadata: Dictionary containing hierarchy-level values
            (e.g., work, author, translator).
    """

    text: str
    metadata: dict[str, object]

    def __getattr__(self, attr: str) -> object:
        """Allow dot notation access to metadata fields."""
        if attr == "metadata":
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{attr}'"
            )
        if "metadata" in self.__dict__ and attr in self.metadata:
            return self.metadata[attr]
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{attr}'"
        )

    def __repr__(self) -> str:
        return f"Sentence(text='{self.text[:50]}...', metadata={self.metadata})"
