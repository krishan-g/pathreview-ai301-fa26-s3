"""Check if generated feedback is supported by retrieved context."""

import re

import structlog

logger = structlog.get_logger()

TOKEN_PATTERN = re.compile(r"[a-z0-9+#]+")

STOP_WORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "and",
    "or",
    "but",
    "in",
    "of",
    "to",
    "for",
    "that",
    "with",
}

# Words that say someone has a skill without naming it. Judgment call
# (issue #59): tuned on profile-style feedback, not exhaustive.
GENERIC_PROFILE_WORDS = {
    "has",
    "have",
    "shows",
    "knows",
    "skilled",
    "skills",
    "strong",
    "developer",
    "candidate",
    "expert",
    "expertise",
    "experience",
    "experienced",
    "knowledge",
}

# A claim counts as supported when at least this share of its content
# words appears in the context.
SUPPORT_THRESHOLD = 0.5


class FaithfulnessChecker:
    """Verify that feedback claims are supported by context."""

    def check(self, feedback: str, context_chunks: list[dict]) -> float:
        """Check faithfulness of feedback to context.

        Args:
            feedback: Generated feedback text
            context_chunks: Retrieved context chunks

        Returns:
            Faithfulness score 0.0-1.0 (mean share of each claim's content
            words found in the context)
        """
        if not feedback or not context_chunks:
            logger.info(
                "faithfulness_empty_input",
                has_feedback=bool(feedback),
                has_chunks=bool(context_chunks),
            )
            return 0.0

        # Extract key claims from feedback (sentences)
        claims = self._extract_claims(feedback)
        if not claims:
            logger.info("faithfulness_no_claims_extracted")
            return 0.5  # Default to neutral if no extractable claims

        # Concatenate context text
        context_text = " ".join([chunk.get("text", "") for chunk in context_chunks])

        # Score each claim by how much of it the context supports
        shares = [self._support_share(claim, context_text) for claim in claims]
        supported = sum(1 for share in shares if share >= SUPPORT_THRESHOLD)

        score = sum(shares) / len(shares)

        logger.info(
            "faithfulness_checked", claims_count=len(claims), supported_count=supported, score=score
        )

        return score

    @staticmethod
    def _extract_claims(text: str) -> list[str]:
        """Extract key claims from feedback text.

        Args:
            text: Feedback text

        Returns:
            List of claims (sentences)
        """
        # Split by sentence (simple regex)
        sentences = re.split(r"[.!?]+", text)
        claims = [s.strip() for s in sentences if len(s.split()) >= 2]
        return claims[:10]  # Limit to 10 claims for scoring

    @staticmethod
    def _is_supported(claim: str, context: str) -> bool:
        """Check if a claim is supported by context.

        Args:
            claim: Claim text
            context: Context text

        Returns:
            True if at least half of the claim's content words are in context
        """
        return FaithfulnessChecker._support_share(claim, context) >= SUPPORT_THRESHOLD

    @staticmethod
    def _support_share(claim: str, context: str) -> float:
        """Share of a claim's content words that appear in the context.

        Args:
            claim: Claim text
            context: Context text

        Returns:
            Fraction 0.0-1.0 of the claim's content words found in context
        """
        # Tokenize with a regex so punctuation doesn't stick to words
        claim_tokens = set(TOKEN_PATTERN.findall(claim.lower()))
        context_tokens = set(TOKEN_PATTERN.findall(context.lower()))

        content_words = claim_tokens - STOP_WORDS - GENERIC_PROFILE_WORDS
        if not content_words:
            return 0.0

        return len(content_words & context_tokens) / len(content_words)
