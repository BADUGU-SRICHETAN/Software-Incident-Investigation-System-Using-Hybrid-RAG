import re
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document

from src.analysis.query_scope_checker import QueryScopeChecker


class AbstentionChecker:

    STOPWORDS = {
        "a", "an", "the", "is", "are", "was", "were",
        "be", "been", "being", "do", "does", "did",
        "why", "what", "when", "where", "who", "which",
        "how", "can", "could", "would", "should",
        "will", "may", "might", "must",
        "and", "or", "but", "if", "because",
        "to", "of", "in", "on", "at", "by", "for",
        "from", "with", "without", "about", "into",
        "through", "during", "after", "before",
        "it", "its", "this", "that", "these", "those",
        "my", "our", "their", "they", "them",
        "we", "you", "your", "i",
        "explain", "describe", "investigate",
        "resolve", "fix", "help", "issue", "problem",
        "application", "system",
        "previous", "associated", "occurs"
    }

    TECHNOLOGY_ALIASES = {
        "postgresql": {"postgresql", "postgres", "pgsql"},
        "docker": {"docker"},
        "kafka": {"kafka"},
        "kubernetes": {"kubernetes", "k8s"},
        "redis": {"redis"},
        "mongodb": {"mongodb", "mongo"},
        "mysql": {"mysql"},
        "elasticsearch": {"elasticsearch"},
        "nginx": {"nginx"},
        "python": {"python"},
    }

    # --------------------------------------------------
    # General Troubleshooting Intent Detection
    # --------------------------------------------------

    # Questions requesting general technical troubleshooting
    # or diagnostic guidance require external technical context.
    #
    # Retrospective questions about troubleshooting already
    # performed should be evaluated using historical evidence.

    GENERAL_TROUBLESHOOTING_PATTERNS = [
        r"\brecommended\s+troubleshooting\s+steps\b",
        r"\bcommon\s+methods\s+for\s+diagnosing\b",
        r"\bmethods\s+for\s+diagnosing\b",
        r"\bhow\s+can\b.*\bdiagnosed\b",
        r"\bhow\s+to\s+diagnose\b",
        r"\bdiagnostic\s+methods\b",
        r"\bdiagnostic\s+procedures\b",
    ]

    def __init__(
        self,
        vector_score_threshold: float = 0.35,
        score_top_k: int = 3,
        minimum_term_coverage: float = 0.25
    ):

        if not 0 <= vector_score_threshold <= 1:
            raise ValueError(
                "Vector score threshold must be between 0 and 1."
            )

        if score_top_k <= 0:
            raise ValueError(
                "score_top_k must be greater than zero."
            )

        if not 0 <= minimum_term_coverage <= 1:
            raise ValueError(
                "Minimum term coverage must be between 0 and 1."
            )

        self.vector_score_threshold = vector_score_threshold
        self.score_top_k = score_top_k
        self.minimum_term_coverage = minimum_term_coverage

        self.scope_checker = QueryScopeChecker()

    # --------------------------------------------------
    # Query Normalization
    # --------------------------------------------------

    @staticmethod
    def _normalize(text: str) -> str:
        return text.lower()

    @staticmethod
    def _normalize_term(term: str) -> str:
        """
        Normalize punctuation and common grammatical variations
        while preserving technical identifiers.
        """

        term = term.lower()

        term = term.strip(".,;!?()[]{}\"'")

        irregular_forms = {
            "rebalancing": "rebalance",
            "rebalances": "rebalance",
            "rebalanced": "rebalance",
            "errors": "error",
            "outages": "outage",
        }

        if term in irregular_forms:
            return irregular_forms[term]

        if (
            term.endswith("ies")
            and len(term) > 4
            and "." not in term
        ):
            return term[:-3] + "y"

        if (
            term.endswith("s")
            and not term.endswith(("ss", "us", "is"))
            and len(term) > 3
            and "." not in term
            and "/" not in term
            and ":" not in term
        ):
            return term[:-1]

        return term

    @classmethod
    def _extract_terms(cls, text: str) -> set:
        """
        Extract meaningful words and preserve technical identifiers.
        """

        text = cls._normalize(text)

        terms = re.findall(
            r"[a-zA-Z0-9_./:-]+",
            text
        )

        normalized_terms = set()

        for term in terms:

            term = cls._normalize_term(term)

            if term in cls.STOPWORDS:
                continue

            if len(term) <= 1:
                continue

            normalized_terms.add(term)

        return normalized_terms

    # --------------------------------------------------
    # General Troubleshooting Intent Detection
    # --------------------------------------------------

    @classmethod
    def _requires_external_context(cls, query: str) -> bool:
        """
        Detect questions requesting general troubleshooting
        procedures or diagnostic guidance.

        These questions should use external technical
        documentation alongside historical incident evidence.
        """

        query_lower = query.lower()

        return any(
            re.search(pattern, query_lower)
            for pattern in cls.GENERAL_TROUBLESHOOTING_PATTERNS
        )

    # --------------------------------------------------
    # Query Term Coverage
    # --------------------------------------------------

    @classmethod
    def _calculate_term_coverage(
        cls,
        query: str,
        document_content: str
    ) -> Dict[str, Any]:

        query_terms = cls._extract_terms(query)

        document_terms = cls._extract_terms(document_content)

        if not query_terms:
            return {
                "coverage": 1.0,
                "matched_terms": [],
                "missing_terms": []
            }

        matched_terms = query_terms.intersection(document_terms)

        missing_terms = query_terms.difference(document_terms)

        coverage = len(matched_terms) / len(query_terms)

        return {
            "coverage": coverage,
            "matched_terms": sorted(matched_terms),
            "missing_terms": sorted(missing_terms)
        }

    # --------------------------------------------------
    # Technology Detection
    # --------------------------------------------------

    @classmethod
    def _extract_required_technologies(cls, query: str) -> set:
        """
        Identify explicitly named technologies in the query.
        """

        query_lower = query.lower()

        required = set()

        for technology, aliases in cls.TECHNOLOGY_ALIASES.items():

            for alias in aliases:

                pattern = (
                    r"(?<![a-zA-Z0-9_])"
                    + re.escape(alias)
                    + r"(?![a-zA-Z0-9_])"
                )

                if re.search(pattern, query_lower):
                    required.add(technology)
                    break

        return required

    @classmethod
    def _check_technology_evidence(
        cls,
        query: str,
        document_content: str
    ) -> Dict[str, Any]:

        required = cls._extract_required_technologies(query)

        content = document_content.lower()

        supported = set()
        missing = set()

        for technology in required:

            aliases = cls.TECHNOLOGY_ALIASES[technology]

            found = False

            for alias in aliases:

                pattern = (
                    r"(?<![a-zA-Z0-9_])"
                    + re.escape(alias)
                    + r"(?![a-zA-Z0-9_])"
                )

                if re.search(pattern, content):
                    found = True
                    break

            if found:
                supported.add(technology)
            else:
                missing.add(technology)

        return {
            "required_technologies": sorted(required),
            "supported_technologies": sorted(supported),
            "missing_technologies": sorted(missing),
            "satisfied": not missing
        }

    # --------------------------------------------------
    # Hybrid Metadata Detection
    # --------------------------------------------------

    @staticmethod
    def _has_hybrid_metadata(document: Document) -> bool:

        metadata = document.metadata

        return any(
            key in metadata
            for key in (
                "bm25_score",
                "rrf_score",
                "retrieval_ranks"
            )
        )

    # --------------------------------------------------
    # Main Abstention Check
    # --------------------------------------------------

    def check(
        self,
        query: str,
        retrieved_documents: Optional[List[Document]] = None
    ) -> Dict[str, Any]:

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        # 1. Validate query scope.

        scope_result = self.scope_checker.check(query)

        if not scope_result["in_scope"]:

            return {
                "accepted": False,
                "reason": "out_of_scope",
                "category": scope_result["category"],
                "message": scope_result["reason"],
                "max_vector_score": None,
                "threshold": self.vector_score_threshold
            }

        # 2. Detect general troubleshooting intent.
        # Historical evidence alone should not be treated
        # as sufficient for general diagnostic guidance.

        if self._requires_external_context(query):

            return {
                "accepted": False,
                "reason": "general_troubleshooting_requires_external",
                "category": "general_technical_guidance",
                "message": (
                    "The query requests general troubleshooting "
                    "or diagnostic guidance. External technical "
                    "documentation is required alongside relevant "
                    "historical incident evidence."
                ),
                "max_vector_score": None,
                "threshold": self.vector_score_threshold
            }

        # 3. Check historical evidence availability.

        if not retrieved_documents:

            return {
                "accepted": False,
                "reason": "no_evidence",
                "category": "insufficient_evidence",
                "message": (
                    "No historical incident evidence was retrieved."
                ),
                "max_vector_score": None,
                "threshold": self.vector_score_threshold
            }

        # 4. Consider top-k documents.

        scoring_documents = retrieved_documents[:self.score_top_k]

        vector_scores = []

        for document in scoring_documents:

            score = document.metadata.get("vector_score")

            if score is None:
                continue

            score = float(score)

            if not 0 <= score <= 1:
                raise ValueError(
                    "Vector scores must be between 0 and 1."
                )

            vector_scores.append(score)

        # 5. Check vector score availability.

        if not vector_scores:

            return {
                "accepted": False,
                "reason": "no_vector_score",
                "category": "insufficient_evidence",
                "message": (
                    "No vector similarity scores were available."
                ),
                "max_vector_score": None,
                "threshold": self.vector_score_threshold
            }

        max_vector_score = max(vector_scores)

        # 6. Check vector relevance.

        if max_vector_score < self.vector_score_threshold:

            return {
                "accepted": False,
                "reason": "weak_evidence",
                "category": "insufficient_evidence",
                "message": (
                    "Retrieved evidence did not meet the "
                    "minimum vector similarity threshold."
                ),
                "max_vector_score": max_vector_score,
                "threshold": self.vector_score_threshold
            }

        # 7. Determine whether hybrid metadata is available.

        is_hybrid = any(
            self._has_hybrid_metadata(document)
            for document in scoring_documents
        )

        if is_hybrid:

            # Include BM25-only documents in hybrid evidence coverage.
            evidence_documents = scoring_documents

        else:

            # Preserve existing standalone vector behavior.
            evidence_documents = [
                document
                for document in scoring_documents
                if (
                    document.metadata.get("vector_score") is not None
                    and float(document.metadata["vector_score"])
                    >= self.vector_score_threshold
                )
            ]

        if not evidence_documents:

            return {
                "accepted": False,
                "reason": "insufficient_specificity",
                "category": "insufficient_evidence",
                "message": (
                    "No retrieved document met the vector "
                    "similarity threshold."
                ),
                "max_vector_score": max_vector_score,
                "threshold": self.vector_score_threshold
            }

        # 8. Combine retrieved evidence.

        combined_content = "\n".join(
            document.page_content
            for document in evidence_documents
        )

        coverage_result = self._calculate_term_coverage(
            query=query,
            document_content=combined_content
        )

        term_coverage = coverage_result["coverage"]

        # 9. Validate technology-specific evidence.

        technology_result = self._check_technology_evidence(
            query=query,
            document_content=combined_content
        )

        if term_coverage < self.minimum_term_coverage:

            return {
                "accepted": False,
                "reason": "insufficient_specificity",
                "category": "insufficient_evidence",
                "message": (
                    "Historical evidence is semantically related "
                    "but does not contain enough query-specific "
                    "technical information. External retrieval "
                    "is recommended."
                ),
                "max_vector_score": max_vector_score,
                "term_coverage": term_coverage,
                "minimum_term_coverage": self.minimum_term_coverage,
                "matched_terms": coverage_result["matched_terms"],
                "missing_terms": coverage_result["missing_terms"],
                "technology_evidence": technology_result,
                "threshold": self.vector_score_threshold
            }

        if not technology_result["satisfied"]:

            return {
                "accepted": False,
                "reason": "insufficient_technology_evidence",
                "category": "insufficient_evidence",
                "message": (
                    "Historical evidence does not explicitly "
                    "support all technologies named in the query. "
                    "External technical documentation is recommended."
                ),
                "max_vector_score": max_vector_score,
                "term_coverage": term_coverage,
                "minimum_term_coverage": self.minimum_term_coverage,
                "matched_terms": coverage_result["matched_terms"],
                "missing_terms": coverage_result["missing_terms"],
                "technology_evidence": technology_result,
                "threshold": self.vector_score_threshold
            }

        # 10. Accept historical evidence.

        return {
            "accepted": True,
            "reason": "evidence_available",
            "category": "historical_incident_investigation",
            "message": (
                "Historical evidence passed vector similarity, "
                "combined evidence coverage and technology-specific "
                "evidence checks."
            ),
            "max_vector_score": max_vector_score,
            "term_coverage": term_coverage,
            "minimum_term_coverage": self.minimum_term_coverage,
            "matched_terms": coverage_result["matched_terms"],
            "missing_terms": coverage_result["missing_terms"],
            "technology_evidence": technology_result,
            "hybrid_evidence_used": is_hybrid,
            "threshold": self.vector_score_threshold
        }