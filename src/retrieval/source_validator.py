import re
from typing import List, Set, Optional
from urllib.parse import urlparse, urlunparse

from langchain_core.documents import Document


class SourceValidator:
    """
    Validates, classifies, deduplicates, and ranks external web evidence.

    Features:
    - URL validation
    - Blocked-domain filtering
    - Content-length validation
    - Lexical relevance scoring
    - Trust classification
    - Version-aware documentation deduplication
    - Source authority ranking
    - Source diversity

    Trust classification is based on domain allowlists.
    It does not guarantee factual accuracy.
    """

    TRUSTED_DOMAINS = {
        "kafka.apache.org",
        "redhat.com",
        "strimzi.io",
        "learn.microsoft.com",
        "docs.python.org",
        "postgresql.org",
        "docs.github.com",
        "developer.mozilla.org",
        "kubernetes.io",
        "docs.docker.com",
        "owasp.org",
        "nvd.nist.gov",
        "cve.org",
    }

    COMMUNITY_DOMAINS = {
        "github.com",
        "stackoverflow.com",
        "serverfault.com",
    }

    THIRD_PARTY_DOMAINS = {
        "netdata.cloud",
        "acemq.com",
    }

    BLOCKED_DOMAINS = {
        "youtube.com",
        "youtu.be",
    }

    STOPWORDS = {
        "a", "an", "the", "is", "are", "was", "were",
        "our", "we", "they", "it", "this", "that",
        "and", "or", "but", "if", "then", "because",
        "to", "of", "in", "on", "for", "with", "from",
        "by", "at", "as", "be", "been", "being",
        "how", "what", "why", "when", "where",
        "can", "could", "should", "would", "will",
        "do", "does", "did", "have", "has", "had",
        "their", "your", "my",
        "application", "system", "incident",
        "experiencing", "experiences",
    }

    TRUST_PRIORITY = {
        "high": 3,
        "community": 2,
        "third_party": 1,
        "unverified": 0,
    }

    def __init__(
        self,
        min_content_length: int = 100,
        min_relevance_score: float = 0.15,
        max_per_domain: Optional[int] = 3,
    ):

        if min_content_length <= 0:
            raise ValueError(
                "min_content_length must be greater than zero."
            )

        if not 0.0 <= min_relevance_score <= 1.0:
            raise ValueError(
                "min_relevance_score must be between 0 and 1."
            )

        if max_per_domain is not None and max_per_domain <= 0:
            raise ValueError(
                "max_per_domain must be greater than zero or None."
            )

        self.min_content_length = min_content_length
        self.min_relevance_score = min_relevance_score
        self.max_per_domain = max_per_domain

    @staticmethod
    def normalize_domain(url: str) -> str:

        parsed = urlparse(url)

        domain = (parsed.hostname or "").lower()

        return domain.removeprefix("www.")

    @staticmethod
    def _matches_domain(
        domain: str,
        allowed_domains: Set[str],
    ) -> bool:

        return any(
            domain == allowed
            or domain.endswith("." + allowed)
            for allowed in allowed_domains
        )

    @staticmethod
    def _canonical_domain(domain: str) -> str:
        """
        Normalize versioned Kubernetes documentation hosts.

        Examples:
        v1-31.docs.kubernetes.io -> kubernetes.io
        v1-32.docs.kubernetes.io -> kubernetes.io
        """

        domain = domain.lower().removeprefix("www.")

        if re.match(
            r"^v\d+-\d+\.docs\.kubernetes\.io$",
            domain,
        ):
            return "kubernetes.io"

        return domain

    @classmethod
    def normalize_url(cls, url: str) -> str:
        """
        Create a canonical URL for duplicate detection.

        Removes:
        - URL fragments
        - Tracking parameters
        - Trailing slashes
        - Versioned Kubernetes documentation hosts
        """

        parsed = urlparse(url.strip())

        domain = cls.normalize_domain(url)
        canonical_domain = cls._canonical_domain(domain)

        path = parsed.path.rstrip("/")

        if not path:
            path = "/"

        # Remove common tracking parameters.
        ignored_parameters = {
            "utm_source",
            "utm_medium",
            "utm_campaign",
            "utm_term",
            "utm_content",
            "ref",
            "source",
        }

        query_parameters = []

        for parameter in parsed.query.split("&"):
            if not parameter:
                continue

            key = parameter.split("=")[0].lower()

            if key not in ignored_parameters:
                query_parameters.append(parameter)

        normalized_query = "&".join(
            sorted(query_parameters)
        )

        return urlunparse(
            (
                "https",
                canonical_domain,
                path,
                "",
                normalized_query,
                "",
            )
        ).lower()

    def classify_source(self, url: str) -> str:

        domain = self.normalize_domain(url)

        if self._matches_domain(
            domain,
            self.TRUSTED_DOMAINS,
        ):
            return "high"

        if self._matches_domain(
            domain,
            self.COMMUNITY_DOMAINS,
        ):
            return "community"

        if self._matches_domain(
            domain,
            self.THIRD_PARTY_DOMAINS,
        ):
            return "third_party"

        return "unverified"

    @classmethod
    def tokenize(cls, text: str) -> Set[str]:

        tokens = set(
            re.findall(
                r"\b[a-zA-Z0-9_+#.-]+\b",
                text.lower(),
            )
        )

        return {
            token
            for token in tokens
            if token not in cls.STOPWORDS
            and len(token) > 1
        }

    def calculate_relevance(
        self,
        query: str,
        document: Document,
    ) -> float:

        query_tokens = self.tokenize(query)

        if not query_tokens:
            return 0.0

        title = document.metadata.get("title", "")

        document_text = (
            title + " " + document.page_content
        )

        document_tokens = self.tokenize(document_text)

        matching_tokens = query_tokens.intersection(
            document_tokens
        )

        return round(
            len(matching_tokens) / len(query_tokens),
            4,
        )

    def _calculate_ranking_score(
        self,
        document: Document,
    ) -> float:
        """
        Rank documents using trust, relevance, and search score.

        This is a ranking heuristic, not a factual-confidence score.
        """

        trust_level = document.metadata.get(
            "trust_level",
            "unverified",
        )

        trust_score = self.TRUST_PRIORITY.get(
            trust_level,
            0,
        ) / 3

        relevance_score = document.metadata.get(
            "relevance_score",
            0.0,
        ) or 0.0

        search_score = document.metadata.get(
            "score",
            0.0,
        ) or 0.0

        return round(
            (0.40 * trust_score)
            + (0.40 * relevance_score)
            + (0.20 * search_score),
            4,
        )

    def _sort_documents(
        self,
        documents: List[Document],
    ) -> List[Document]:

        return sorted(
            documents,
            key=lambda document: (
                self._calculate_ranking_score(document),
                document.metadata.get("relevance_score") or 0,
                document.metadata.get("score") or 0,
            ),
            reverse=True,
        )

    def validate(
        self,
        documents: List[Document],
        query: str = None,
    ) -> List[Document]:

        validated_documents = []
        seen_urls = set()

        for document in documents:

            metadata = document.metadata.copy()

            url = metadata.get("url", "").strip()
            content = document.page_content.strip()

            # 1. Validate URL and content.
            if not url or not content:
                continue

            parsed = urlparse(url)

            if parsed.scheme not in {"http", "https"}:
                continue

            if not parsed.hostname:
                continue

            # 2. Reject blocked domains.
            domain = self.normalize_domain(url)

            if self._matches_domain(
                domain,
                {
                    item.removeprefix("www.")
                    for item in self.BLOCKED_DOMAINS
                },
            ):
                continue

            # 3. Check minimum content length.
            if len(content) < self.min_content_length:
                continue

            # 4. Calculate relevance.
            if query is not None:

                relevance_score = self.calculate_relevance(
                    query,
                    document,
                )

                if relevance_score < self.min_relevance_score:
                    continue

                metadata["relevance_score"] = relevance_score

            else:
                metadata["relevance_score"] = None

            # 5. Classify source.
            metadata["trust_level"] = self.classify_source(url)

            metadata["source_type"] = "external_web"
            metadata["validation_status"] = "accepted"

            # 6. Canonical URL deduplication.
            normalized_url = self.normalize_url(url)

            if normalized_url in seen_urls:
                continue

            seen_urls.add(normalized_url)

            validated_documents.append(
                Document(
                    page_content=content,
                    metadata=metadata,
                )
            )

        # 7. Rank by source quality and relevance.
        ranked_documents = self._sort_documents(
            validated_documents
        )

        # 8. Limit repeated sources from the same domain.
        if self.max_per_domain is not None:

            domain_counts = {}
            diverse_documents = []

            for document in ranked_documents:

                url = document.metadata.get("url", "")
                domain = self._canonical_domain(
                    self.normalize_domain(url)
                )

                current_count = domain_counts.get(
                    domain,
                    0,
                )

                if current_count >= self.max_per_domain:
                    continue

                domain_counts[domain] = current_count + 1

                diverse_documents.append(document)

            ranked_documents = diverse_documents

        return ranked_documents