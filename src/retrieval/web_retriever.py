import os
from typing import List, Optional
from urllib.parse import urlparse

from dotenv import load_dotenv
from langchain_core.documents import Document
from tavily import TavilyClient


class WebRetriever:
    """
    Retrieves external technical evidence using Tavily.

    Retrieval strategy:
    1. Search trusted technical documentation domains.
    2. Perform a broader web search to find additional evidence.
    3. Remove duplicate URLs.
    4. Preserve source metadata for validation and ranking.

    External results remain separate from historical incidents.
    """

    # Domains prioritized for official technical documentation.
    PREFERRED_DOMAINS = [
        "kubernetes.io",
        "kafka.apache.org",
        "apache.org",
        "redhat.com",
        "learn.microsoft.com",
        "docs.docker.com",
        "docs.python.org",
        "postgresql.org",
        "owasp.org",
        "nvd.nist.gov",
        "docs.github.com",
    ]

    def __init__(
        self,
        max_results: int = 5,
        search_depth: str = "advanced"
    ):
        load_dotenv()

        api_key = os.getenv("TAVILY_API_KEY")

        if not api_key:
            raise ValueError(
                "TAVILY_API_KEY is missing from the environment."
            )

        if not isinstance(max_results, int) or max_results <= 0:
            raise ValueError(
                "max_results must be a positive integer."
            )

        if search_depth not in {"basic", "advanced"}:
            raise ValueError(
                "search_depth must be 'basic' or 'advanced'."
            )

        self.client = TavilyClient(api_key=api_key)
        self.max_results = max_results
        self.search_depth = search_depth

    @staticmethod
    def _normalize_url(url: str) -> str:
        """Normalize URLs for duplicate detection."""

        parsed = urlparse(url)

        return (
            f"{parsed.scheme.lower()}://"
            f"{parsed.netloc.lower()}"
            f"{parsed.path.rstrip('/')}"
        )

    def _search(
        self,
        query: str,
        domains: Optional[List[str]] = None
    ) -> List[dict]:
        """Execute one Tavily search."""

        arguments = {
            "query": query,
            "max_results": self.max_results,
            "search_depth": self.search_depth,
            "include_answer": False,
            "include_raw_content": False,
        }

        if domains:
            arguments["include_domains"] = domains

        try:
            response = self.client.search(**arguments)

        except Exception as error:
            raise RuntimeError(
                f"External web search failed: {error}"
            ) from error

        return response.get("results", [])

    def _convert_results(
        self,
        results: List[dict],
        search_tier: str
    ) -> List[Document]:
        """Convert Tavily results into LangChain documents."""

        documents = []

        for result in results:

            content = result.get("content", "").strip()
            url = result.get("url", "").strip()
            title = result.get("title", "").strip()

            if not content or not url:
                continue

            parsed = urlparse(url)

            if parsed.scheme not in {"http", "https"}:
                continue

            if not parsed.hostname:
                continue

            metadata = {
                "source_type": "external_web",
                "title": title,
                "url": url,
                "source": url,
                "score": result.get("score"),
                "published_date": result.get("published_date"),
                "search_tier": search_tier,
            }

            documents.append(
                Document(
                    page_content=content,
                    metadata=metadata
                )
            )

        return documents

    def retrieve(
        self,
        query: str,
        domains: Optional[List[str]] = None
    ) -> List[Document]:

        if not isinstance(query, str) or not query.strip():
            raise ValueError(
                "Web search query cannot be empty."
            )

        clean_query = query.strip()

        search_query = (
            f"{clean_query} "
            "technical troubleshooting documentation "
            "configuration error diagnosis resolution"
        )

        documents = []
        seen_urls = set()

        # --------------------------------------------------
        # 1. Prioritize official technical documentation
        # --------------------------------------------------

        preferred_domains = domains or self.PREFERRED_DOMAINS

        print("\nSearching preferred technical documentation...")

        preferred_results = self._search(
            query=search_query,
            domains=preferred_domains
        )

        preferred_documents = self._convert_results(
            preferred_results,
            search_tier="preferred_documentation"
        )

        for document in preferred_documents:

            normalized_url = self._normalize_url(
                document.metadata["url"]
            )

            if normalized_url not in seen_urls:
                seen_urls.add(normalized_url)
                documents.append(document)

        # --------------------------------------------------
        # 2. Broader web search for additional evidence
        # --------------------------------------------------

        remaining = self.max_results - len(documents)

        if remaining > 0:

            print("\nSearching broader technical sources...")

            broad_results = self._search(
                query=search_query
            )

            broad_documents = self._convert_results(
                broad_results,
                search_tier="broader_web"
            )

            for document in broad_documents:

                normalized_url = self._normalize_url(
                    document.metadata["url"]
                )

                if normalized_url not in seen_urls:
                    seen_urls.add(normalized_url)
                    documents.append(document)

                if len(documents) >= self.max_results:
                    break

        # --------------------------------------------------
        # 3. Return prioritized external evidence
        # --------------------------------------------------

        return documents[:self.max_results]