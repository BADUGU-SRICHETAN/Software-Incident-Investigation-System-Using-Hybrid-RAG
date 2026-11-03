from typing import Dict, Any, List

from langchain_core.documents import Document

from src.ingestion.document_loader import DocumentLoader
from src.preprocessing.text_cleaner import TextCleaner

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.vector_retriever import VectorRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.web_retriever import WebRetriever
from src.retrieval.source_validator import SourceValidator

from src.analysis.root_cause_analyzer import RootCauseAnalyzer
from src.analysis.abstention_checker import AbstentionChecker


class RootCausePipeline:

    def __init__(
        self,
        data_directory: str = "data/raw",
        persist_directory: str = "vectorstore/rootcause_chroma",
        model_name: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small"
    ):

        print("\nInitializing RootCause AI Pipeline...")

        # 1. Load historical incident documents
        self.loader = DocumentLoader(data_directory)

        documents = self.loader.load_documents()

        if not documents:
            raise ValueError(
                "No documents were loaded. Check the data directory."
            )

        # 2. Clean documents
        self.cleaner = TextCleaner()

        cleaned_documents = self.cleaner.clean_documents(documents)

        if not cleaned_documents:
            raise ValueError(
                "No valid documents remain after preprocessing."
            )

        print(
            f"Documents after preprocessing: {len(cleaned_documents)}"
        )

        # 3. Initialize BM25
        self.bm25_retriever = BM25Retriever(
            cleaned_documents
        )

        # 4. Initialize vector retrieval
        self.vector_retriever = VectorRetriever(
            documents=cleaned_documents,
            persist_directory=persist_directory,
            model_name=embedding_model,
            collection_name="rootcause_incidents_openai"
        )

        # 5. Initialize Hybrid RAG
        self.hybrid_retriever = HybridRetriever(
            bm25_retriever=self.bm25_retriever,
            vector_retriever=self.vector_retriever
        )

        # 6. Initialize abstention checker
        self.abstention_checker = AbstentionChecker(
            vector_score_threshold=0.35,
            score_top_k=3
        )

        # 7. Initialize analyzer
        self.analyzer = RootCauseAnalyzer(
            model_name=model_name
        )

        # 8. Initialize external retrieval lazily
        self.web_retriever = None

        self.source_validator = SourceValidator(
            min_content_length=100
        )

        print("\nRootCause AI Pipeline initialized successfully!")
        print(f"Embedding model: {embedding_model}")
        print(f"LLM model: {model_name}")
        print("External web retrieval: enabled")

    # --------------------------------------------------
    # Lazy Web Retriever Initialization
    # --------------------------------------------------

    def _get_web_retriever(self) -> WebRetriever:

        if self.web_retriever is None:

            self.web_retriever = WebRetriever(
                max_results=5,
                search_depth="advanced"
            )

        return self.web_retriever

    # --------------------------------------------------
    # Format Historical Evidence
    # --------------------------------------------------

    @staticmethod
    def _format_historical_evidence(
        documents: List[Document]
    ) -> List[Dict[str, Any]]:

        evidence = []

        for document in documents:

            metadata = document.metadata

            evidence.append({
                "incident_id": metadata.get("incident_id"),
                "source": metadata.get("source"),
                "content": document.page_content,
                "rrf_score": metadata.get("rrf_score"),
                "vector_score": metadata.get("vector_score"),
                "bm25_score": metadata.get("bm25_score"),
                "retrieval_ranks": metadata.get("retrieval_ranks"),
                "source_type": "historical"
            })

        return evidence

    # --------------------------------------------------
    # Format External Evidence
    # --------------------------------------------------

    @staticmethod
    def _format_external_evidence(
        documents: List[Document]
    ) -> List[Dict[str, Any]]:

        evidence = []

        for document in documents:

            metadata = document.metadata

            evidence.append({
                "title": metadata.get("title"),
                "url": metadata.get("url"),
                "content": document.page_content,
                "trust_level": metadata.get("trust_level"),
                "relevance_score": metadata.get("relevance_score"),
                "validation_status": metadata.get(
                    "validation_status"
                ),
                "score": metadata.get("score"),
                "published_date": metadata.get("published_date"),
                "source_type": metadata.get(
                    "source_type",
                    "external_web"
                )
            })

        return evidence

    # --------------------------------------------------
    # Main Investigation Pipeline
    # --------------------------------------------------

    def run(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 10
    ) -> Dict[str, Any]:

        if not query or not query.strip():
            raise ValueError(
                "Incident query cannot be empty."
            )

        # 1. Check query scope
        print("\nChecking query scope...")

        scope_result = self.abstention_checker.scope_checker.check(
            query
        )

        if not scope_result["in_scope"]:

            print("\nInvestigation rejected: Query is out of scope.")

            return {
                "query": query,
                "status": "abstained",
                "retrieved_count": 0,
                "retrieved_evidence": [],
                "external_retrieved_count": 0,
                "external_evidence": [],
                "evidence_source": "none",
                "abstention": {
                    "accepted": False,
                    "reason": "out_of_scope",
                    "category": scope_result["category"],
                    "message": scope_result["reason"],
                    "max_vector_score": None,
                    "threshold": 0.35
                },
                "analysis": (
                    "Investigation cannot proceed because the "
                    "question is outside the software incident "
                    "investigation scope."
                )
            }

        # 2. Retrieve historical incidents
        print("\nSearching historical incidents...")

        retrieved_documents = self.hybrid_retriever.retrieve(
            query=query,
            top_k=top_k,
            candidate_k=candidate_k
        )

        print(
            f"Retrieved {len(retrieved_documents)} historical documents."
        )

        # 3. Evaluate historical evidence
        print("\nEvaluating historical evidence...")

        abstention_result = self.abstention_checker.check(
            query=query,
            retrieved_documents=retrieved_documents
        )

        historical_evidence = self._format_historical_evidence(
            retrieved_documents
        )

        # 4. Historical evidence is sufficient
        if abstention_result["accepted"]:

            print("\nHistorical evidence is sufficient.")
            print("\nAnalyzing historical evidence...")

            analysis = self.analyzer.analyze(
                query=query,
                retrieved_documents=retrieved_documents
            )

            return {
                "query": query,
                "status": "completed",
                "retrieved_count": len(retrieved_documents),
                "retrieved_evidence": historical_evidence,
                "external_retrieved_count": 0,
                "external_evidence": [],
                "evidence_source": "historical",
                "abstention": abstention_result,
                "analysis": analysis
            }

        # 5. Historical evidence is insufficient.
        # Trigger external web fallback.
        print("\nHistorical evidence is insufficient.")
        print("\nStarting external web retrieval...")

        try:

            web_retriever = self._get_web_retriever()

            external_documents = web_retriever.retrieve(
                query=query
            )

            print(
                f"Retrieved {len(external_documents)} external documents."
            )

            # 6. Validate external sources
            print("\nValidating external sources...")

            validated_documents = self.source_validator.validate(
                external_documents,
                query=query
            )

            print(
                f"Validated {len(validated_documents)} external documents."
            )

        except Exception as error:

            print(f"\nExternal retrieval failed: {error}")

            return {
                "query": query,
                "status": "abstained",
                "retrieved_count": len(retrieved_documents),
                "retrieved_evidence": historical_evidence,
                "external_retrieved_count": 0,
                "external_evidence": [],
                "evidence_source": "none",
                "abstention": {
                    **abstention_result,
                    "accepted": False,
                    "reason": "external_retrieval_failed",
                    "message": (
                        "Historical evidence was insufficient and "
                        "external retrieval failed."
                    )
                },
                "analysis": (
                    "Investigation abstained because historical "
                    "evidence was insufficient and external "
                    "retrieval failed."
                )
            }

        external_evidence = self._format_external_evidence(
            validated_documents
        )

        # 7. No valid external evidence
        if not validated_documents:

            print("\nNo valid external evidence found.")

            return {
                "query": query,
                "status": "abstained",
                "retrieved_count": len(retrieved_documents),
                "retrieved_evidence": historical_evidence,
                "external_retrieved_count": 0,
                "external_evidence": [],
                "evidence_source": "none",
                "abstention": {
                    **abstention_result,
                    "accepted": False,
                    "reason": "no_valid_external_evidence",
                    "message": (
                        "No external sources passed validation."
                    )
                },
                "analysis": (
                    "Insufficient evidence: No validated external "
                    "technical sources were found."
                )
            }

        # 8. Analyze historical and external evidence
        print("\nExternal evidence is available.")
        print("\nAnalyzing combined evidence...")

        try:

            analysis = self.analyzer.analyze_with_external_evidence(
                query=query,
                historical_documents=retrieved_documents,
                external_documents=validated_documents
            )

        except Exception as error:

            print(f"\nExternal evidence analysis failed: {error}")

            return {
                "query": query,
                "status": "abstained",
                "retrieved_count": len(retrieved_documents),
                "retrieved_evidence": historical_evidence,
                "external_retrieved_count": len(validated_documents),
                "external_evidence": external_evidence,
                "evidence_source": "historical_and_external",
                "abstention": {
                    **abstention_result,
                    "accepted": False,
                    "reason": "analysis_failed",
                    "message": (
                        "Evidence was retrieved, but the analysis "
                        "could not be completed."
                    )
                },
                "analysis": (
                    "Investigation could not be completed because "
                    "the analysis step failed."
                )
            }

        # 9. Return final report
        result = {
            "query": query,
            "status": "completed",
            "retrieved_count": len(retrieved_documents),
            "retrieved_evidence": historical_evidence,
            "external_retrieved_count": len(validated_documents),
            "external_evidence": external_evidence,
            "evidence_source": "historical_and_external",

            # Historical abstention and overall investigation
            # outcome are represented separately.
            "abstention": {
                **abstention_result,
                "accepted": False,
                "reason": "historical_evidence_insufficient",
                "message": (
                    "Historical evidence was insufficient. "
                    "External technical sources were retrieved "
                    "and used for supplementary investigation."
                )
            },

            "analysis": analysis
        }

        print("\nInvestigation completed using external evidence.")

        return result