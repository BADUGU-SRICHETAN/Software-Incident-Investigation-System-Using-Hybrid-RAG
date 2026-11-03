from types import SimpleNamespace

from langchain_core.documents import Document

from src.pipeline.root_cause_pipeline import RootCausePipeline
from src.retrieval.source_validator import SourceValidator


def test_web_fallback_when_historical_evidence_is_insufficient():

    # 1. Create pipeline without running its expensive initialization
    pipeline = RootCausePipeline.__new__(RootCausePipeline)

    # 2. Mock query scope validation
    pipeline.abstention_checker = SimpleNamespace(
        scope_checker=SimpleNamespace(
            check=lambda query: {
                "in_scope": True,
                "category": "software_incident",
                "reason": "Valid software incident query."
            }
        ),
        check=lambda query, retrieved_documents: {
            "accepted": False,
            "reason": "weak_evidence",
            "category": "software_incident",
            "message": "Historical evidence is insufficient.",
            "max_vector_score": 0.10,
            "threshold": 0.35
        }
    )

    # 3. Mock historical retrieval
    historical_document = Document(
        page_content="A historical incident with limited relevance.",
        metadata={
            "incident_id": "INC001",
            "source": "incidents.json",
            "vector_score": 0.10,
            "bm25_score": 0.05,
            "rrf_score": 0.02
        }
    )

    pipeline.hybrid_retriever = SimpleNamespace(
        retrieve=lambda query, top_k, candidate_k: [
            historical_document
        ]
    )

    # 4. Mock external web retrieval
    external_document = Document(
        page_content=(
            "Kubernetes liveness probes determine whether a container "
            "is healthy. If a liveness probe repeatedly fails, "
            "Kubernetes restarts the container. Applications with "
            "long startup times should use startup probes to prevent "
            "premature liveness probe failures. Probe timing and "
            "failure thresholds should be configured appropriately."
        ),
        metadata={
            "title": "Kubernetes Probe Configuration",
            "url": (
                "https://kubernetes.io/docs/concepts/configuration/"
                "liveness-readiness-startup-probes/"
            ),
            "score": 0.95,
            "published_date": None
        }
    )

    mock_web_retriever = SimpleNamespace(
        retrieve=lambda query: [external_document]
    )

    pipeline.web_retriever = mock_web_retriever

    # 5. Initialize the real source validator
    pipeline.source_validator = SourceValidator(
        min_content_length=100
    )

    # 6. Mock the analyzer
    analyzer_called = {"value": False}

    def mock_external_analysis(
        query,
        historical_documents,
        external_documents
    ):
        analyzer_called["value"] = True

        assert len(historical_documents) == 1
        assert len(external_documents) == 1

        return "Mock investigation completed using combined evidence."

    pipeline.analyzer = SimpleNamespace(
        analyze_with_external_evidence=mock_external_analysis
    )

    # 7. Run the investigation
    result = pipeline.run(
        query=(
            "Our Kubernetes application keeps restarting because "
            "liveness probes fail during slow startup."
        ),
        top_k=3,
        candidate_k=5
    )

    # 8. Verify investigation completion
    assert result["status"] == "completed"

    assert result["evidence_source"] == "historical_and_external"

    assert result["external_retrieved_count"] == 1

    # 9. Verify historical evidence assessment
    assert result["abstention"]["accepted"] is False

    assert result["abstention"]["reason"] == (
        "historical_evidence_insufficient"
    )

    # 10. Verify external analysis was executed
    assert analyzer_called["value"] is True

    # 11. Verify external evidence metadata
    assert result["external_evidence"][0]["title"] == (
        "Kubernetes Probe Configuration"
    )

    assert result["external_evidence"][0]["source_type"] == (
        "external_web"
    )

    print("\nWeb fallback test passed successfully!")


if __name__ == "__main__":
    test_web_fallback_when_historical_evidence_is_insufficient()