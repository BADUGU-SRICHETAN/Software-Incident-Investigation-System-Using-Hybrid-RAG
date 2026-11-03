import pytest

from langchain_core.documents import Document

from src.analysis.abstention_checker import AbstentionChecker


@pytest.fixture
def checker():
    return AbstentionChecker(
        vector_score_threshold=0.35,
        score_top_k=3,
        minimum_term_coverage=0.25
    )


def create_document(content, vector_score=None):

    metadata = {}

    if vector_score is not None:
        metadata["vector_score"] = vector_score

    return Document(
        page_content=content,
        metadata=metadata
    )


PAYMENT_EVIDENCE = """
Incident: INC001
Service: Payment Service
Error: ConnectionPoolTimeoutException
Symptoms: Payment requests return HTTP 500 errors.
Root cause: Database connection pool exhausted
because peak payment traffic exceeded pool capacity.
Resolution: Increased pool capacity and configured
connection acquisition timeout.
"""


def test_out_of_scope_query_is_rejected(checker):

    result = checker.check(
        query="What will be the exact number of API requests tomorrow?"
    )

    assert result["accepted"] is False
    assert result["reason"] == "out_of_scope"


def test_empty_retrieved_documents_are_rejected(checker):

    result = checker.check(
        query="Why is the Payment Service returning HTTP 500 errors?",
        retrieved_documents=[]
    )

    assert result["accepted"] is False
    assert result["reason"] == "no_evidence"


def test_weak_evidence_is_rejected(checker):

    documents = [
        create_document(PAYMENT_EVIDENCE, 0.20),
        create_document(PAYMENT_EVIDENCE, 0.25),
        create_document(PAYMENT_EVIDENCE, 0.30)
    ]

    result = checker.check(
        query="Why is the Payment Service returning HTTP 500 errors?",
        retrieved_documents=documents
    )

    assert result["accepted"] is False
    assert result["reason"] == "weak_evidence"


def test_sufficient_evidence_is_accepted(checker):

    documents = [
        create_document(PAYMENT_EVIDENCE, 0.20),
        create_document(PAYMENT_EVIDENCE, 0.42),
        create_document(PAYMENT_EVIDENCE, 0.30)
    ]

    result = checker.check(
        query="Why is the Payment Service returning HTTP 500 errors?",
        retrieved_documents=documents
    )

    assert result["accepted"] is True
    assert result["reason"] == "evidence_available"
    assert result["max_vector_score"] == pytest.approx(0.42)


def test_missing_vector_scores_are_ignored(checker):

    documents = [
        create_document(PAYMENT_EVIDENCE),
        create_document(PAYMENT_EVIDENCE, 0.40),
        create_document(PAYMENT_EVIDENCE, 0.25)
    ]

    result = checker.check(
        query="Why is the Payment Service returning HTTP 500 errors?",
        retrieved_documents=documents
    )

    assert result["accepted"] is True
    assert result["max_vector_score"] == pytest.approx(0.40)


def test_no_vector_scores_are_rejected(checker):

    documents = [
        create_document(PAYMENT_EVIDENCE),
        create_document(PAYMENT_EVIDENCE),
        create_document(PAYMENT_EVIDENCE)
    ]

    result = checker.check(
        query="Why is the Payment Service returning HTTP 500 errors?",
        retrieved_documents=documents
    )

    assert result["accepted"] is False
    assert result["reason"] == "no_vector_score"


def test_only_top_three_documents_are_scored(checker):

    documents = [
        create_document("Unrelated incident evidence.", 0.20),
        create_document("Unrelated incident evidence.", 0.25),
        create_document("Unrelated incident evidence.", 0.30),
        create_document(PAYMENT_EVIDENCE, 0.90)
    ]

    result = checker.check(
        query="Why is the Payment Service returning HTTP 500 errors?",
        retrieved_documents=documents
    )

    assert result["accepted"] is False
    assert result["max_vector_score"] == pytest.approx(0.30)


def test_empty_query_is_rejected(checker):

    with pytest.raises(ValueError):
        checker.check(
            query="",
            retrieved_documents=[]
        )


def test_threshold_boundary_is_accepted(checker):

    documents = [
        create_document(PAYMENT_EVIDENCE, 0.35)
    ]

    result = checker.check(
        query="Why is the Payment Service returning HTTP 500 errors?",
        retrieved_documents=documents
    )

    assert result["accepted"] is True


def test_generic_evidence_triggers_specificity_rejection(checker):

    documents = [
        create_document(
            "Thread pool starvation caused application delays.",
            0.42
        )
    ]

    result = checker.check(
        query=(
            "Apache Kafka consumer group rebalancing occurs "
            "because batch processing exceeds max.poll.interval.ms."
        ),
        retrieved_documents=documents
    )

    assert result["accepted"] is False
    assert result["reason"] == "insufficient_specificity"


def test_kafka_specific_evidence_is_accepted(checker):

    documents = [
        create_document(
            """
            Kafka consumer group repeatedly rebalances.
            Batch processing exceeds max.poll.interval.ms,
            causing the consumer to leave the group.
            """
            ,
            0.50
        )
    ]

    result = checker.check(
        query=(
            "Apache Kafka consumer group rebalancing occurs "
            "because batch processing exceeds max.poll.interval.ms."
        ),
        retrieved_documents=documents
    )

    assert result["accepted"] is True
    assert result["reason"] == "evidence_available"


def test_invalid_vector_score_is_rejected(checker):

    documents = [
        create_document(PAYMENT_EVIDENCE, 1.5)
    ]

    with pytest.raises(ValueError):
        checker.check(
            query="Why is the Payment Service returning HTTP 500 errors?",
            retrieved_documents=documents
        )