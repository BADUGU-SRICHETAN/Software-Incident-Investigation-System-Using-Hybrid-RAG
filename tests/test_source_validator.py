import pytest

from langchain_core.documents import Document

from src.retrieval.source_validator import SourceValidator


def make_document(content, url):
    return Document(
        page_content=content,
        metadata={
            "url": url,
            "title": "Test source"
        }
    )


# --------------------------------------------------
# Original validation tests
# --------------------------------------------------

def test_accepts_valid_http_source():
    validator = SourceValidator(min_content_length=20)

    docs = [
        make_document(
            "This is a sufficiently detailed technical article.",
            "https://example.com/article"
        )
    ]

    result = validator.validate(docs)

    assert len(result) == 1
    assert result[0].metadata["source_type"] == "external_web"
    assert result[0].metadata["validation_status"] == "accepted"


def test_rejects_youtube_source():
    validator = SourceValidator(min_content_length=20)

    docs = [
        make_document(
            "This is a sufficiently detailed technical article.",
            "https://www.youtube.com/watch?v=123"
        )
    ]

    assert validator.validate(docs) == []


def test_rejects_short_content():
    validator = SourceValidator(min_content_length=100)

    docs = [
        make_document(
            "Too short",
            "https://example.com/article"
        )
    ]

    assert validator.validate(docs) == []


def test_rejects_invalid_url():
    validator = SourceValidator(min_content_length=20)

    docs = [
        make_document(
            "This is a sufficiently detailed technical article.",
            "ftp://example.com/article"
        )
    ]

    assert validator.validate(docs) == []


def test_removes_duplicate_urls():
    validator = SourceValidator(min_content_length=20)

    content = "This is a sufficiently detailed technical article."

    docs = [
        make_document(content, "https://example.com/article"),
        make_document(content, "https://example.com/article")
    ]

    result = validator.validate(docs)

    assert len(result) == 1


# --------------------------------------------------
# New relevance filtering tests
# --------------------------------------------------

def test_accepts_relevant_external_document():

    validator = SourceValidator(
        min_content_length=20,
        min_relevance_score=0.15
    )

    query = "Kubernetes liveness probe failure during slow startup"

    document = make_document(
        """
        Kubernetes liveness probes can fail when applications
        experience slow startup. Configure startup probes to
        prevent premature container restarts.
        """,
        "https://kubernetes.io/docs/example"
    )

    result = validator.validate(
        [document],
        query=query
    )

    assert len(result) == 1
    assert result[0].metadata["relevance_score"] >= 0.15
    assert result[0].metadata["validation_status"] == "accepted"


def test_rejects_irrelevant_external_document():

    validator = SourceValidator(
        min_content_length=20,
        min_relevance_score=0.15
    )

    query = "Kubernetes liveness probe failure during slow startup"

    document = make_document(
        """
        Python pandas is a popular data analysis library.
        It provides DataFrame operations for processing
        structured datasets and performing statistical analysis.
        """,
        "https://example.com/python-pandas"
    )

    result = validator.validate(
        [document],
        query=query
    )

    assert result == []


def test_rejects_document_below_relevance_threshold():

    validator = SourceValidator(
        min_content_length=20,
        min_relevance_score=0.80
    )

    query = "Kubernetes liveness probe failure during slow startup"

    document = make_document(
        """
        Kubernetes is an orchestration platform used to manage
        containers across distributed infrastructure.
        """,
        "https://example.com/kubernetes"
    )

    result = validator.validate(
        [document],
        query=query
    )

    assert result == []


def test_relevance_score_is_between_zero_and_one():

    validator = SourceValidator()

    query = "PostgreSQL connection pool exhaustion"

    document = make_document(
        """
        PostgreSQL connection pools manage database connections.
        Exhaustion can cause connection acquisition failures
        and application request timeouts.
        """,
        "https://example.com/postgresql"
    )

    score = validator.calculate_relevance(
        query,
        document
    )

    assert 0.0 <= score <= 1.0


def test_rejects_invalid_relevance_threshold():

    with pytest.raises(ValueError):
        SourceValidator(min_relevance_score=1.5)

# --------------------------------------------------
# Source trust classification tests
# --------------------------------------------------

@pytest.mark.parametrize(
    "url, expected_trust",
    [
        ("https://kafka.apache.org/documentation/", "high"),
        ("https://docs.redhat.com/en/documentation/", "high"),
        ("https://strimzi.io/documentation/", "high"),
        ("https://docs.python.org/3/", "high"),
        ("https://stackoverflow.com/questions/123", "community"),
        ("https://github.com/apache/kafka/issues/123", "community"),
        ("https://www.netdata.cloud/guides/kafka/", "third_party"),
        ("https://acemq.com/blogs/kafka-rebalance", "third_party"),
        ("https://example.com/article", "unverified"),
    ],
)
def test_classifies_source_trust_level(url, expected_trust):

    validator = SourceValidator()

    assert validator.classify_source(url) == expected_trust


def test_does_not_trust_similar_looking_domain():

    validator = SourceValidator()

    assert (
        validator.classify_source(
            "https://fake-redhat.com/article"
        )
        == "unverified"
    )


def test_trust_classification_supports_subdomains():

    validator = SourceValidator()

    assert (
        validator.classify_source(
            "https://docs.redhat.com/article"
        )
        == "high"
    )