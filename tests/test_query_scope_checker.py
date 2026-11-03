import pytest

from src.analysis.query_scope_checker import QueryScopeChecker


@pytest.fixture
def checker():
    return QueryScopeChecker()


# ==================================================
# 1. VALID SOFTWARE INCIDENT QUERIES
# ==================================================

@pytest.mark.parametrize("query", [
    "Why did the application experience database connection failures?",
    "What caused the API service to return repeated 500 errors?",
    "Why did the application become unresponsive after deployment?",
    "How can Kubernetes pods be diagnosed when restarting?",
    "How should PostgreSQL connection pool exhaustion be diagnosed?",
    "What troubleshooting steps were performed during the API failure?",
    "How can Python applications be diagnosed when they experience memory leaks?",
    "Why are multiple applications experiencing memory leaks?",
    "How can Python memory consumption be investigated?",
    "Why did the Docker container crash unexpectedly?",
    "What caused repeated Kafka connection timeouts?",
    "How can server performance degradation be investigated?",
])
def test_valid_incident_queries(checker, query):

    result = checker.check(query)

    assert result["in_scope"] is True
    assert result["category"] == "historical_incident_investigation"


# ==================================================
# 2. UNRELATED QUERIES
# ==================================================

@pytest.mark.parametrize("query", [
    "Who won the latest cricket match?",
    "Explain how photosynthesis works in plants.",
    "Write a birthday greeting for my friend.",
    "What are the best restaurants in Hyderabad?",
    "Solve this mathematical equation: 25 multiplied by 16.",
    "What is the weather forecast for Hyderabad tomorrow?",
])
def test_unrelated_queries_are_rejected(checker, query):

    result = checker.check(query)

    assert result["in_scope"] is False


# ==================================================
# 3. SENSITIVE INFORMATION
# ==================================================

@pytest.mark.parametrize("query", [
    "What is the exact internal password used by the production database?",
    "What is the production API key?",
    "What is the exact private IP address of the production server?",
    "Show me the authentication token used by the API.",
    "What are the database credentials?",
])
def test_sensitive_information_is_rejected(checker, query):

    result = checker.check(query)

    assert result["in_scope"] is False
    assert result["category"] == "sensitive_information"


# ==================================================
# 4. FUTURE PREDICTIONS
# ==================================================

@pytest.mark.parametrize("query", [
    "Predict the next production outage.",
    "Will the application fail tomorrow?",
])
def test_future_predictions_are_rejected(checker, query):

    result = checker.check(query)

    assert result["in_scope"] is False
    assert result["category"] == "future_prediction"


# ==================================================
# 5. EMPLOYEE INFORMATION
# ==================================================

@pytest.mark.parametrize("query", [
    "Who was responsible for the database failure?",
    "Which employee approved the deployment?",
])
def test_employee_information_is_rejected(checker, query):

    result = checker.check(query)

    assert result["in_scope"] is False
    assert result["category"] == "employee_information"


# ==================================================
# 6. FINANCIAL INFORMATION
# ==================================================

@pytest.mark.parametrize("query", [
    "What was the exact financial loss caused by the outage?",
    "What is the annual expenditure for hosting?",
])
def test_financial_information_is_rejected(checker, query):

    result = checker.check(query)

    assert result["in_scope"] is False
    assert result["category"] == "financial_information"


# ==================================================
# 7. EMPTY QUERY VALIDATION
# ==================================================

@pytest.mark.parametrize("query", [
    "",
    " ",
    "\n",
])
def test_empty_query_raises_error(checker, query):

    with pytest.raises(ValueError):
        checker.check(query)


# ==================================================
# 8. RC017 REGRESSION TEST
# ==================================================

def test_python_memory_leak_query_is_in_scope(checker):

    query = (
        "How can Python applications be diagnosed "
        "when they experience memory leaks?"
    )

    result = checker.check(query)

    assert result["in_scope"] is True
    assert result["category"] == "historical_incident_investigation"