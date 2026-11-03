from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.analysis.query_scope_checker import QueryScopeChecker
from src.pipeline.root_cause_pipeline import RootCausePipeline


@pytest.fixture
def pipeline():
    instance = RootCausePipeline.__new__(RootCausePipeline)

    instance.abstention_checker = SimpleNamespace(
        scope_checker=QueryScopeChecker()
    )

    instance.hybrid_retriever = Mock()
    instance.web_retriever = Mock()

    return instance


@pytest.mark.parametrize("query", [
    "Who won the latest cricket match?",
    "Explain how photosynthesis works.",
    "What are the best restaurants in Hyderabad?",
    "What is the production database password?",
    "Solve 25 multiplied by 16.",
])
def test_out_of_scope_queries_skip_retrieval(pipeline, query):

    result = pipeline.run(query)

    assert result["status"] == "abstained"
    assert result["evidence_source"] == "none"
    assert result["abstention"]["reason"] == "out_of_scope"

    pipeline.hybrid_retriever.retrieve.assert_not_called()

    assert pipeline.web_retriever.retrieve.call_count == 0