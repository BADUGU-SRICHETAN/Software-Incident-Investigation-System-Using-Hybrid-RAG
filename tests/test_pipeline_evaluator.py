import pytest

from src.evaluation.pipeline_evaluator import PipelineEvaluator


class MockPipeline:

    def __init__(self, results):
        self.results = results
        self.call_count = 0

    def run(self, query):
        result = self.results[self.call_count]
        self.call_count += 1
        return result


@pytest.fixture
def mock_results():

    return [
        {
            "status": "completed",
            "evidence_source": "historical",
            "retrieved_evidence": [
                {"incident_id": "INC-001"}
            ],
            "external_evidence": [],
        },
        {
            "status": "completed",
            "evidence_source": "historical_and_external",
            "retrieved_evidence": [
                {"incident_id": "INC-002"}
            ],
            "external_evidence": [
                {
                    "url": "https://kubernetes.io/docs/concepts/",
                    "trust_level": "high",
                },
                {
                    "url": "https://docs.python.org/3/",
                    "trust_level": "high",
                },
            ],
        },
        {
            "status": "abstained",
            "evidence_source": "none",
            "retrieved_evidence": [],
            "external_evidence": [],
        },
    ]


def test_evaluate_pipeline(mock_results):

    pipeline = MockPipeline(mock_results)

    evaluator = PipelineEvaluator(pipeline)

    cases = [
        {
            "query": "Why did the database fail?",
            "expected_status": "completed",
            "expected_evidence_source": "historical",
        },
        {
            "query": "Why are Kubernetes pods restarting?",
            "expected_status": "completed",
            "expected_evidence_source": "historical_and_external",
        },
        {
            "query": "What is the weather tomorrow?",
            "expected_status": "abstained",
            "expected_evidence_source": "none",
        },
    ]

    report = evaluator.evaluate(cases)

    assert report["total_cases"] == 3
    assert report["completed"] == 2
    assert report["abstained"] == 1
    assert report["errors"] == 0

    assert report["completion_rate"] == pytest.approx(2 / 3, abs=0.0001)
    assert report["abstention_rate"] == pytest.approx(1 / 3, abs=0.0001)

    assert report["historical_only_count"] == 1
    assert report["external_fallback_count"] == 1

    assert report["total_historical_evidence"] == 2
    assert report["total_external_evidence"] == 2

    assert report["trusted_external_source_ratio"] == 1.0
    assert report["unique_external_domains"] == 2

    assert report["decision_accuracy"] == 1.0
    assert report["labelled_cases"] == 3


def test_external_fallback_is_not_counted_as_abstention():

    pipeline = MockPipeline([
        {
            "status": "completed",
            "evidence_source": "historical_and_external",
            "retrieved_evidence": [],
            "external_evidence": [
                {
                    "url": "https://kubernetes.io/docs/",
                    "trust_level": "high",
                }
            ],
            "abstention": {
                "accepted": False,
                "reason": "historical_evidence_insufficient",
            },
        }
    ])

    evaluator = PipelineEvaluator(pipeline)

    report = evaluator.evaluate([
        {"query": "Why are pods restarting?"}
    ])

    assert report["completed"] == 1
    assert report["abstained"] == 0
    assert report["external_fallback_count"] == 1


def test_empty_evaluation_cases():

    pipeline = MockPipeline([])

    evaluator = PipelineEvaluator(pipeline)

    with pytest.raises(ValueError):
        evaluator.evaluate([])


def test_missing_query():

    pipeline = MockPipeline([])

    evaluator = PipelineEvaluator(pipeline)

    with pytest.raises(ValueError):
        evaluator.evaluate([
            {"expected_status": "completed"}
        ])


def test_pipeline_exception_is_recorded():

    class FailingPipeline:

        def run(self, query):
            raise RuntimeError("Pipeline failed")

    evaluator = PipelineEvaluator(FailingPipeline())

    report = evaluator.evaluate([
        {"query": "Why did the service crash?"}
    ])

    assert report["errors"] == 1
    assert report["error_rate"] == 1.0
    assert report["case_results"][0]["status"] == "error"