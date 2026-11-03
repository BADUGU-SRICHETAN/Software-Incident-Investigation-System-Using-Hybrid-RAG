"""
End-to-end evaluation for the RootCause AI pipeline.
"""

import time
from urllib.parse import urlparse


class PipelineEvaluator:

    TRUSTED_LEVELS = {"high"}

    def __init__(self, pipeline):
        self.pipeline = pipeline

    @staticmethod
    def _get_domain(document):
        url = document.get("url", "")

        if not url:
            return None

        domain = urlparse(url).netloc.lower()
        return domain.removeprefix("www.") or None

    def evaluate_case(self, case):

        query = case.get("query")

        if not isinstance(query, str) or not query.strip():
            raise ValueError(
                "Each evaluation case must contain a valid query."
            )

        start_time = time.perf_counter()

        try:
            result = self.pipeline.run(query)

        except Exception as exc:
            latency = time.perf_counter() - start_time

            return {
                "query": query,
                "status": "error",
                "evidence_source": "none",
                "historical_evidence_count": 0,
                "external_evidence_count": 0,
                "trusted_external_count": 0,
                "trusted_external_ratio": 0.0,
                "unique_external_domains": 0,
                "latency_seconds": round(latency, 4),
                "decision_correct": None,
                "error": str(exc),
            }

        latency = time.perf_counter() - start_time

        status = result.get("status", "unknown")
        evidence_source = result.get("evidence_source", "none")

        historical = result.get("retrieved_evidence") or []
        external = result.get("external_evidence") or []

        domains = {
            domain
            for document in external
            if (domain := self._get_domain(document))
        }

        trusted_count = sum(
            1
            for document in external
            if str(document.get("trust_level", "")).lower()
            in self.TRUSTED_LEVELS
        )

        trusted_ratio = (
            trusted_count / len(external)
            if external else 0.0
        )

        decision_correct = None

        expected_status = case.get("expected_status")

        if expected_status is not None:
            decision_correct = status == expected_status

        expected_source = case.get("expected_evidence_source")

        if expected_source is not None:
            source_correct = evidence_source == expected_source

            decision_correct = (
                source_correct
                if decision_correct is None
                else decision_correct and source_correct
            )

        return {
            "query": query,
            "status": status,
            "evidence_source": evidence_source,
            "historical_evidence_count": len(historical),
            "external_evidence_count": len(external),
            "trusted_external_count": trusted_count,
            "trusted_external_ratio": round(trusted_ratio, 4),
            "unique_external_domains": len(domains),
            "latency_seconds": round(latency, 4),
            "decision_correct": decision_correct,
        }

    def evaluate(self, evaluation_cases):

        if not evaluation_cases:
            raise ValueError("Evaluation cases cannot be empty.")

        results = [
            self.evaluate_case(case)
            for case in evaluation_cases
        ]

        total = len(results)

        completed = sum(
            result["status"] == "completed"
            for result in results
        )

        abstained = sum(
            result["status"] == "abstained"
            for result in results
        )

        errors = sum(
            result["status"] == "error"
            for result in results
        )

        historical_only = sum(
            result["evidence_source"] == "historical"
            and result["status"] == "completed"
            for result in results
        )

        external_fallback = sum(
            result["evidence_source"] == "historical_and_external"
            and result["status"] == "completed"
            for result in results
        )

        total_external = sum(
            result["external_evidence_count"]
            for result in results
        )

        trusted_external = sum(
            result["trusted_external_count"]
            for result in results
        )

        total_domains = sum(
            result["unique_external_domains"]
            for result in results
        )

        latencies = [
            result["latency_seconds"]
            for result in results
        ]

        labelled_results = [
            result["decision_correct"]
            for result in results
            if result["decision_correct"] is not None
        ]

        return {
            "total_cases": total,
            "completed": completed,
            "abstained": abstained,
            "errors": errors,

            "completion_rate": round(completed / total, 4),
            "abstention_rate": round(abstained / total, 4),
            "error_rate": round(errors / total, 4),

            "historical_only_count": historical_only,
            "external_fallback_count": external_fallback,

            "external_fallback_rate": round(
                external_fallback / total, 4
            ),

            "total_historical_evidence": sum(
                result["historical_evidence_count"]
                for result in results
            ),

            "total_external_evidence": total_external,

            "trusted_external_source_ratio": round(
                trusted_external / total_external, 4
            ) if total_external else 0.0,

            "unique_external_domains": total_domains,

            "average_external_sources_per_case": round(
                total_external / total, 2
            ),

            "average_latency_seconds": round(
                sum(latencies) / total, 4
            ),

            "decision_accuracy": round(
                sum(labelled_results) / len(labelled_results), 4
            ) if labelled_results else None,

            "labelled_cases": len(labelled_results),

            "case_results": results,
        }