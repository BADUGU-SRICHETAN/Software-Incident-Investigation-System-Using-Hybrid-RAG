"""
Run end-to-end evaluation of the RootCause AI pipeline.
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timezone

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.root_cause_pipeline import RootCausePipeline
from src.evaluation.pipeline_evaluator import PipelineEvaluator


DATA_DIRECTORY = PROJECT_ROOT / "data" / "raw"

# Use the same persisted vector store as the FastAPI application.
VECTORSTORE_DIRECTORY = (
    PROJECT_ROOT / "vectorstore" / "api_investigation"
)

EVALUATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "pipeline_evaluation_cases.json"
)

REPORT_DIRECTORY = PROJECT_ROOT / "reports" / "evaluation"


def load_evaluation_cases():

    if not EVALUATION_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {EVALUATION_FILE}"
        )

    with open(EVALUATION_FILE, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    cases = dataset.get("cases", [])

    if not cases:
        raise ValueError("No evaluation cases found.")

    return cases


def save_report(report):

    REPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    report["evaluation_timestamp"] = (
        datetime.now(timezone.utc).isoformat()
    )

    report["project"] = "RootCause AI"

    output_file = REPORT_DIRECTORY / "pipeline_evaluation_report.json"

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    return output_file


def main():

    print("\n" + "=" * 60)
    print("ROOTCAUSE AI - PIPELINE EVALUATION")
    print("=" * 60)

    cases = load_evaluation_cases()

    print(f"\nEvaluation cases loaded: {len(cases)}")

    print("\nInitializing RootCause AI pipeline...")

    pipeline = RootCausePipeline(
        data_directory=str(DATA_DIRECTORY),
        persist_directory=str(VECTORSTORE_DIRECTORY),
        model_name="gpt-4o-mini",
    )

    evaluator = PipelineEvaluator(pipeline)

    print("\nStarting evaluation...\n")

    report = evaluator.evaluate(cases)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    metrics = [
        ("Total Cases", "total_cases"),
        ("Completed", "completed"),
        ("Abstained", "abstained"),
        ("Errors", "errors"),
        ("Completion Rate", "completion_rate"),
        ("Abstention Rate", "abstention_rate"),
        ("Error Rate", "error_rate"),
        ("Historical Only", "historical_only_count"),
        ("External Fallback", "external_fallback_count"),
        ("Historical Evidence", "total_historical_evidence"),
        ("External Evidence", "total_external_evidence"),
        ("Trusted External Ratio", "trusted_external_source_ratio"),
        ("Unique External Domains", "unique_external_domains"),
        ("Average Latency (seconds)", "average_latency_seconds"),
        ("Decision Accuracy", "decision_accuracy"),
        ("Labelled Cases", "labelled_cases"),
    ]

    for label, key in metrics:
        print(f"{label:<32}: {report.get(key)}")

    output_file = save_report(report)

    print("\n" + "=" * 60)
    print("EVALUATION REPORT SAVED")
    print("=" * 60)

    print(output_file)

    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()