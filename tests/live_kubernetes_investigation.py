from src.pipeline.root_cause_pipeline import RootCausePipeline


def main():

    print("\n========== ROOTCAUSE AI INVESTIGATION ==========")

    # 1. Initialize the pipeline
    print("\nInitializing RootCause AI Pipeline...")

    pipeline = RootCausePipeline(
        data_directory="data/raw",
        persist_directory="vectorstore/rootcause_chroma",
        model_name="gpt-4o-mini"
    )

    # 2. Define the incident query
    query = """
    Our Kubernetes application is experiencing repeated pod
    restarts because liveness probes are failing after deployment.
    The application starts slowly, and Kubernetes terminates
    containers before they finish initializing.

    How can we investigate this incident and identify the
    potential causes and appropriate solutions?
    """

    print("\n========== INCIDENT QUERY ==========")
    print(query.strip())

    # 3. Execute the investigation
    print("\nRunning investigation...")

    result = pipeline.run(
        query=query,
        top_k=3,
        candidate_k=5
    )

    # 4. Display investigation results
    print("\n========== INVESTIGATION RESULT ==========")

    print("\nStatus:", result.get("status", "unknown"))

    print(
        "Evidence source:",
        result.get("evidence_source", "not available")
    )

    print(
        "Historical documents:",
        result.get("retrieved_count", 0)
    )

    print(
        "External documents:",
        result.get("external_retrieved_count", 0)
    )

    # 5. Display historical evidence assessment
    abstention = result.get("abstention") or {}

    print("\n========== HISTORICAL EVIDENCE ASSESSMENT ==========")

    print(
        "Historical evidence accepted:",
        abstention.get("accepted", "not available")
    )

    print(
        "Reason:",
        abstention.get("reason", "not available")
    )

    print(
        "Category:",
        abstention.get("category", "not available")
    )

    print(
        "Assessment message:",
        abstention.get("message", "not available")
    )

    if abstention.get("max_vector_score") is not None:
        print(
            "Maximum vector score:",
            abstention["max_vector_score"]
        )

    if abstention.get("term_coverage") is not None:
        print(
            "Term coverage:",
            abstention["term_coverage"]
        )

    # 6. Handle unsuccessful investigations
    if result.get("status") != "completed":

        print("\nInvestigation could not be completed.")

        print(
            "Reason:",
            result.get("reason", "Unknown")
        )

        return

    # 7. Display external technical sources
    external_sources = result.get("external_evidence") or []

    print("\n========== EXTERNAL TECHNICAL SOURCES ==========")

    if external_sources:

        for index, source in enumerate(external_sources, start=1):

            print(f"\nSource {index}")

            print(
                "Title:",
                source.get("title", "Not available")
            )

            print(
                "Trust level:",
                source.get("trust_level", "unverified")
            )

            print(
                "Relevance score:",
                source.get("relevance_score", "Not available")
            )

            print(
                "Tavily search score:",
                source.get("score", "Not available")
            )

            print(
                "Published date:",
                source.get("published_date", "Not available")
            )

            print(
                "URL:",
                source.get("url", "Not available")
            )

    else:

        print("\nNo external sources were used.")

    # 8. Display AI-generated investigation report
    print("\n========== AI INVESTIGATION ANALYSIS ==========")

    analysis = result.get("analysis")

    if analysis:
        print(analysis)
    else:
        print("No investigation analysis was generated.")

    # 9. Final investigation summary
    print("\n========== FINAL SUMMARY ==========")

    if result.get("status") == "completed":

        print("Investigation completed successfully.")

        evidence_source = result.get("evidence_source")

        if evidence_source == "historical_and_external":

            print(
                "Evidence used: Historical incident records "
                "and external technical documentation."
            )

        elif evidence_source == "historical":

            print(
                "Evidence used: Historical incident records."
            )

        else:

            print(
                "Evidence used:",
                evidence_source or "Not available"
            )

        print(
            "External sources used:",
            len(external_sources)
        )

        print(
            "Historical evidence sufficient:",
            abstention.get("accepted", "not available")
        )

        print(
            "\nNote: Suggested root causes are hypotheses "
            "until confirmed using actual incident logs, "
            "configuration files and diagnostic evidence."
        )

    else:

        print("Investigation was not completed.")

    print("\n========== END OF INVESTIGATION ==========")


if __name__ == "__main__":
    main()