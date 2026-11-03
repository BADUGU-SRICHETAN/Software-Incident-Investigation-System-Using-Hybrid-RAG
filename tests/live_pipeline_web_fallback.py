from src.pipeline.root_cause_pipeline import RootCausePipeline


def main():

    print("\n========== ROOTCAUSE AI LIVE PIPELINE TEST ==========")

    pipeline = RootCausePipeline()

    query = (
    "Apache Kafka consumer applications repeatedly experience "
    "consumer group rebalancing because batch processing exceeds "
    "the max.poll.interval.ms configuration. How can this issue "
    "be investigated and resolved?"
    )

    print("\nIncident Query:")
    print(query)

    result = pipeline.run(
        query=query,
        top_k=5,
        candidate_k=10
    )

    print("\n========== INVESTIGATION RESULT ==========")

    print("\nStatus:", result["status"])
    print("Evidence source:", result["evidence_source"])

    print("\nHistorical documents:", result["retrieved_count"])
    print("External documents:", result["external_retrieved_count"])

    print("\nAbstention information:")
    print(result["abstention"])

    print("\n========== AI ANALYSIS ==========")
    print(result["analysis"])

    print("\n========== EXTERNAL SOURCES ==========")

    for index, source in enumerate(
        result["external_evidence"],
        start=1
    ):

        print(f"\nSource {index}")
        print("Title:", source["title"])
        print("URL:", source["url"])
        print("Trust level:", source["trust_level"])
        print("Relevance:", source.get("relevance_score"))

    print("\n========== TEST SUMMARY ==========")

    if result["status"] == "completed":
        print("Pipeline completed successfully.")
    else:
        print("Pipeline abstained.")
        print("Reason:", result["abstention"]["reason"])


if __name__ == "__main__":
    main()