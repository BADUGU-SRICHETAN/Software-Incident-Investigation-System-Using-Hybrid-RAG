from src.pipeline.root_cause_pipeline import RootCausePipeline


def main():

    pipeline = RootCausePipeline(
        data_directory="data/raw",
        persist_directory="vectorstore/rootcause_chroma",
        model_name="gpt-4o-mini"
    )

    query = """
    Our Kubernetes application is experiencing repeated pod
    restarts because liveness probes are failing after deployment.
    The application starts slowly, and Kubernetes terminates
    containers before they finish initializing. How can we
    investigate this incident?
    """

    result = pipeline.run(
        query=query,
        top_k=3,
        candidate_k=5
    )

    print("\n========== INVESTIGATION RESULT ==========")

    print("\nStatus:", result["status"])
    print("Evidence source:", result.get("evidence_source"))
    print("Historical documents:", result["retrieved_count"])
    print("External documents:", result["external_retrieved_count"])

    print("\n========== EXTERNAL SOURCES ==========")

    for source in result["external_evidence"]:

        print("\nTitle:", source["title"])
        print("Trust:", source["trust_level"])
        print("URL:", source["url"])

    print("\n========== ANALYSIS ==========")

    print(result["analysis"])


if __name__ == "__main__":
    main()