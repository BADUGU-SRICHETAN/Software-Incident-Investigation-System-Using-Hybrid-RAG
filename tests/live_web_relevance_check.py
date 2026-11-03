from src.retrieval.web_retriever import WebRetriever
from src.retrieval.source_validator import SourceValidator


def main():

    query = (
        "Kubernetes application pod restarts because "
        "liveness probes fail during slow startup"
    )

    print("\nInitializing Tavily Web Retriever...")

    web_retriever = WebRetriever(
        max_results=5,
        search_depth="advanced"
    )

    validator = SourceValidator(
        min_content_length=100,
        min_relevance_score=0.15
    )

    print("\nSearching the web...")

    external_documents = web_retriever.retrieve(query=query)

    print(f"\nRetrieved documents: {len(external_documents)}")

    print("\nApplying relevance and source validation...")

    validated_documents = validator.validate(
        external_documents,
        query=query
    )

    print(f"Validated documents: {len(validated_documents)}")

    print("\n========== VALIDATED SOURCES ==========")

    for index, document in enumerate(validated_documents, start=1):

        metadata = document.metadata

        print(f"\nSource {index}")
        print("Title:", metadata.get("title"))
        print("URL:", metadata.get("url"))
        print("Trust level:", metadata.get("trust_level"))
        print("Relevance score:", metadata.get("relevance_score"))
        print("Tavily score:", metadata.get("score"))
        print("Content preview:", document.page_content[:300])

    rejected_count = (
        len(external_documents) - len(validated_documents)
    )

    print("\n========== SUMMARY ==========")
    print("Retrieved:", len(external_documents))
    print("Accepted:", len(validated_documents))
    print("Rejected:", rejected_count)


if __name__ == "__main__":
    main()