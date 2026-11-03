import sys
from pathlib import Path

# Add project root to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.root_cause_pipeline import RootCausePipeline


QUERIES = [
    "What are the recommended troubleshooting steps for Kubernetes pods repeatedly restarting?",
    "What are common methods for diagnosing HTTP 502 errors in reverse proxies?",
    "How can Python applications be diagnosed when they experience memory leaks?"
]


def main():

    pipeline = RootCausePipeline(
        data_directory="data/raw",
        persist_directory="vectorstore/api_investigation",
        model_name="gpt-4o-mini"
    )

    for query in QUERIES:

        print("\n" + "=" * 90)
        print("QUERY:", query)
        print("=" * 90)

        documents = pipeline.hybrid_retriever.retrieve(
            query=query,
            top_k=5,
            candidate_k=10
        )

        abstention = pipeline.abstention_checker.check(
            query=query,
            retrieved_documents=documents
        )

        print("\nABSTENTION RESULT:")
        print(abstention)

        print("\nRETRIEVED DOCUMENTS:")

        for rank, document in enumerate(documents, start=1):

            metadata = document.metadata

            print("\n" + "-" * 80)
            print(f"RANK: {rank}")
            print("INCIDENT ID:", metadata.get("incident_id"))
            print("BM25 SCORE:", metadata.get("bm25_score"))
            print("VECTOR SCORE:", metadata.get("vector_score"))
            print("RRF SCORE:", metadata.get("rrf_score"))
            print("RETRIEVAL RANKS:", metadata.get("retrieval_ranks"))

            print("\nCONTENT:")
            print(document.page_content)


if __name__ == "__main__":
    main()