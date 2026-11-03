from src.pipeline.root_cause_pipeline import RootCausePipeline
import json

pipeline = RootCausePipeline(
    data_directory="data/raw",
    persist_directory="vectorstore/api_investigation",
    model_name="gpt-4o-mini"
)

query = "Which errors were associated with previous service outages?"

documents = pipeline.hybrid_retriever.retrieve(
    query=query,
    top_k=5,
    candidate_k=10
)

result = pipeline.abstention_checker.check(
    query=query,
    retrieved_documents=documents
)

print("\n========== ABSTENTION RESULT ==========")
print(json.dumps(result, indent=2))

print("\n========== RETRIEVED HISTORICAL EVIDENCE ==========")

for index, document in enumerate(documents, start=1):
    print(f"\nDOCUMENT {index}")
    print("Incident ID:", document.metadata.get("incident_id"))
    print("Vector score:", document.metadata.get("vector_score"))
    print("BM25 score:", document.metadata.get("bm25_score"))
    print("Content:", document.page_content)