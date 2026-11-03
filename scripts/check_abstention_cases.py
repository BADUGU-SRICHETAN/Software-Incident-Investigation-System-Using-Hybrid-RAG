from src.analysis.abstention_checker import AbstentionChecker

checker = AbstentionChecker(
    vector_score_threshold=0.35,
    score_top_k=3
)

test_queries = [
    "What are the recommended troubleshooting steps for Kubernetes pods repeatedly restarting?",
    "What are common methods for diagnosing HTTP 502 errors in reverse proxies?",
    "How can Python applications be diagnosed when they experience memory leaks?"
]

for index, query in enumerate(test_queries, start=1):
    result = checker.check(query, [])

    print(f"\nTest {index}")
    print(f"Query: {query}")
    print(f"Accepted: {result['accepted']}")
    print(f"Reason: {result['reason']}")
    print(f"Category: {result.get('category')}")
