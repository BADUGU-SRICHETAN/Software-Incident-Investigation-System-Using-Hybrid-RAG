import re
from typing import Dict, Any


class QueryScopeChecker:
    """
    Validates whether a query belongs to software incident investigation.

    Rejects:
    - General knowledge questions
    - Unrelated topics
    - Requests for secrets and confidential information
    - Unsupported future predictions
    """

    def __init__(self):

        self.sensitive_patterns = [
            r"\bpasswords?\b",
            r"\bcredentials?\b",
            r"\bsecret keys?\b",
            r"\bprivate keys?\b",
            r"\bapi keys?\b",
            r"\bauthentication tokens?\b",
            r"\bexact private ip\b",
            r"\bprivate ip address\b",
        ]

        self.out_of_scope_patterns = {

            "financial_information": [
                r"\bexact financial loss\b",
                r"\bfinancial loss\b",
                r"\bannual expenditure\b",
                r"\bhosting cost\b",
                r"\bmonthly cloud hosting\b",
                r"\bsalar(?:y|ies)\b",
                r"\bpurchase price\b",
                r"\bbudget\b",
                r"\brestaurants?\b",
            ],

            "employee_information": [
                r"\bwho is responsible for\b",
                r"\bwho was responsible for\b",
                r"\bwhich employee\b",
                r"\bwhich engineer\b",
                r"\bwho approved\b",
                r"\bwho fixed\b",
                r"\bwho resolved\b",
            ],

            "future_prediction": [
                r"\btomorrow\b",
                r"\bpredict(?:ion)?\b",
                r"\bfuture\b",
                r"\bnext production outage\b",
            ],

            "general_knowledge": [
                r"\bcricket\b",
                r"\bphotosynthesis\b",
                r"\bbirthday greeting\b",
                r"\bweather forecast\b",
                r"\bsolve this mathematical equation\b",
                r"\bmultiplied by\b",
                r"\bcapital of\b",
                r"\bhistory of\b",
            ],
        }

        self.in_scope_patterns = [

            # Programming languages
            r"\bpython\b",
            r"\bjava\b",
            r"\bjavascript\b",

            # Software systems
            r"\bapplications?\b",
            r"\bsoftware\b",
            r"\bservices?\b",
            r"\bproduction\b",
            r"\bdatabase\b",
            r"\bapi\b",
            r"\bserver\b",

            # Infrastructure
            r"\bkubernetes\b",
            r"\bdocker\b",
            r"\bkafka\b",
            r"\bpostgresql\b",

            # Incident investigation
            r"\bdeployment\b",
            r"\bincident\b",
            r"\boutage\b",
            r"\bfailure\b",
            r"\berrors?\b",
            r"\bcrashes?\b",
            r"\bdebugging\b",
            r"\btroubleshooting\b",
            r"\blogs?\b",

            # Technical problems
            r"\bmemory\s+leaks?\b",
            r"\bmemory\b",
            r"\bconnection\b",
            r"\btimeout\b",
            r"\bperformance\b",
            r"\bconfiguration\b",
            r"\bnetwork\b",
            r"\bsecurity vulnerability\b",
            r"\bsql injection\b",

            # HTTP errors
            r"\bhttp 500\b",
            r"\bhttp 502\b",
            r"\bhttp 503\b",
            r"\bhttp 504\b",
        ]

    def check(self, query: str) -> Dict[str, Any]:

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        normalized_query = re.sub(
            r"\s+", " ", query.lower().strip()
        )

        # 1. Confidential information check
        for pattern in self.sensitive_patterns:

            if re.search(pattern, normalized_query):

                return {
                    "in_scope": False,
                    "category": "sensitive_information",
                    "matched_pattern": pattern,
                    "reason": (
                        "This request involves confidential information "
                        "that should not be retrieved or disclosed."
                    )
                }

        # 2. Explicitly unsupported categories
        for category, patterns in self.out_of_scope_patterns.items():

            for pattern in patterns:

                if re.search(pattern, normalized_query):

                    return {
                        "in_scope": False,
                        "category": category,
                        "matched_pattern": pattern,
                        "reason": (
                            "This question is outside the scope of "
                            "software incident investigation."
                        )
                    }

        # 3. Recognized technical topics
        for pattern in self.in_scope_patterns:

            if re.search(pattern, normalized_query):

                return {
                    "in_scope": True,
                    "category": "historical_incident_investigation",
                    "matched_pattern": pattern,
                    "reason": (
                        "The query relates to software systems or "
                        "incident investigation and can proceed "
                        "to evidence retrieval."
                    )
                }

        # 4. Reject unknown topics
        return {
            "in_scope": False,
            "category": "unrelated_topic",
            "matched_pattern": None,
            "reason": (
                "The query does not contain a recognized "
                "software incident or troubleshooting topic."
            )
        }