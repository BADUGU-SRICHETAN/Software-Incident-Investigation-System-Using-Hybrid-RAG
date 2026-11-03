# RootCause AI

### Hybrid RAG-Based Software Incident Investigation System

RootCause AI is a Hybrid Retrieval-Augmented Generation (RAG) prototype
that supports software incident investigation using historical incident
records. It combines BM25 keyword retrieval and vector-based semantic
retrieval, merges their rankings with Reciprocal Rank Fusion (RRF),
evaluates the available evidence, and uses GPT-4o-mini to generate an
investigation summary grounded in retrieved records.

The system is designed to help engineers find similar historical
incidents more efficiently. It provides investigation support, not a
guaranteed diagnosis of the actual cause of a new incident.

------------------------------------------------------------------------

## Key Features

-   **Hybrid Retrieval:** Combines BM25 keyword retrieval with
    vector-based semantic retrieval.
-   **Vector Search:** Uses OpenAI `text-embedding-3-small` embeddings
    and ChromaDB.
-   **Reciprocal Rank Fusion (RRF):** Merges the ranked results from
    BM25 and vector retrieval.
-   **Query Scope Validation:** Checks whether a query belongs to
    software incident investigation.
-   **Evidence Evaluation and Abstention:** Avoids generating an
    investigation when the available evidence is insufficient.
-   **LLM-Based Investigation:** Uses GPT-4o-mini to prepare an
    investigation summary from retrieved evidence.
-   **Incident References:** Presents relevant historical incident
    records as supporting context.
-   **Retrieval Evaluation:** Compares BM25, vector, and hybrid
    retrieval.
-   **REST API:** Uses FastAPI for backend requests.
-   **Interactive Dashboard:** Uses Streamlit as the user interface.

------------------------------------------------------------------------

## System Architecture

``` text
                         Historical Incident Dataset
                                      |
                                      v
                                Data Ingestion
                                      |
                         +------------+------------+
                         |                         |
                         v                         v
                    BM25 Index              Embedding Generation
                                                   |
                                                   v
                                                ChromaDB

User
 |
 v
Streamlit Dashboard
 |
 v
FastAPI Backend
 |
 v
Query Scope Validation
 |
 v
Hybrid Retrieval
 |                         |
 v                         v
BM25 Search           Vector Search
 |                         |
 +------------+------------+
              |
              v
     Reciprocal Rank Fusion
              |
              v
      Evidence Evaluation
          /         \
         v           v
   Sufficient     Insufficient
    Evidence        Evidence
         |             |
         v             v
   GPT-4o-mini     Abstention
         |
         v
 Investigation Summary
 + Supporting Historical Evidence
         |
         v
 FastAPI Response
         |
         v
 Streamlit Dashboard
```

------------------------------------------------------------------------

## Technology Stack

  -----------------------------------------------------------------------
  Technology                          Purpose
  ----------------------------------- -----------------------------------
  Python                              Main implementation language

  BM25 (`rank-bm25`)                  Keyword-based document retrieval

  OpenAI `text-embedding-3-small`     Generates text embeddings for
                                      semantic retrieval

  ChromaDB                            Stores and searches vector
                                      embeddings

  Reciprocal Rank Fusion              Combines BM25 and vector rankings

  GPT-4o-mini                         Generates evidence-grounded
                                      investigation summaries

  LangChain ecosystem                 Supports document and retrieval
                                      workflows

  FastAPI                             Backend REST API

  Streamlit                           Interactive frontend

  scikit-learn                        Evaluation and metric calculations

  pytest                              Automated testing
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## Dataset

The current prototype uses a **synthetic software incident dataset**.

-   **Incident records:** 50 (INC001--INC050).
-   **Domain:** Software incident investigation.
-   **Record information:** Incident descriptions, technical symptoms,
    affected services, investigation details, and resolutions.
-   **Evaluation set:** 90 curated retrieval test cases.

The dataset is synthetic and should not be interpreted as real
production incident data. Evaluation results describe performance on the
current curated test set and may not generalize to production
environments.

------------------------------------------------------------------------

## Retrieval Approach

### 1. BM25 Keyword Retrieval

BM25 ranks incident records according to keyword relevance. It is useful
for exact technical terms, error messages, log phrases, and service
names.

### 2. Vector-Based Semantic Retrieval

Incident text is represented as embeddings using OpenAI's
`text-embedding-3-small` model. ChromaDB stores and searches these
vectors. Semantic retrieval can find related incidents even when the
wording differs.

### 3. Reciprocal Rank Fusion

RRF combines the ranked lists from BM25 and vector retrieval into a
unified ranking. This allows the system to use both exact keyword
matches and semantic similarity.

------------------------------------------------------------------------

## Evidence Evaluation

The system includes two safeguards:

-   **Query Scope Validation:** Checks whether the submitted question is
    within the software incident investigation scope.
-   **Evidence-Based Abstention:** Evaluates whether the retrieved
    information is sufficient before generating an investigation. The
    current prototype uses a provisional threshold of **0.35**,
    calibrated on its evaluation data. This is not a universal threshold
    and requires further validation on larger and real-world datasets.

------------------------------------------------------------------------

## Investigation Output

For an in-scope query with sufficient evidence, the system can generate
an investigation summary containing:

-   A summary of the reported incident.
-   Relevant historical incident records.
-   Similarities between the new incident and retrieved records.
-   Possible contributing causes supported by the retrieved evidence.
-   Previously recorded resolutions.
-   Supporting historical evidence.

A similar historical incident is evidence for investigation, not proof
that the same cause is responsible for a new incident.

------------------------------------------------------------------------

## Evaluation

The project evaluates three retrieval approaches using 90 curated test
cases:

1.  BM25 retrieval.
2.  Vector retrieval.
3.  Hybrid retrieval.

The reported evaluation findings are:

-   Hybrid retrieval achieved the highest **Precision@2**, **Recall@2**,
    and **nDCG** among the three approaches tested.
-   Vector retrieval achieved the highest **MRR** and **Hit Rate**.

These findings indicate that hybrid retrieval provided a strong balance
on the current test set. They measure retrieval performance; they do not
independently establish the factual correctness of every LLM-generated
investigation report.

------------------------------------------------------------------------

## Project Structure

The repository is organized around the following top-level folders and
files:

``` text
RootCause-AI/
├── app/
├── data/
├── reports/
├── scripts/
├── src/
├── tests/
├── .env.example
├── .gitignore
├── app.py
├── inspect_rc005.py
├── README.md
└── requirements.txt
```

-   `app.py` --- Streamlit application entry point.
-   `src/` --- Main application implementation, including API,
    retrieval, pipeline, and analysis modules.
-   `data/` --- Incident dataset and evaluation data.
-   `reports/` --- Evaluation reports.
-   `scripts/` --- Utility and evaluation scripts.
-   `tests/` --- Automated tests.
-   `.env.example` --- Example environment variable configuration.

------------------------------------------------------------------------

## Setup and Installation

### Prerequisites

-   Python installed.
-   An OpenAI API key.
-   Git (if cloning the repository).

### 1. Clone the repository

``` bash
git clone https://github.com/BADUGU-SRICHETAN/RootCause-AI-Using-Hybrid-RAG.git
cd RootCause-AI-Using-Hybrid-RAG
```

### 2. Create and activate a virtual environment

**Windows PowerShell:**

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use the Python executable inside
`.venv` directly or follow your organization's PowerShell
execution-policy guidance.

### 3. Install dependencies

``` powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy the example file:

``` powershell
Copy-Item .env.example .env
```

Open `.env` locally and add your own API key:

``` env
OPENAI_API_KEY=your_openai_api_key_here
```

Keep `.env` private. **Never commit `.env` or paste API keys into
GitHub, screenshots, or public messages.** The repository should contain
`.env.example` with placeholders only.

------------------------------------------------------------------------

## Run the Application

Open two PowerShell terminals from the project root. Activate the
virtual environment in each terminal.

### Terminal 1 --- Start FastAPI

``` powershell
uvicorn src.api.main:app --host 127.0.0.1 --port 8001 --reload
```

The API health endpoint is:

``` text
http://127.0.0.1:8001/health
```

Interactive API documentation is available at:

``` text
http://127.0.0.1:8001/docs
```

### Terminal 2 --- Start Streamlit

``` powershell
streamlit run app.py
```

Open the local Streamlit URL shown in the terminal. Enter a software
incident description to submit an investigation request.

------------------------------------------------------------------------

## Run Tests

From the project root, run:

``` powershell
python -m pytest tests/ -v
```

A previous project test run completed with **113 tests passed**. Run the
tests in your current environment to verify the latest repository state.

------------------------------------------------------------------------

## Example Investigation Query

``` text
The payment service started returning HTTP 500 errors after the latest
deployment. Application logs show repeated PostgreSQL connection timeouts
and database connection pool exhaustion. Investigate similar historical
incidents and identify the most likely contributing causes.
```

The system searches historical records, evaluates the retrieved
evidence, and---when sufficient evidence is available---uses GPT-4o-mini
to produce an investigation summary with supporting incident references.

------------------------------------------------------------------------

## Future Enhancements

-   **Knowledge Base Expansion:** Add more diverse and verified incident
    records.
-   **Retrieval Improvement:** Experiment with retrieval parameters and
    ranking strategies.
-   **Better Evaluation:** Assess answer faithfulness, factual
    correctness, and evidence relevance in addition to retrieval
    metrics.
-   **Scalability:** Support larger incident knowledge bases.
-   **Log Integration:** Connect application logs and monitoring
    systems.
-   **Performance Optimization:** Reduce retrieval latency and API
    costs.
-   **Cloud Deployment:** Make the application accessible through cloud
    hosting.

------------------------------------------------------------------------

## Limitations

-   The current knowledge base contains 50 synthetic incident records.
-   Results depend on the coverage and quality of the available records.
-   The abstention threshold is provisional and calibrated on the
    current evaluation data.
-   Retrieved historical causes are possible explanations, not confirmed
    diagnoses of new incidents.
-   Retrieval metrics do not by themselves prove the factual correctness
    of generated answers.
-   Production use would require validation with representative real
    incident data, security review, and operational testing.

------------------------------------------------------------------------

## Author

**Srichetan Badugu**

-   GitHub: [BADUGU-SRICHETAN](https://github.com/BADUGU-SRICHETAN)
-   LinkedIn: [b-srichetan](https://www.linkedin.com/in/b-srichetan)

------------------------------------------------------------------------

## License

Add a license file if you intend to publish this project under an
open-source license.
