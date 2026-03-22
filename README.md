# 🧠 DocMind — AI-Powered Document Q&A Chatbot

> Ask questions in plain English. Get grounded, cited answers from your own documents.
> Powered by **Endee** vector database, **SentenceTransformers**, and **GPT / Claude**.

---

## Problem Statement

Knowledge workers spend enormous time searching through PDFs, reports, and documents to find specific information. Traditional keyword search misses semantically related content, and LLMs alone hallucinate facts not in their training data.

**DocMind solves this with RAG (Retrieval-Augmented Generation):**
- Ingest any PDF or text file once
- Ask natural language questions
- Get accurate, document-grounded answers with source citations

---

## Solution Overview

DocMind implements a complete RAG pipeline:

```
Documents → Chunking → Embedding → Endee (vector store)
                                        ↓
User Question → Embedding → Endee Query → Top-K Chunks → LLM → Answer
```

All retrieval is semantic (not keyword-based), so "What causes inflammation?" correctly matches chunks about "immune response" and "cytokines" even without the exact words.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    INGESTION PIPELINE                    │
│                                                         │
│  PDF/TXT  →  ingest.py  →  embedding.py  →  Endee DB   │
│  (files)     (chunks)      (384-d vectors)  (HNSW index)│
└─────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────┐
│                     QUERY PIPELINE                      │
│                                                         │
│  Question  →  embed  →  Endee query  →  LLM  →  Answer  │
│  (user)       (384-d)   (top-5 chunks)  (GPT/Claude)   │
└─────────────────────────────────────────────────────────┘
```

### Module Breakdown

| Module | Responsibility |
|--------|---------------|
| `src/config.py` | All settings via environment variables |
| `src/logger.py` | Structured logging (console + file) |
| `src/ingest.py` | PDF/text loading + overlap-aware chunking |
| `src/embedding.py` | SentenceTransformer embedding (local, free) |
| `src/vectorstore.py` | **Endee** create/upsert/query/delete operations |
| `src/query.py` | RAG: embed → retrieve → LLM → answer |
| `src/pipeline.py` | Orchestrates the full ingestion pipeline |
| `app.py` | Streamlit chat UI |
| `main.py` | CLI (ingest / query / chat / status) |

---

## How Endee Is Used

DocMind uses Endee as its vector database through the official Python SDK (`pip install endee`).

### Index creation
```python
from endee import Endee, Precision

client = Endee()                         # connects to localhost:8080
client.set_base_url("http://localhost:8080/api/v1")

client.create_index(
    name="docmind_index",
    dimension=384,                        # matches all-MiniLM-L6-v2
    space_type="cosine",
    precision=Precision.INT8,             # 4× smaller, minimal accuracy loss
)
```

### Storing document chunks
```python
index = client.get_index("docmind_index")

index.upsert([
    {
        "id": "chunk_abc123",
        "vector": [0.12, -0.05, ...],    # 384-d embedding
        "meta": {
            "text": "Endee is a high-performance vector database...",
            "source": "ai_overview.txt",
            "chunk_index": 3,
        },
        "filter": {"source": "ai_overview.txt"},
    }
])
```

### Semantic search
```python
results = index.query(
    vector=query_embedding,              # embedded user question
    top_k=5,
    ef=128,
)

for item in results:
    print(item["id"], item["similarity"], item["meta"]["text"])
```

### Filtered search (by document)
```python
results = index.query(
    vector=query_embedding,
    top_k=5,
    filter=[{"source": {"$eq": "report_2024.pdf"}}],
)
```

---

## Installation

### Prerequisites

- Python 3.10+
- Docker & Docker Compose (for Endee server)
- An OpenAI or Anthropic API key

### Step 1 — Clone the repository
```bash
git clone https://github.com/your-username/docmind.git
cd docmind
```

### Step 2 — Start Endee
```bash
docker compose up -d endee
```
Endee will be available at `http://localhost:8080`.

### Step 3 — Set up Python environment
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Step 4 — Configure environment variables
```bash
copy.env.example .env
# Edit .env and set your API key:
#   OPENAI_API_KEY=sk-...        (for OpenAI)
#   ANTHROPIC_API_KEY=sk-ant-... (for Anthropic)
#   LLM_PROVIDER=openai          (or "anthropic")
```

---

## Usage

### Option A — Streamlit Web UI
```bash
streamlit run app.py
```
Open `http://localhost:8501` → Upload documents in the sidebar → Ask questions.

### Option B — Command Line

**Ingest documents:**
```bash
# From the default data/documents/ directory
python main.py ingest

# From a specific file
python main.py ingest --source data/sample/ai_overview.txt

# From a directory
python main.py ingest --source my_pdfs/
```

**Ask a single question:**
```bash
python main.py query "What is Retrieval Augmented Generation?" --show-sources
```

**Interactive chat session:**
```bash
python main.py chat
```

**Check index status:**
```bash
python main.py status
```

### Option C — Full Docker Stack
```bash
docker compose up -d
```
- Endee: `http://localhost:8080`
- DocMind UI: `http://localhost:8501`

---

## Example Queries

After ingesting `data/sample/ai_overview.txt`:

| Question | Expected Answer |
|----------|----------------|
| "What is RAG?" | Explanation of Retrieval-Augmented Generation with steps |
| "When was BERT released?" | 2018, by Google |
| "What vector databases are mentioned?" | Pinecone, Weaviate, Qdrant, Milvus, Endee |
| "What are the ethics concerns in AI?" | Bias, privacy, transparency, alignment |
| "How many vectors can Endee handle?" | Up to 1 billion on a single node |

---

## Running Tests

```bash
pytest tests/ -v
```

Tests mock the SentenceTransformer model — no GPU or model download needed.

---

## Project Structure

```
docmind/
├── app.py                  # Streamlit web UI
├── main.py                 # CLI entry point
├── requirements.txt        # Python dependencies
├── Dockerfile              # DocMind container
├── docker-compose.yml      # Endee + DocMind stack
├── .env.example            # Environment variable template
│
├── src/
│   ├── __init__.py
│   ├── config.py           # All configuration
│   ├── logger.py           # Structured logging
│   ├── ingest.py           # Document loading + chunking
│   ├── embedding.py        # SentenceTransformer embeddings
│   ├── vectorstore.py      # Endee SDK integration
│   ├── query.py            # RAG pipeline (retrieve + generate)
│   └── pipeline.py         # Ingestion orchestrator
│
├── data/
│   ├── documents/          # Place your PDFs/TXT files here
│   └── sample/             # Sample dataset (ai_overview.txt)
│
├── tests/
│   ├── test_ingest.py
│   └── test_embedding.py
│
└── logs/
    └── docmind.log
```

---

## Configuration Reference

All settings are controlled via environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `ENDEE_BASE_URL` | `http://localhost:8080/api/v1` | Endee server URL |
| `ENDEE_INDEX_NAME` | `docmind_index` | Name of the vector index |
| `LLM_PROVIDER` | `openai` | `openai` or `anthropic` |
| `OPENAI_API_KEY` | — | Your OpenAI API key |
| `ANTHROPIC_API_KEY` | — | Your Anthropic API key |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `TOP_K` | `5` | Chunks retrieved per query |
| `EF_SEARCH` | `128` | HNSW search quality |

---

## Future Improvements

- **Hybrid search**: Combine dense embeddings with BM25 sparse vectors (Endee supports this natively with `sparse_model="endee_bm25"`)
- **Multi-index support**: Separate indexes per project or user
- **Re-ranking**: Apply a cross-encoder re-ranker on retrieved chunks before LLM call
- **Streaming responses**: Stream LLM output token-by-token in the Streamlit UI
- **Chat history**: Pass conversation history for multi-turn awareness
- **Metadata filters**: Filter by date range, document type, or custom tags
- **Authentication**: Per-user document namespacing and access control
- **Async pipeline**: Concurrent embedding and upsert for large document batches

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [Endee](https://endee.io) — High-performance open-source vector database
- [SentenceTransformers](https://sbert.net) — Local embedding models
- [Streamlit](https://streamlit.io) — Rapid UI development
- [OpenAI](https://openai.com) / [Anthropic](https://anthropic.com) — LLM providers
