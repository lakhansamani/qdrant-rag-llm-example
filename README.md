# Local RAG Knowledge Base Demo

This project is a **fully local Retrieval-Augmented Generation (RAG) demo**: you load plain-text documents, ask questions in a web UI, and get answers grounded in those files with cited sources. Embeddings (FastEmbed), vector search (Qdrant), and text generation (Ollama) all run on your machine—no OpenAI or other cloud APIs in the pipeline.

> Stack: **Qdrant** · **FastEmbed** · **Ollama** · **Gradio** · Python 3.11+

---

## What it does

1. **Ingest** — Splits `.txt` files into chunks, embeds them, and stores vectors in Qdrant.
2. **Retrieve** — Finds the most relevant chunks for each question (cosine similarity).
3. **Generate** — Sends context + question to a local Ollama model and returns an answer plus source snippets.

Sample documents live in `data/knowledge_base/` (security policy, tech stack, onboarding). Replace or extend them with your own `.txt` files.

---

## Prerequisites

| Requirement | Notes |
|-------------|--------|
| **Python 3.11+** | [python.org](https://www.python.org/downloads/) |
| **Docker** | [docker.com](https://www.docker.com/) — runs the Qdrant container (which ships with a dashboard UI for browsing embeddings) |
| **Ollama** | [ollama.com/download](https://ollama.com/download) — local LLM server |
| **~2–4 GB disk** | For one Ollama model; first run also downloads ~25 MB embedding weights |

Ollama is **not** installed via `pip`. Install it separately, then pull at least one model (see below).

---

## Getting started

Run all commands from the **project root** (`rag-local-demo/`).

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd rag-local-demo
```

### 2. Create a virtual environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

**Optional dev tools** (already listed in `requirements.txt`):

```bash
ruff check src/ tests/
mypy src/ --ignore-missing-imports
```

### 3. Start Qdrant (Docker)

The demo defaults to a Qdrant server running on `http://localhost:6333`, which
also serves a built-in **dashboard UI** for browsing collections and visualising
embeddings.

```bash
docker compose up -d         # uses docker-compose.yml in the repo root
```

Then open the dashboard at **http://localhost:6333/dashboard** — go to the
`kb` collection → **Visualize** tab to see your vectors plotted in 2D.

> Prefer to skip Docker? You can still run with embedded storage:
> `python src/app.py --storage :memory:` or `--storage ./qdrant_data`.

### 4. Install Ollama and pull a model

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows: installer from https://ollama.com/download
```

Pull a model (required before asking questions):

```bash
ollama pull llama3.2      # default for this demo (~2 GB)
# or: ollama pull mistral | ollama pull phi3 | ollama pull llama3.1
```

Check what you have installed:

```bash
ollama list
```

Ollama usually starts automatically when you run `ollama` commands. If needed:

```bash
ollama serve
```

### 5. Run the web UI

```bash
python src/app.py
```

- Browser: **http://localhost:7860** (may open automatically).
- On first run, FastEmbed downloads `BAAI/bge-small-en-v1.5` once (~25 MB), then works offline.
- Startup checks that your chosen Ollama model exists; if the default `llama3.2` is missing, use a model you pulled:

```bash
python src/app.py --model llama3.1:latest
```

### 6. Try a question

Use the chat box or example prompts, e.g.:

- *How do I report a security incident?*
- *What is our data residency policy?*

---

## Architecture

```
User Question
      │
      ▼
┌─────────────────────────────┐
│  FastEmbed (bge-small-en)   │  ← 384-dim vectors, local ONNX
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│  Qdrant                     │  ← Top-k chunk search (cosine)
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│  Ollama (local LLM)         │  ← Context + question → answer
└─────────────┬───────────────┘
              ▼
         answer + sources
```

---

## Usage

### Web UI

```bash
python src/app.py                                       # llama3.2, Qdrant on http://localhost:6333
python src/app.py --model mistral                       # Different Ollama model
python src/app.py --storage :memory:                    # Embedded, no Docker, data lost on exit
python src/app.py --storage ./qdrant_data               # Embedded file storage
python src/app.py --storage http://localhost:6333       # Qdrant Docker server (default)
python src/app.py --port 8080                           # Custom UI port
python src/app.py --data /path/to/docs                  # Custom document folder
```

### Pre-ingest (large knowledge bases)

```bash
python scripts/ingest.py --storage ./qdrant_data
python src/app.py --storage ./qdrant_data
```

### Python API

```python
from pathlib import Path
from src.pipeline import RAGPipeline

pipeline = RAGPipeline(llm_model="llama3.2")
pipeline.ingest_directory(Path("data/knowledge_base"))

response = pipeline.ask("What is our data residency policy?")
response.pretty_print()
print(response.answer)
print(response.sources[0].source, response.sources[0].score)
```

**Permission-aware mode is enforced in the Python API too, not just the UI.** When
the pipeline is built with an `AuthzClient`, `ask()` **requires** a valid Authorizer
token and is **fail-closed** — a missing, empty, or forged token raises
`AuthorizationError` and the LLM is never called:

```python
from src.authz import AuthzClient

authz = AuthzClient("http://localhost:8080")
pipeline = RAGPipeline(llm_model="llama3.2", authz=authz)
pipeline.ingest_directory(Path("data/knowledge_base"))

token = authz.login("alice@example.com", "Demo@Pass123")     # engineering
pipeline.ask("What was our Q4 revenue?", user_token=token)   # finance never retrieved

pipeline.ask("What was our Q4 revenue?")                     # raises AuthorizationError
pipeline.ask("…", user_token="forged.jwt")                   # raises AuthorizationError
```

Because enforcement lives in `pipeline.ask()` (server-side), the Gradio "Use via API"
endpoints can't bypass it either — the token comes from server-side session state an
API caller can't set, and the pipeline re-validates it regardless.

### Add your own documents

1. Add `.txt` files under `data/knowledge_base/` (or pass `--data`).
2. Restart the app (documents are ingested on startup).

For PDF/DOCX, convert to text first (e.g. `pdfminer.six`).

---

## Fine-grained permissions (FGA)

The demo can run **permission-aware**: each user only retrieves — and therefore only
gets answers from — documents they're allowed to see. Authorization is enforced
*during* the Qdrant vector search (a payload filter built from the user's grants), so
restricted chunks are never scored and never reach the LLM.

> Powered by [Authorizer](https://github.com/authorizerdev/authorizer) (self-hosted
> auth + embedded [OpenFGA](https://openfga.dev) engine). Requires Authorizer ≥ v2.3.0
> — `docker-compose.yml` pins `quay.io/authorizer/authorizer:2.3.0-rc.2`.

### Quick start

```bash
docker compose up -d              # Qdrant + Authorizer
python scripts/fga_seed.py        # demo users, authorization model, grants
python scripts/fga_demo.py        # CLI walk-through (no Ollama needed)
python scripts/fga_demo.py --llm  # same, with generated answers via Ollama
python src/app.py --authorizer http://localhost:8080   # web UI with login
```

The seed creates three users (password `Demo@Pass123`) with this access matrix:

| document | `alice` (eng) | `bob` (new hire) | `carol` (finance) | why |
|----------|:----:|:----:|:----:|-----|
| `onboarding_guide.txt` | ✅ | ✅ | ✅ | public (`user:*` viewer) |
| `tech_stack.txt` | ✅ | ❌ | ❌ | `team:engineering#member` viewer |
| `financial_report.txt` | ❌ | ❌ | ✅ | `team:finance#member` viewer |
| `security_policy.txt` | ❌ | ❌ | ❌ | `team:security#member` viewer (nobody is) |

The headline example: **an engineer is blocked from the financial report.** Ask Alice
*"What was our Q4 revenue?"* and she gets nothing — the document is never retrieved, so
the LLM can't leak it. Ask Carol (finance) the same question and she gets the numbers.

`fga_demo.py` also demonstrates **live revocation**: Bob is granted engineering
membership (one tuple write) and immediately retrieves the tech-stack doc; the tuple
is deleted and his very next question is filtered again. No re-ingestion, no re-login.

### How it works

1. **Login** — the app gets the user's JWT from Authorizer (`src/authz.py`).
2. **Allow-list** — before searching, `list_permissions` returns every document the
   user `can_view`. The call uses the *user's own token*; the subject is pinned
   server-side, so a prompt-injected agent has nothing to escalate with.
3. **Pre-filter** — the allow-list becomes a Qdrant `MatchAny` must-condition
   (`src/vector_store.py`): forbidden vectors are never candidates and top-k stays
   meaningful.
4. **Re-verify** — after generation, the cited sources are batch-checked with
   `check_permissions` so a grant revoked mid-request can't leak (`src/pipeline.py`).

Every failure mode is **fail closed**: Authorizer unreachable, an expired token, or a
truncated permission list all mean *no documents*, never *all documents*.

Documents are FGA objects named after the chunk payload's `source` field
(`document:tech_stack.txt`), so the existing Qdrant payload is the join key — no
mapping table, and granting/revoking access never touches the vector index.

---

## Running tests

```bash
pytest tests/ -v
pytest tests/test_vector_store.py tests/test_pipeline.py -v   # fast, mocked
pytest tests/ --cov=src --cov-report=term-missing
pytest tests/test_embedder.py -v   # downloads embedding model on first run
```

---

## Project structure

```
rag-local-demo/
├── README.md
├── requirements.txt
├── docker-compose.yml    # Qdrant container (with dashboard UI on :6333)
├── .gitignore
├── src/
│   ├── embedder.py       # FastEmbed wrapper
│   ├── vector_store.py   # Qdrant client (incl. permission pre-filter)
│   ├── retriever.py      # Chunking, ingest, search
│   ├── llm_client.py     # Ollama HTTP client
│   ├── authz.py          # Authorizer/OpenFGA permission client (FGA)
│   ├── pipeline.py       # RAG orchestration
│   └── app.py            # Gradio UI (optional login via --authorizer)
├── data/knowledge_base/  # Sample .txt documents
├── scripts/ingest.py     # CLI pre-ingest
├── scripts/fga_seed.py   # FGA demo setup: users, model, grants
├── scripts/fga_demo.py   # FGA CLI walk-through incl. live revocation
└── tests/
```

---

## Configuration

| CLI flag | Default | Description |
|----------|---------|-------------|
| `--model` | `llama3.2` | Ollama model (must be installed: `ollama list`) |
| `--storage` | `http://localhost:6333` | Qdrant server URL, a file path (e.g. `./qdrant_data`), or `:memory:` |
| `--data` | `data/knowledge_base` | Folder of `.txt` files |
| `--port` | `7860` | Gradio port |
| `--share` | off | Public Gradio tunnel |
| `--authorizer` | off | Authorizer URL (e.g. `http://localhost:8080`) — enables login + fine-grained permissions |

**`RAGPipeline` kwargs:** `chunk_size=400`, `chunk_overlap=80`, `top_k=4`, `score_threshold=0.3`, `embedding_model=BAAI/bge-small-en-v1.5`.

---

## Troubleshooting

**Model not found / HTTP 404 from Ollama**

The server is running but the model name is wrong or not pulled:

```bash
ollama list
ollama pull llama3.2
python src/app.py --model <name-from-ollama-list>
```

**Cannot reach Ollama**

```bash
ollama serve
# or
ollama run llama3.2
```

**`ModuleNotFoundError: No module named 'src'`**

Run from the project root: `python src/app.py` (not from inside `src/`).

**FastEmbed download fails (first run only)**

Check network access to Hugging Face, or pre-warm:

```bash
python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"
```

**Reset persisted Qdrant data**

```bash
docker compose down -v       # if using the Docker server
rm -rf ./qdrant_data         # if using embedded file storage
```

**Cannot reach Qdrant at http://localhost:6333**

Make sure the container is running, or fall back to embedded storage:

```bash
docker compose up -d
# or run without Docker:
python src/app.py --storage :memory:
```

**Gradio API errors**

This repo targets Gradio 4.44+ (including Gradio 6). Use a recent install: `pip install -U gradio`.

---

## Privacy

- No cloud LLM or embedding APIs in the core path.
- Qdrant local/in-memory mode keeps vectors on your machine.
- Suitable for internal docs, air-gapped use after initial model downloads.

---

## Learn more

- [Qdrant documentation](https://qdrant.tech/documentation/)
- [FastEmbed](https://qdrant.github.io/fastembed/)
- [Ollama](https://ollama.com)
- [Gradio](https://gradio.app)

---

## License

MIT — use freely, including commercially.
