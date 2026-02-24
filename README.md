# RAG FastAPI – Local Personal Q&A API

A fully functional **RAG (Retrieval-Augmented Generation)** API that answers personal questions using AI, running entirely on your local machine. It uses ChromaDB for a vector knowledge base, Ollama for embeddings and the LLM, and FastAPI for the REST API.

---

## What You'll Learn

- **Perform RAG** – Retrieval-Augmented Generation both manually and with code.
- **Build a personal knowledge base** – Using ChromaDB and vector embeddings.
- **Build a REST API** – FastAPI implementing a full RAG pipeline (retrieve → augment → generate).
- **Use local models** – `nomic-embed-text` for semantic search and `qwen2.5:0.5b` for answer generation.
- **Multi-user directory** – Extend the API with dynamic document ingestion and per-user filtering.

---

## Prerequisites

- **Python 3.9+**
- **Ollama** installed and running locally ([ollama.ai](https://ollama.ai))

Pull the required Ollama models:

```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:0.5b
```

---

## Setup

1. **Clone or navigate to the project**

   ```bash
   cd RAG_Fastapi
   ```

2. **Create a virtual environment (recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install fastapi uvicorn pydantic ollama chromadb
   ```

4. **(Optional) Build the initial knowledge base from `profile.txt`**

   ```bash
   python build_knowledge_base.py
   ```

   This loads `profile.txt`, chunks it by paragraph, and adds it to the `personal_profile` ChromaDB collection. Documents added this way do not have a `user_name` in metadata; use the `/documents` endpoint for multi-user profiles.

---

## Running the API

Start the server:

```bash
uvicorn main:app --reload
```

- API: **http://127.0.0.1:8000**
- Interactive docs: **http://127.0.0.1:8000/docs**

---

## API Endpoints

### `GET /ask`

Ask a question over the knowledge base. Optionally restrict to one user’s profile.

| Query parameter | Type   | Description                                |
|----------------|--------|--------------------------------------------|
| `question`     | string | Your question (required)                   |
| `user`         | string | Optional. Filter by `user_name` in metadata |

**Examples:**

```bash
# Search all profiles
curl "http://127.0.0.1:8000/ask?question=What%20are%20my%20hobbies?"

# Search only UDAY's profile
curl "http://127.0.0.1:8000/ask?question=What%20are%20my%20hobbies?&user=UDAY"
```

**Response:** `question`, `answer`, `context_used` (chunks used), `filtered_by_user` (or `null` if searching all).

---

### `POST /documents`

Add a user’s profile so it can be queried via `/ask` (with optional `user` filter).

**Body (JSON):**

```json
{
  "user_name": "UDAY",
  "content": "Name: UDAY.\n\nHobbies: Reading, hiking, coding.\n\nWork: Software engineer."
}
```

Content is split into chunks by double newlines (`\n\n`); each chunk is stored with `user_name` in metadata.

**Example:**

```bash
curl -X POST "http://127.0.0.1:8000/documents" \
  -H "Content-Type: application/json" \
  -d '{"user_name": "UDAY", "content": "Name: UDAY.\n\nHobbies: Reading, hiking."}'
```

**Response:** `message`, `user_name`, `chunks_added`.

---

## Project Layout

| File / folder        | Purpose |
|----------------------|--------|
| `main.py`            | FastAPI app: `/ask` (RAG) and `/documents` (ingest). |
| `build_knowledge_base.py` | One-time script to load `profile.txt` into ChromaDB. |
| `profile.txt`       | Sample profile used by the build script. |
| `chroma_db/`         | Persistent ChromaDB storage (created automatically). |

---

## How RAG Works Here

1. **Retrieve** – Query ChromaDB with the user’s question; get the most similar text chunks (optionally filtered by `user_name`).
2. **Augment** – Build a prompt that includes those chunks as “context” and the original question.
3. **Generate** – Send the augmented prompt to the local LLM (`qwen2.5:0.5b`) and return the answer plus `context_used` and `filtered_by_user`.

---

## Notes

- **User filter:** The `user` parameter on `/ask` only returns chunks that have matching `user_name` in ChromaDB metadata. Documents added via `build_knowledge_base.py` do not set `user_name`; only documents added via `POST /documents` can be filtered by user.
- **Ollama:** Must be running (e.g. `ollama serve`) so the API can call the embedding model and the LLM.

---

Nice work — you’ve built a local, multi-user–capable RAG API.
