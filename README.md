# RAG Pipeline — Retrieval-Augmented Generation on Local Documents

A retrieval-augmented generation (RAG) system: given a question, retrieves the most
relevant chunks from a document set using semantic search, then generates a grounded
answer using a locally-served Llama 3.1 8B — answering strictly from retrieved context,
refusing when the answer isn't present rather than guessing.

## Results

**Retrieval accuracy: 100% (8/8)** on a labeled test set spanning all 5 source documents,
including an out-of-scope question with no correct answer in the corpus.

**Grounding verified directly** — asked a question entirely unrelated to the document
set ("What is the capital of France?"). Retrieval still returned a (irrelevant) top
match, as it always does, but generation correctly identified the context didn't
actually answer the question and refused rather than answering from general knowledge:

> "I don't have enough information to answer that."

This is the core claim RAG systems make and the one most worth verifying directly —
confirmed here with a real adversarial test, not just a happy-path example.

## Architecture

```
Documents (5 Wikipedia articles)
        │
        ▼
Chunking (500 chars, 50-char overlap)
        │
        ▼
Embedding — all-MiniLM-L6-v2 (sentence-transformers)
        │
        ▼
ChromaDB (persistent vector store, 777 chunks)
        │
        ▼
Query → embedded with the SAME model → top-k retrieval (cosine similarity)
        │
        ▼
Grounded prompt (retrieved chunks + question) → Llama 3.1 8B via Ollama (temperature=0)
        │
        ▼
Answer + cited source document(s)
```

## How to run

Requires [Ollama](https://ollama.com/) installed with `llama3.1:8b` pulled.

```bash
git clone https://github.com/Raed-Najeeb/rag-pipeline.git
cd rag-pipeline
py -m venv venv
source venv/Scripts/activate   # Windows Git Bash
py -m pip install -r requirements.txt

py src/rag/fetch_documents.py      # download source documents
py src/rag/chunking.py             # chunk into data/results/chunks.jsonl
py src/rag/build_index.py          # embed + store in ChromaDB
py src/rag/generate.py             # ask a sample question
py src/rag/evaluate_retrieval.py   # run the labeled retrieval test set
```

## Limitations

- **Small, topically distinct document set** (5 articles). The 100% retrieval accuracy
  reflects questions with a clear best-match document — not tested against documents
  with heavy topical overlap or highly similar content, where retrieval precision
  would likely be harder to achieve.
- **Character-count chunking, not sentence-aware.** Chunks can split mid-word or
  mid-sentence; overlap mitigates but doesn't eliminate information loss at boundaries.
- **No re-ranking.** Retrieval returns the top-k by raw embedding similarity only — a
  cross-encoder re-ranking step (explicitly out of scope per `SCOPE.md`) would likely
  improve precision, especially for harder or ambiguous queries.
- **Retrieval always returns a result**, even for out-of-scope questions — there's no
  built-in "no good match" signal. Correct refusal currently depends entirely on the
  generation step's grounding instruction, not on retrieval itself detecting irrelevance.

## What I'd build next

- Harder retrieval test cases — questions designed to probe overlap between documents
  (e.g. transformers vs. neural networks, both plausible matches)
- Cross-encoder re-ranking to improve precision on ambiguous queries
- A distance-threshold cutoff so retrieval itself can signal "no good match" rather
  than relying solely on generation to catch irrelevant context
