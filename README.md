# Applied AI Lab

A runnable Python portfolio project with retrieval augmented generation (RAG), retrieval evaluation, an optional FastAPI endpoint, and a small supervised ML example. The default path works without a paid API key.

## What it shows

| Area | Implementation |
| --- | --- |
| Document ingestion | Recursive Markdown/text loading, word-based chunks with overlap, traceable source IDs |
| Retrieval | TF-IDF unigram/bigram cosine similarity or BM25, deterministic ranking |
| Answers | Cited extractive baseline; optional LLM generation with source-only instructions and citation validation |
| Evaluation | Labeled queries, hit@k, MRR@k, precision@k, recall@k and per-query results at document level |
| ML | TF-IDF + logistic regression ticket classifier with stratified holdout evaluation |
| Service | Optional FastAPI `/ask` and `/health` endpoints; GitHub Actions tests |

The sample corpus and ticket data are intentionally tiny and synthetic. Scores on them are a smoke check, **not** evidence of production accuracy. The default answer is an excerpt from the best matching passage, not a synthesized answer. This baseline uses lexical retrieval; a next experiment is to compare embeddings and hybrid search against the same labeled question set.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e '.[dev]'
ai-lab ask "How do I evaluate retrieval?"
ai-lab evaluate
ai-lab evaluate --retriever bm25
ai-lab classify
pytest -q
```

To use your own documents, put `.md` or `.txt` files in a folder and pass `--docs path/to/folder`. The CLI loads files on each invocation; this favors a simple, reproducible demo over a persistent index.

```bash
ai-lab ask "What changed in the deployment?" --docs path/to/documents --top-k 3
ai-lab ask "How do I evaluate retrieval?" --retriever bm25
```

For generated answers, install the optional client and set your key in your shell (never commit it):

```bash
python -m pip install -e '.[llm]'
export OPENAI_API_KEY="your-key"  # PowerShell: $env:OPENAI_API_KEY="your-key"
ai-lab ask "What does RAG do?" --llm
```

The LLM receives retrieved excerpts, instructions to answer only from them, and source identifiers. The program checks that its answer includes only retrieved citation identifiers. This check does not prove that every claim is supported; production use requires claim-level evaluation and access control.

## API

```bash
python -m pip install -e '.[api]'
uvicorn ai_lab.api:app --reload
curl -X POST http://127.0.0.1:8000/ask -H 'Content-Type: application/json' -d '{"question":"How is MRR calculated?","top_k":3}'
```

Set `AI_LAB_DOCS` to point the API to another document directory. Open `/docs` for interactive API documentation. Do not expose this sample API publicly without authentication and document-level authorization.

## Project layout

```text
src/ai_lab/rag.py              ingestion, retrieval, answers
src/ai_lab/evaluation.py       labeled retrieval metrics
src/ai_lab/classification.py   supervised learning example
src/ai_lab/api.py              optional API
data/docs/                     synthetic RAG corpus
data/eval/questions.json       labeled questions
data/classification/           synthetic support tickets
tests/                        behavior checks
```

## Possible extensions

1. Add a sentence-transformer embedding retriever and compare hit@k/MRR with the lexical baseline.
2. Add a persistent vector index, incremental ingestion, and document versioning.
3. Build a larger manually reviewed evaluation set with unanswerable and adversarial queries.
4. Add per-user authorization before retrieval, tracing, latency monitoring, and a safe rollout workflow.

## License

MIT. See [LICENSE](LICENSE).
