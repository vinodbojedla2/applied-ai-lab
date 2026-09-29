# Evaluating AI search

Hit at k measures the share of questions for which at least one relevant document appears in the first k results. Mean reciprocal rank (MRR) averages the inverse rank of the first relevant result. These metrics test retrieval, not answer correctness. A complete RAG evaluation also reviews groundedness, citation correctness, refusal behavior, latency, and cost on representative questions.

Keep a held-out question set and inspect failures by category. For example, a relevant source missing from the retrieved set points to ingestion or retrieval, while an unsupported answer despite relevant context points to generation. Label quality matters: incomplete relevance judgments can mislead the retrieval scores.
