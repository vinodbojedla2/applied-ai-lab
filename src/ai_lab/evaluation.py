"""Retrieval metrics against a small labeled question set."""

import json
from pathlib import Path

from .rag import Retriever


def evaluate(retriever: Retriever, questions_path: str | Path, top_k: int = 3) -> dict:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    questions = json.loads(Path(questions_path).read_text(encoding="utf-8"))
    if not questions:
        raise ValueError("Evaluation set must be nonempty")
    rows = []
    for item in questions:
        relevant = set(item["relevant_sources"])
        if not relevant:
            raise ValueError("Each question requires relevant_sources")
        # Score distinct documents, even when a document yields multiple chunks.
        found = []
        for hit in retriever.search(item["question"], len(retriever.passages)):
            if hit.passage.source not in found:
                found.append(hit.passage.source)
            if len(found) == top_k:
                break
        rank = next((i for i, source in enumerate(found, 1) if source in relevant), None)
        rows.append({"question": item["question"], "retrieved": found,
                     "hit": int(rank is not None), "reciprocal_rank": 1 / rank if rank else 0.0,
                     "precision": len(relevant.intersection(found)) / top_k,
                     "recall": len(relevant.intersection(found)) / len(relevant)})
    return {"questions": len(rows), "hit_at_k": sum(r["hit"] for r in rows) / len(rows),
            "mrr_at_k": sum(r["reciprocal_rank"] for r in rows) / len(rows),
            "precision_at_k": sum(r["precision"] for r in rows) / len(rows),
            "recall_at_k": sum(r["recall"] for r in rows) / len(rows), "details": rows}
