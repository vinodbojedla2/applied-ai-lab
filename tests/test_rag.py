import json

from ai_lab.evaluation import evaluate
from ai_lab.rag import BM25Retriever, Passage, Retriever, answer, chunk_text


def test_chunk_overlap():
    assert chunk_text("one two three four five", max_words=3, overlap=1) == [
        "one two three", "three four five"
    ]


def test_retrieval_and_citation():
    retriever = Retriever([
        Passage("rag.md", 1, "retrieval augmented generation finds source passages"),
        Passage("ops.md", 1, "monitor latency and errors in production"),
    ])
    result = answer("source passages retrieval", retriever)
    assert result["citations"][0]["source"] == "rag.md#chunk-1"
    assert "[rag.md#chunk-1]" in result["answer"]
    assert answer("quantum banana", retriever)["citations"] == []


def test_metrics(tmp_path):
    data = tmp_path / "questions.json"
    data.write_text(json.dumps([{"question": "refund invoice", "relevant_sources": ["billing.md"]}]))
    retriever = Retriever([Passage("billing.md", 1, "refund invoice payment"),
                           Passage("other.md", 1, "server error log")])
    result = evaluate(retriever, data, top_k=1)
    assert result["hit_at_k"] == 1.0
    assert result["mrr_at_k"] == 1.0
    assert result["precision_at_k"] == 1.0
    assert result["recall_at_k"] == 1.0


def test_evaluation_counts_distinct_documents(tmp_path):
    data = tmp_path / "questions.json"
    data.write_text(json.dumps([{"question": "refund invoice", "relevant_sources": ["billing.md"]}]))
    retriever = Retriever([
        Passage("billing.md", 1, "refund invoice payment"),
        Passage("billing.md", 2, "invoice refund charge"),
        Passage("other.md", 1, "invoice tracking"),
    ])
    result = evaluate(retriever, data, top_k=2)
    assert result["details"][0]["retrieved"] == ["billing.md", "other.md"]
    assert result["precision_at_k"] == 0.5
    assert result["recall_at_k"] == 1.0


def test_bm25_ranks_relevant_passage_and_handles_no_match():
    retriever = BM25Retriever([
        Passage("search.md", 1, "BM25 scores query terms by frequency and document length"),
        Passage("deploy.md", 1, "Deployment rollback uses canary metrics"),
    ])
    assert retriever.search("BM25 frequency")[0].passage.source == "search.md"
    assert retriever.search("unmatchedxyz") == []
