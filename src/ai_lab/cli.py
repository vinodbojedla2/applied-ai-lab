"""Command line entry point."""

import argparse
import json

from .classification import train_and_evaluate
from .evaluation import evaluate
from .rag import BM25Retriever, HybridRetriever, Retriever, answer, load_documents
from .storage import DocumentStore


RETRIEVERS = {"tfidf": Retriever, "bm25": BM25Retriever, "hybrid": HybridRetriever}


def main() -> None:
    parser = argparse.ArgumentParser(description="Applied AI lab")
    commands = parser.add_subparsers(dest="command", required=True)
    ask = commands.add_parser("ask", help="Ask a question over local documents")
    ask.add_argument("question")
    ask.add_argument("--docs", default="data/docs")
    ask.add_argument("--top-k", type=int, default=3)
    ask.add_argument("--retriever", choices=RETRIEVERS, default="tfidf")
    ask.add_argument("--index", help="Use passages from a previously ingested SQLite index")
    ask.add_argument("--llm", action="store_true", help="Use OpenAI for grounded generation")
    assessment = commands.add_parser("evaluate", help="Evaluate retrieval")
    assessment.add_argument("--docs", default="data/docs")
    assessment.add_argument("--questions", default="data/eval/questions.json")
    assessment.add_argument("--top-k", type=int, default=3)
    assessment.add_argument("--retriever", choices=RETRIEVERS, default="tfidf")
    assessment.add_argument("--index", help="Use passages from a previously ingested SQLite index")
    ingestion = commands.add_parser("ingest", help="Sync documents into a persistent SQLite index")
    ingestion.add_argument("--docs", default="data/docs")
    ingestion.add_argument("--index", default="data/index.sqlite")
    classifier = commands.add_parser("classify", help="Train and evaluate text classifier")
    classifier.add_argument("--data", default="data/classification/tickets.csv")
    args = parser.parse_args()
    if args.command == "classify":
        result = train_and_evaluate(args.data)
    elif args.command == "ingest":
        result = DocumentStore(args.index).sync(args.docs)
    else:
        passages = DocumentStore(args.index).passages() if args.index else load_documents(args.docs)
        retriever = RETRIEVERS[args.retriever](passages)
        result = answer(args.question, retriever, args.top_k, args.llm) if args.command == "ask" else evaluate(retriever, args.questions, args.top_k)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
