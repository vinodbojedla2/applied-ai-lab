"""Optional HTTP interface: pip install -e '.[api]' then uvicorn ai_lab.api:app."""

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .rag import Retriever, answer, load_documents

app = FastAPI(title="Applied AI Lab", version="0.1.0")


class Question(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)
    use_llm: bool = False


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask")
def ask(payload: Question) -> dict:
    try:
        retriever = Retriever(load_documents(os.getenv("AI_LAB_DOCS", "data/docs")))
        return answer(payload.question, retriever, payload.top_k, payload.use_llm)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
