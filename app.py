import os
import json
import shutil
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from src.graph import build_graph
from src.ingest import load_docs, chunk_docs, embed_docs

api = FastAPI(title="NyayaSetu", description="RAG-powered legal assistant for Indian law")
graph = build_graph()

DOCS_DIR = "data/docs"
DB_DIR = "data/chroma"
FEEDBACK_FILE = "data/feedback.json"


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    category: str


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: str
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    status: str


class DocumentInfo(BaseModel):
    filename: str
    size_bytes: int


@api.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    result = graph.invoke({"question": req.question, "retry_count": 0})
    return QueryResponse(
        answer=result.get("answer", ""),
        sources=result.get("sources", []),
        category=result.get("category", "")
    )


@api.post("/ingest")
def ingest(files: List[UploadFile] = File(...)):
    os.makedirs(DOCS_DIR, exist_ok=True)
    saved = []
    for f in files:
        if not f.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{f.filename} is not a PDF")
        path = os.path.join(DOCS_DIR, f.filename)
        with open(path, "wb") as buf:
            shutil.copyfileobj(f.file, buf)
        saved.append(f.filename)

    docs = load_docs(DOCS_DIR)
    chunks = chunk_docs(docs)
    embed_docs(chunks, DB_DIR)

    return {"status": "ingested", "files": saved, "chunks": len(chunks)}


@api.get("/documents", response_model=List[DocumentInfo])
def list_documents():
    if not os.path.exists(DOCS_DIR):
        return []
    result = []
    for f in os.listdir(DOCS_DIR):
        fp = os.path.join(DOCS_DIR, f)
        if os.path.isfile(fp):
            result.append(DocumentInfo(filename=f, size_bytes=os.path.getsize(fp)))
    return result


@api.post("/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest):
    if req.rating not in ("up", "down"):
        raise HTTPException(status_code=400, detail="rating must be 'up' or 'down'")

    entry = {
        "question": req.question,
        "answer": req.answer,
        "rating": req.rating,
        "comment": req.comment,
        "timestamp": datetime.now().isoformat()
    }

    existing = []
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r") as f:
            existing = json.load(f)

    existing.append(entry)
    os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(existing, f, indent=2)

    return FeedbackResponse(status="recorded")
