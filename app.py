import os
import json
import uuid
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

sessions = {}


class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    category: str
    session_id: str
    web_search_used: bool


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
    sid = req.session_id or str(uuid.uuid4())
    history = sessions.get(sid, [])

    context_question = req.question
    if history:
        recent = history[-3:]
        prev = " | ".join([f"Q: {h['q']} A: {h['a'][:100]}" for h in recent])
        context_question = f"Previous conversation: {prev}\n\nCurrent question: {req.question}"

    result = graph.invoke({
        "question": context_question,
        "retry_count": 0,
        "chat_history": history,
        "session_id": sid,
        "web_search_used": False
    })

    history.append({"q": req.question, "a": result.get("answer", "")})
    sessions[sid] = history

    return QueryResponse(
        answer=result.get("answer", ""),
        sources=result.get("sources", []),
        category=result.get("category", ""),
        session_id=sid,
        web_search_used=result.get("web_search_used", False)
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
