import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .ingest import ingest_pdf
from .rag import answer

app = FastAPI(title="DocLens", description="RAG troubleshooter over uploaded manuals")


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    source: str | None = Field(None, description="Limit to one uploaded manual (file name)")
    k: int = Field(5, ge=1, le=20)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
def upload(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF manuals are supported")
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / Path(file.filename).name
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        chunks = ingest_pdf(str(dest), display_name=dest.name)
    return {"source": dest.name, "chunks": chunks}


@app.post("/ask")
def ask(req: AskRequest):
    return answer(req.question, k=req.k, source=req.source)
