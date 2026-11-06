"""PDF -> chunks -> Qdrant. Every chunk keeps source + page so answers can cite them.

CLI:  python -m app.ingest manuals/*.pdf
"""
import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import CHUNK_OVERLAP, CHUNK_SIZE, COLLECTION
from .store import client, source_filter, vector_store


def ingest_pdf(path: str, display_name: str | None = None) -> int:
    name = display_name or Path(path).name
    pages = PyPDFLoader(path).load()
    for p in pages:
        p.metadata = {"source": name, "page": int(p.metadata.get("page", 0)) + 1}  # 1-based

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(pages)
    chunks = [c for c in chunks if c.page_content.strip()]
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = f"{name}:p{c.metadata['page']}:{i}"

    store = vector_store()  # makes sure the collection exists
    # Re-uploading the same manual replaces it instead of duplicating chunks.
    client().delete(COLLECTION, points_selector=source_filter(name))
    store.add_documents(chunks)
    return len(chunks)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python -m app.ingest <file.pdf> [...]")
    for f in sys.argv[1:]:
        print(f"{f}: {ingest_pdf(f)} chunks")
