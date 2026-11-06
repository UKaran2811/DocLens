"""Retrieval + grounded, citation-backed generation. Both steps are traced in LangSmith."""
import re
from functools import lru_cache

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable

from .config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, TOP_K
from .store import source_filter, vector_store

SYSTEM = """You are a troubleshooting assistant. Answer ONLY from the manual excerpts provided.
Each excerpt is labelled [n]. Cite the label after every claim, e.g. [1] or [2][3].
If the excerpts do not contain the answer, say you could not find it in the uploaded manuals.
Never invent steps, part numbers, torque values or specs. For fixes, give a short numbered list."""

PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM), ("human", "Question: {question}\n\nExcerpts:\n{context}")]
)

def make_llm(model: str, max_tokens: int = 1024) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        temperature=0,
        max_tokens=max_tokens,
    )

@lru_cache
def llm() -> ChatOpenAI:
    return make_llm(LLM_MODEL)

def to_text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict))


@traceable(name="retrieve", run_type="retriever")
def retrieve(question: str, k: int = TOP_K, source: str | None = None):
    return vector_store().similarity_search_with_score(question, k=k, filter=source_filter(source))


@traceable(name="answer_question")
def answer(question: str, k: int = TOP_K, source: str | None = None) -> dict:
    hits = retrieve(question, k=k, source=source)
    context = "\n\n".join(
        f"[{i}] ({d.metadata['source']}, p.{d.metadata['page']})\n{d.page_content}"
        for i, (d, _) in enumerate(hits, 1)
    )
    reply = (PROMPT | llm()).invoke({"question": question, "context": context})
    text = to_text(reply.content)

    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 1 <= int(n) <= len(hits)})
    sources = [
        {
            "ref": i,
            "source": d.metadata["source"],
            "page": d.metadata["page"],
            "score": round(float(score), 4),
            "cited": i in cited,
            "text": d.page_content,
        }
        for i, (d, score) in enumerate(hits, 1)
    ]
    return {"answer": text, "cited_refs": cited, "sources": sources}
