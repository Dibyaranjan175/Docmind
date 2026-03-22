from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.config import cfg
from src.embedding import embed_query
from src.logger import get_logger
from src.vectorstore import query_index

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    text: str
    source: str
    similarity: float
    chunk_index: int


@dataclass
class QueryResult:
    question: str
    answer: str
    chunks: list[RetrievedChunk] = field(default_factory=list)
    tokens_used: int = 0


SYSTEM_PROMPT = """You are DocMind, a precise and helpful document assistant.
Answer questions using ONLY the context passages provided below.
If the answer is not contained in the context, say "I couldn't find that in the provided documents."
Always cite the source document(s) at the end of your answer in the format: [Source: filename].
Be concise, factual, and clear."""

CONTEXT_TEMPLATE = """CONTEXT PASSAGES:
{passages}

QUESTION: {question}

ANSWER:"""


def _build_context(chunks: list[RetrievedChunk]) -> str:
    passages = []
    for i, chunk in enumerate(chunks, 1):
        passages.append(
            f"[{i}] (Source: {chunk.source}, similarity={chunk.similarity:.3f})\n"
            f"{chunk.text}"
        )
    return "\n\n".join(passages)


def _call_openai(context: str, question: str) -> tuple[str, int]:
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install openai")

    if not cfg.openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")

    client = OpenAI(api_key=cfg.openai_api_key)
    user_message = CONTEXT_TEMPLATE.format(passages=context, question=question)
    response = client.chat.completions.create(
        model=cfg.openai_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        max_tokens=cfg.llm_max_tokens,
        temperature=cfg.llm_temperature,
    )
    answer = response.choices[0].message.content.strip()
    tokens = response.usage.total_tokens if response.usage else 0
    return answer, tokens


def _call_anthropic(context: str, question: str) -> tuple[str, int]:
    try:
        import anthropic
    except ImportError:
        raise ImportError("pip install anthropic")

    if not cfg.anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is not set.")

    client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)
    user_message = CONTEXT_TEMPLATE.format(passages=context, question=question)
    response = client.messages.create(
        model=cfg.anthropic_model,
        max_tokens=cfg.llm_max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    answer = response.content[0].text.strip()
    tokens = response.usage.input_tokens + response.usage.output_tokens
    return answer, tokens


def _call_groq(context: str, question: str) -> tuple[str, int]:
    try:
        from groq import Groq
    except ImportError:
        raise ImportError("pip install groq")

    import os
    groq_api_key = os.getenv("GROQ_API_KEY", "")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set.")

    client = Groq(api_key=groq_api_key)
    user_message = CONTEXT_TEMPLATE.format(passages=context, question=question)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        max_tokens=cfg.llm_max_tokens,
        temperature=cfg.llm_temperature,
    )
    answer = response.choices[0].message.content.strip()
    tokens = response.usage.total_tokens if response.usage else 0
    return answer, tokens


def _generate_answer(context: str, question: str) -> tuple[str, int]:
    provider = cfg.llm_provider.lower()
    if provider == "openai":
        return _call_openai(context, question)
    elif provider == "anthropic":
        return _call_anthropic(context, question)
    elif provider == "groq":
        return _call_groq(context, question)
    else:
        raise ValueError(
            f"Unknown LLM provider '{cfg.llm_provider}'. "
            "Choose 'openai', 'anthropic' or 'groq'."
        )


def answer_question(
    question: str,
    top_k: int = cfg.top_k,
    source_filter: str | None = None,
) -> QueryResult:
    if not question.strip():
        return QueryResult(question=question, answer="Please enter a question.")

    logger.info(f"Processing question: '{question}'")

    query_vector = embed_query(question)

    raw_results = query_index(
        query_vector=query_vector,
        top_k=top_k,
        source_filter=source_filter,
    )

    if not raw_results:
        return QueryResult(
            question=question,
            answer="No relevant content found. Please ingest documents first.",
        )

    chunks = [
        RetrievedChunk(
            text=r["text"],
            source=r["source"],
            similarity=r["similarity"],
            chunk_index=r["chunk_index"],
        )
        for r in raw_results
    ]

    context = _build_context(chunks)
    answer, tokens = _generate_answer(context=context, question=question)

    logger.info(f"Answer generated — {tokens} tokens used")

    return QueryResult(
        question=question,
        answer=answer,
        chunks=chunks,
        tokens_used=tokens,
    )