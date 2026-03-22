"""
DocMind — Command-Line Interface
Usage examples are printed with:  python main.py --help
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.config import cfg
from src.logger import get_logger
from src.pipeline import run_ingestion
from src.query import answer_question
from src.vectorstore import delete_index, describe_index

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Sub-commands
# ─────────────────────────────────────────────────────────────────────────────

def cmd_ingest(args: argparse.Namespace) -> None:
    """Load, chunk, embed, and store documents."""
    print(f"\n📂 Ingesting documents from: {args.source or cfg.documents_dir}")
    result = run_ingestion(
        source=args.source,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print("\n✅ Ingestion complete")
    print(f"   Documents : {result['document_count']}")
    print(f"   Chunks    : {result['chunk_count']}")
    print(f"   Vectors   : {result['vector_count']}")


def cmd_query(args: argparse.Namespace) -> None:
    """Ask a single question and print the answer."""
    question = args.question
    print(f"\n❓ Question: {question}\n")

    result = answer_question(
        question=question,
        top_k=args.top_k,
        source_filter=args.source_filter,
    )

    print("━" * 60)
    print("🧠 Answer:\n")
    print(result.answer)
    print("━" * 60)

    if args.show_sources:
        print(f"\n📚 Source chunks ({len(result.chunks)}):\n")
        for i, chunk in enumerate(result.chunks, 1):
            print(f"  [{i}] {chunk.source} (similarity={chunk.similarity:.4f})")
            print(f"      {chunk.text[:200]}…\n")

    print(f"🪙 Tokens used: {result.tokens_used}")


def cmd_chat(args: argparse.Namespace) -> None:
    """Interactive Q&A session in the terminal."""
    print("\n🧠 DocMind Interactive Chat")
    print("   Type your question and press Enter.  Type 'exit' to quit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break
        if not question:
            continue

        result = answer_question(question=question, top_k=args.top_k)
        print(f"\nDocMind: {result.answer}")
        print(
            f"         [sources: "
            + ", ".join({c.source for c in result.chunks})
            + f" | {result.tokens_used} tokens]\n"
        )


def cmd_status(args: argparse.Namespace) -> None:
    """Print the current index status."""
    try:
        info = describe_index()
        print("\n📊 Endee Index Status")
        print(json.dumps(info, indent=2))
    except Exception as exc:
        print(f"\n❌ Cannot reach Endee: {exc}")
        print("   Is the server running?  docker compose up -d")
        sys.exit(1)


def cmd_delete(args: argparse.Namespace) -> None:
    """Delete the vector index."""
    confirm = input(
        f"Delete index '{cfg.endee_index_name}'? This cannot be undone. [y/N] "
    )
    if confirm.lower() == "y":
        delete_index()
        print("✅ Index deleted.")
    else:
        print("Aborted.")


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="docmind",
        description="DocMind — AI-powered document Q&A backed by Endee",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── ingest ────────────────────────────────────────────────────────────────
    p_ingest = sub.add_parser("ingest", help="Ingest documents into Endee")
    p_ingest.add_argument(
        "--source",
        type=str,
        default=None,
        help=f"File or directory to ingest (default: {cfg.documents_dir})",
    )
    p_ingest.add_argument("--chunk-size", type=int, default=cfg.chunk_size)
    p_ingest.add_argument("--chunk-overlap", type=int, default=cfg.chunk_overlap)

    # ── query ─────────────────────────────────────────────────────────────────
    p_query = sub.add_parser("query", help="Ask a single question")
    p_query.add_argument("question", type=str, help="The question to answer")
    p_query.add_argument("--top-k", type=int, default=cfg.top_k)
    p_query.add_argument("--source-filter", type=str, default=None,
                         help="Restrict retrieval to this document filename")
    p_query.add_argument("--show-sources", action="store_true",
                         help="Print retrieved source chunks")

    # ── chat ──────────────────────────────────────────────────────────────────
    p_chat = sub.add_parser("chat", help="Interactive Q&A session")
    p_chat.add_argument("--top-k", type=int, default=cfg.top_k)

    # ── status ────────────────────────────────────────────────────────────────
    sub.add_parser("status", help="Show Endee index status")

    # ── delete ────────────────────────────────────────────────────────────────
    sub.add_parser("delete", help="Delete the vector index")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "ingest": cmd_ingest,
        "query": cmd_query,
        "chat": cmd_chat,
        "status": cmd_status,
        "delete": cmd_delete,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
