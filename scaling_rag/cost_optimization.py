"""Simple cost estimator connected to the current RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass

try:
    from app.embedding import load_and_split_chunks
except ModuleNotFoundError:
    from embedding import load_and_split_chunks


@dataclass(frozen=True)
class Pricing:
    # Prices are USD per 1M tokens.
    embedding_input_per_1m: float = 0.02  # text-embedding-3-small
    llm_input_per_1m: float = 0.15  # gpt-4o-mini input
    llm_output_per_1m: float = 0.60  # gpt-4o-mini output


def estimate_tokens(text: str) -> int:
    """Fast token estimate: good enough for planning costs."""
    return max(1, int(len(text.split()) * 1.3))


def estimate_indexing_cost(chunks: list, pricing: Pricing) -> dict:
    """Estimate one-time embedding cost for all chunk texts."""
    chunk_tokens = sum(estimate_tokens(doc.page_content) for doc in chunks if doc.page_content)
    embedding_cost = (chunk_tokens / 1_000_000) * pricing.embedding_input_per_1m

    return {
        "chunk_count": len(chunks),
        "embedding_tokens": chunk_tokens,
        "embedding_cost_usd": embedding_cost,
    }


def estimate_query_cost(
    query: str,
    chunks: list,
    pricing: Pricing,
    top_k: int = 4,
    expected_output_tokens: int = 250,
) -> dict:
    """Estimate per-query generation cost for RAG answer stage."""
    query_tokens = estimate_tokens(query)

    # Approximate context size from average chunk size.
    avg_chunk_tokens = (
        int(sum(estimate_tokens(doc.page_content) for doc in chunks if doc.page_content) / max(len(chunks), 1))
        if chunks
        else 0
    )
    context_tokens = avg_chunk_tokens * top_k
    prompt_tokens = query_tokens + context_tokens

    input_cost = (prompt_tokens / 1_000_000) * pricing.llm_input_per_1m
    output_cost = (expected_output_tokens / 1_000_000) * pricing.llm_output_per_1m

    return {
        "query_tokens": query_tokens,
        "context_tokens": context_tokens,
        "prompt_tokens": prompt_tokens,
        "output_tokens": expected_output_tokens,
        "llm_input_cost_usd": input_cost,
        "llm_output_cost_usd": output_cost,
        "query_total_cost_usd": input_cost + output_cost,
    }


def pipeline_cost_report(
    data_dir: str = "./data",
    query: str = "What is attention mechanism?",
    top_k: int = 4,
    expected_output_tokens: int = 250,
) -> dict:
    """Run the existing pipeline loading/chunking and return a full cost report."""
    pricing = Pricing()
    chunks = load_and_split_chunks(data_dir)

    indexing = estimate_indexing_cost(chunks, pricing)
    query_cost = estimate_query_cost(
        query=query,
        chunks=chunks,
        pricing=pricing,
        top_k=top_k,
        expected_output_tokens=expected_output_tokens,
    )

    report = {
        "indexing": indexing,
        "query": query_cost,
        "total_first_query_cost_usd": indexing["embedding_cost_usd"] + query_cost["query_total_cost_usd"],
    }

    return report


def print_cost_report(report: dict) -> None:
    """Print a readable summary for quick checks."""
    print("\n=== RAG Cost Report ===")
    print(f"Chunks: {report['indexing']['chunk_count']}")
    print(f"Embedding tokens: {report['indexing']['embedding_tokens']}")
    print(f"Embedding cost (one-time): ${report['indexing']['embedding_cost_usd']:.6f}")
    print("-")
    print(f"Prompt tokens/query: {report['query']['prompt_tokens']}")
    print(f"Output tokens/query: {report['query']['output_tokens']}")
    print(f"LLM input cost/query: ${report['query']['llm_input_cost_usd']:.6f}")
    print(f"LLM output cost/query: ${report['query']['llm_output_cost_usd']:.6f}")
    print(f"Total query cost: ${report['query']['query_total_cost_usd']:.6f}")
    print("-")
    print(f"Total first-query cost (indexing + one query): ${report['total_first_query_cost_usd']:.6f}")


if __name__ == "__main__":
    cost_report = pipeline_cost_report()
    print_cost_report(cost_report)
