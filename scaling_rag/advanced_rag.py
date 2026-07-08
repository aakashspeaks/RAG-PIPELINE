from __future__ import annotations

from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor

try:
    from app.retriver import build_hybrid_retriever
except ModuleNotFoundError:
    from retriver import build_hybrid_retriever

load_dotenv()


@dataclass
class AdvancedRAGResult:
    query: str
    expanded_queries: list[str]
    answer: str
    source_count: int
    sources: list[dict]


def build_query_variants(query: str) -> list[str]:
    """Simple query expansion for better recall."""
    q = query.strip()
    if not q:
        return []

    variants = [
        q,
        f"Explain: {q}",
        f"Key points about: {q}",
    ]

    # Keep order, remove duplicates.
    unique: list[str] = []
    for item in variants:
        if item not in unique:
            unique.append(item)
    return unique


def _doc_key(doc) -> tuple[str, str, str]:
    source = str(doc.metadata.get("source_file", "unknown"))
    page = str(doc.metadata.get("page_label", doc.metadata.get("page", "?")))
    snippet = doc.page_content[:120]
    return source, page, snippet


def retrieve_with_expanded_queries(query: str, top_k: int = 6, per_query_k: int = 3) -> tuple[list, list[str]]:
    """Run hybrid retrieval for query variants and merge unique documents."""
    variants = build_query_variants(query)
    if not variants:
        return [], []

    retriever = build_hybrid_retriever(vector_k=per_query_k, bm25_k=per_query_k)

    merged_docs = []
    seen_keys = set()

    for variant in variants:
        docs = retriever.invoke(variant)
        for doc in docs:
            key = _doc_key(doc)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            merged_docs.append(doc)
            if len(merged_docs) >= top_k:
                return merged_docs, variants

    return merged_docs, variants


def retrieve_with_compression(
    query: str,
    top_k: int = 6,
    per_query_k: int = 3,
    compressor_model_name: str = "gpt-4o-mini",
) -> tuple[list, list[str]]:
    """Retrieve docs, then compress each doc to only query-relevant text."""
    base_docs, variants = retrieve_with_expanded_queries(
        query=query,
        top_k=top_k,
        per_query_k=per_query_k,
    )
    if not base_docs:
        return [], variants

    compressor_llm = ChatOpenAI(model=compressor_model_name, temperature=0)
    compressor = LLMChainExtractor.from_llm(compressor_llm)

    # Use the same hybrid retriever as base so compression happens on retrieved docs.
    base_retriever = build_hybrid_retriever(vector_k=per_query_k, bm25_k=per_query_k)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )

    compressed_docs = compression_retriever.invoke(query)

    # Keep only up to top_k compressed docs.
    return compressed_docs[:top_k], variants


def _format_context(docs: list) -> str:
    blocks: list[str] = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page_label", doc.metadata.get("page", "?"))
        blocks.append(f"[Source {i}] file={source}, page={page}\n{doc.page_content.strip()}")
    return "\n\n".join(blocks)


def generate_advanced_rag_answer(
    query: str,
    model_name: str = "gpt-4o-mini",
    top_k: int = 6,
    per_query_k: int = 3,
    use_llm_compression: bool = True,
    compressor_model_name: str = "gpt-4o-mini",
) -> AdvancedRAGResult:
    """Advanced but simple RAG: query expansion + hybrid retrieval + optional LLM compression + generation."""
    if use_llm_compression:
        docs, variants = retrieve_with_compression(
            query=query,
            top_k=top_k,
            per_query_k=per_query_k,
            compressor_model_name=compressor_model_name,
        )
    else:
        docs, variants = retrieve_with_expanded_queries(query, top_k=top_k, per_query_k=per_query_k)

    if not docs:
        return AdvancedRAGResult(
            query=query,
            expanded_queries=variants,
            answer="I could not find relevant context in the indexed documents.",
            source_count=0,
            sources=[],
        )

    context = _format_context(docs)
    llm = ChatOpenAI(model=model_name, temperature=0)

    prompt = (
        "You are a helpful assistant. Use only the provided context to answer the question. "
        "If context is insufficient, clearly say so. Keep the answer concise and factual.\n\n"
        f"Question:\n{query}\n\n"
        f"Context:\n{context}\n\n"
        "At the end, include bullet-point sources in format: file name + page."
    )

    response = llm.invoke(prompt)

    sources = [
        {
            "source_file": doc.metadata.get("source_file", "unknown"),
            "page": doc.metadata.get("page_label", doc.metadata.get("page", "?")),
        }
        for doc in docs
    ]

    return AdvancedRAGResult(
        query=query,
        expanded_queries=variants,
        answer=response.content,
        source_count=len(sources),
        sources=sources,
    )


if __name__ == "__main__":
    q = "What is attention mechanism in transformers?"
    result = generate_advanced_rag_answer(q, use_llm_compression=True)

    print("\n=== Advanced RAG Output ===")
    print(f"Query: {result.query}")
    print(f"Expanded queries: {len(result.expanded_queries)}")
    for i, eq in enumerate(result.expanded_queries, start=1):
        print(f"  {i}. {eq}")
    print(f"Retrieved sources: {result.source_count}")
    print("\nAnswer:")
    print(result.answer)
