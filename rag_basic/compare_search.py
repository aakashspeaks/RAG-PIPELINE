from __future__ import annotations
import os
from dotenv import load_dotenv

try:
	from app.rag_supabase import SupabaseVectorRetriever, search_supabase
except ModuleNotFoundError:
	from rag_supabase import SupabaseVectorRetriever, search_supabase

load_dotenv()


def compare_search_methods(query: str, top_k: int = 4):
	"""Compare pgvector search vs hybrid search."""
	
	print("=" * 80)
	print(f"Query: {query}")
	print("=" * 80)
	
	# 1. PGVECTOR SEARCH ONLY
	print("\n1️⃣  PGVECTOR SEARCH ONLY (Semantic similarity)")
	print("-" * 80)
	vector_retriever = SupabaseVectorRetriever(top_k=top_k)
	vector_results = vector_retriever.invoke(query)
	
	print(f"Retrieved {len(vector_results)} results:\n")
	for i, doc in enumerate(vector_results, 1):
		similarity = doc.metadata.get("similarity", 0)
		print(f"{i}. [{doc.metadata.get('source', 'unknown')} - page {doc.metadata.get('page', '?')}] (similarity: {similarity:.3f})")
		print(f"   {doc.page_content[:80]}...")
		print()
	
	# 2. HYBRID SEARCH (pgvector 60% + BM25 40%)
	print("\n2️⃣  HYBRID SEARCH (60% pgvector + 40% BM25)")
	print("-" * 80)
	hybrid_results = search_supabase(query, top_k=top_k)
	
	print(f"Retrieved {len(hybrid_results)} results:\n")
	for i, doc in enumerate(hybrid_results, 1):
		similarity = doc.metadata.get("similarity", "N/A")
		print(f"{i}. [{doc.metadata.get('source', 'unknown')} - page {doc.metadata.get('page', '?')}]")
		print(f"   {doc.page_content[:80]}...")
		print()
	
	# 3. COMPARISON
	print("\n📊 COMPARISON")
	print("-" * 80)
	print(f"Vector-only results: {len(vector_results)} docs")
	print(f"Hybrid results: {len(hybrid_results)} docs")
	
	# Check overlap
	vector_content = {doc.page_content[:50] for doc in vector_results}
	hybrid_content = {doc.page_content[:50] for doc in hybrid_results}
	overlap = len(vector_content & hybrid_content)
	
	print(f"Result overlap: {overlap}/{max(len(vector_results), len(hybrid_results))} docs")
	print(f"\n✅ RECOMMENDATION:")
	print("- Use HYBRID if you want both semantic AND keyword relevance")
	print("- Use PGVECTOR ONLY if you want pure semantic similarity")


if __name__ == "__main__":
	# Test query
	query = "What is the attention mechanism in transformers?"
	compare_search_methods(query, top_k=4)
	
	print("\n" + "=" * 80)
	print("Try other queries:")
	print("- 'transformer architecture'")
	print("- 'neural networks'")
	print("- 'encoder decoder'")
	print("=" * 80)
