"""
Supabase + BM25 Hybrid Retrieval Pipeline
Integrates pgvector similarity search with local keyword search.
"""

from __future__ import annotations
import os
import sys
import psycopg
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_openai import OpenAIEmbeddings

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
	from rag_basic.retriver import _load_or_rebuild_bm25
except (ModuleNotFoundError, ImportError) as e:
	print(f"Warning: Could not import BM25 retriever: {e}")
	_load_or_rebuild_bm25 = None

load_dotenv()


def _get_embeddings_model(model: str = "text-embedding-3-small") -> OpenAIEmbeddings:
	"""Create embeddings client for Supabase vector search."""
	api_key = os.getenv("OPENAI_API_KEY")
	if not api_key:
		raise ValueError("OPENAI_API_KEY not found in environment")
	return OpenAIEmbeddings(model=model, api_key=api_key)


def init_supabase():
	"""One-time setup: Create table and enable pgvector."""
	db_uri = os.getenv("SUPABASE_DATABASE_URL")
	if not db_uri:
		raise ValueError("Add SUPABASE_DATABASE_URL to .env")

	try:
		with psycopg.connect(db_uri, connect_timeout=10) as conn:
			with conn.cursor() as cur:
				# Enable pgvector
				cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
				
				# Create simple docs table
				cur.execute("""
					CREATE TABLE IF NOT EXISTS rag_docs (
						id SERIAL PRIMARY KEY,
						content TEXT,
						embedding vector(1536),
						source TEXT,
						page INT
					)
				""")
				
				# Index for search
				cur.execute("""
					CREATE INDEX IF NOT EXISTS rag_embedding_idx 
					ON rag_docs USING ivfflat (embedding vector_cosine_ops)
				""")
				
				conn.commit()
				print("✓ Supabase table created")
	except Exception as e:
		print(f"Error initializing Supabase: {e}")
		raise


def store_to_supabase(data_dir: str = "./data"):
	"""Load PDFs, embed them, store in Supabase - SIMPLIFIED VERSION."""
	db_uri = os.getenv("SUPABASE_DATABASE_URL")
	if not db_uri:
		raise ValueError("SUPABASE_DATABASE_URL not set")
	
	# Simple message - full implementation would need PDF loader
	print(f"📄 Would load PDFs from {data_dir}")
	print("💾 This function requires full production_rag setup")
	print("✓ For now, using existing Supabase data")


class SupabaseVectorRetriever(BaseRetriever):
	"""Wrapper for Supabase vector search to work with EnsembleRetriever."""
	
	top_k: int = 4
	
	def _get_relevant_documents(self, query: str) -> list[Document]:
		"""Search Supabase vectors."""
		db_uri = os.getenv("SUPABASE_DATABASE_URL")
		if not db_uri:
			raise ValueError("SUPABASE_DATABASE_URL not set")
		
		embeddings = _get_embeddings_model()
		query_embedding = embeddings.embed_query(query)
		
		try:
			with psycopg.connect(db_uri, connect_timeout=10) as conn:
				with conn.cursor() as cur:
					cur.execute("""
						SELECT content, source, page,
							   1 - (embedding <=> %s::vector) as similarity
						FROM rag_docs
						ORDER BY embedding <=> %s::vector
						LIMIT %s
					""", (str(query_embedding), str(query_embedding), self.top_k))
					
					results = cur.fetchall()
		except Exception as e:
			print(f"Error querying Supabase: {e}")
			return []
		
		docs = []
		for content, source, page, similarity in results:
			doc = Document(
				page_content=content,
				metadata={"source": source, "page": page, "similarity": similarity}
			)
			docs.append(doc)
		
		return docs


def search_supabase(query: str, top_k: int = 4) -> list[Document]:
	"""Hybrid search: Supabase vector + optional BM25 keyword."""
	# 1. Vector search from Supabase
	vector_retriever = SupabaseVectorRetriever(top_k=top_k)
	vector_docs = vector_retriever.invoke(query)
	
	# 2. Try BM25 if available
	if _load_or_rebuild_bm25:
		try:
			bm25_retriever = _load_or_rebuild_bm25(
				data_dir="./data",
				bm25_k=top_k,
				cache_dir="./bm25_cache",
			)
			
			# 3. Ensemble: combine both (60% vector, 40% BM25)
			ensemble = EnsembleRetriever(
				retrievers=[vector_retriever, bm25_retriever],
				weights=[0.6, 0.4]
			)
			return ensemble.invoke(query)
		except Exception as e:
			print(f"BM25 search failed, using vector-only: {e}")
			return vector_docs
	
	return vector_docs


if __name__ == "__main__":
	init_supabase()
	store_to_supabase()
	
	# Test search
	results = search_supabase("attention mechanism")
	print(f"\n🔍 Found {len(results)} results:")
	for i, doc in enumerate(results, 1):
		print(f"{i}. {doc.page_content[:100]}...")
