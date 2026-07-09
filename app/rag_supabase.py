"""
Supabase + BM25 Hybrid Retrieval Pipeline
Integrates pgvector similarity search with local keyword search.
"""

from __future__ import annotations
import os
import psycopg
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_openai import OpenAIEmbeddings

from app.embedding import get_embeddings_model
from app.document_loader import process_all_pdfs, document_splitter

try:
	from rag_basic.retriver import _load_or_rebuild_bm25
except (ModuleNotFoundError, ImportError) as e:
	print(f"Warning: Could not import BM25 retriever: {e}")
	_load_or_rebuild_bm25 = None

load_dotenv()


def init_supabase():
	"""One-time setup: Create table and enable pgvector."""
	db_uri = os.getenv("SUPABASE_DATABASE_URL")
	if not db_uri:
		raise ValueError("SUPABASE_DATABASE_URL not set. Add it to your .env file.")

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
	"""Load PDFs, embed them, store in Supabase."""
	db_uri = os.getenv("SUPABASE_DATABASE_URL")
	if not db_uri:
		raise ValueError("SUPABASE_DATABASE_URL not set")
	
	# Initialize table first
	init_supabase()
	
	# Load and split documents
	print("📄 Loading PDFs...")
	documents = process_all_pdfs(data_dir)
	if not documents:
		print("⚠️  No PDFs found in", data_dir)
		return
	
	chunks = document_splitter(documents)
	print(f"✂️  Split into {len(chunks)} chunks")
	
	# Get embeddings
	embeddings = get_embeddings_model()
	
	# Store in Supabase (clear first to avoid duplicates)
	print("💾 Storing in Supabase...")
	with psycopg.connect(db_uri, connect_timeout=10) as conn:
		with conn.cursor() as cur:
			cur.execute("DELETE FROM rag_docs")
			print("🗑️  Cleared existing records")
			
			for i, chunk in enumerate(chunks):
				embedding = embeddings.embed_query(chunk.page_content)
				cur.execute(
					"INSERT INTO rag_docs (content, embedding, source, page) VALUES (%s, %s, %s, %s)",
					(
						chunk.page_content,
						embedding,
						chunk.metadata.get("source", "unknown"),
						chunk.metadata.get("page", 0)
					)
				)
				if (i + 1) % 10 == 0:
					print(f"  ... stored {i + 1}/{len(chunks)} chunks")
			
			conn.commit()
	print(f"✅ Stored {len(chunks)} documents in Supabase")


class SupabaseVectorRetriever(BaseRetriever):
	"""Wrapper for Supabase vector search to work with EnsembleRetriever."""
	
	top_k: int = 4
	
	def _get_relevant_documents(self, query: str) -> list[Document]:
		"""Search Supabase vectors."""
		db_uri = os.getenv("SUPABASE_DATABASE_URL")
		if not db_uri:
			print("Error: SUPABASE_DATABASE_URL not set")
			return []
		
		embeddings = get_embeddings_model()
		
		try:
			query_embedding = embeddings.embed_query(query)
			
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
			
			docs = []
			for content, source, page, similarity in results:
				doc = Document(
					page_content=content,
					metadata={"source": source, "page": page, "similarity": similarity}
				)
				docs.append(doc)
			
			return docs
		except Exception as e:
			print(f"Error querying Supabase: {e}")
			return []


def search_supabase(query: str, top_k: int = 4) -> list[Document]:
	"""Hybrid search: Supabase vector (60%) + optional BM25 keyword (40%)."""
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
	"""
	Script to populate Supabase with documents.
	Run: uv run python -m app.rag_supabase
	or: uv run python app/rag_supabase.py
	"""
	import sys
	
	print("🚀 Starting RAG setup...")
	
	# Check required env vars
	if not os.getenv("SUPABASE_DATABASE_URL"):
		print("❌ Error: SUPABASE_DATABASE_URL not set")
		sys.exit(1)
	
	if not os.getenv("OPENAI_API_KEY"):
		print("❌ Error: OPENAI_API_KEY not set")
		sys.exit(1)
	
	# Initialize and populate
	try:
		store_to_supabase("./data")
		
		# Test search
		print("\n🔍 Testing search...")
		results = search_supabase("machine learning")
		print(f"Found {len(results)} results for 'machine learning'")
		for i, doc in enumerate(results[:2], 1):
			print(f"\n[Result {i}]")
			print(f"Source: {doc.metadata.get('source', 'unknown')}")
			print(f"Content: {doc.page_content[:100]}...")
	except Exception as e:
		print(f"❌ Error: {e}")
		sys.exit(1)
