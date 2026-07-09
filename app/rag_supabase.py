"""
Supabase Vector Search Retrieval
Pure vector-based RAG using pgvector.
"""

from __future__ import annotations
import os
import psycopg
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAIEmbeddings

from app.embedding import get_embeddings_model
from app.document_loader import process_all_pdfs, document_splitter

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


def store_to_supabase(data_dir: str = "./data", force_reload: bool = False):
	"""Load PDFs, embed them, store in Supabase."""
	db_uri = os.getenv("SUPABASE_DATABASE_URL")
	if not db_uri:
		raise ValueError("SUPABASE_DATABASE_URL not set")
	
	# Initialize table first
	init_supabase()
	
	# Check if documents already exist
	if not force_reload:
		with psycopg.connect(db_uri, connect_timeout=10) as conn:
			with conn.cursor() as cur:
				cur.execute("SELECT COUNT(*) FROM rag_docs")
				count = cur.fetchone()[0]
				if count > 0:
					print(f"✓ {count} documents already in Supabase (skipping reload)")
					return
	
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
	
	# Batch embed all chunks at once (much faster than individual API calls)
	print("🔀 Generating embeddings in batches...")
	texts = [chunk.page_content for chunk in chunks]
	batch_embeddings = embeddings.embed_documents(texts)
	
	# Store in Supabase
	print("💾 Storing in Supabase...")
	with psycopg.connect(db_uri, connect_timeout=10) as conn:
		with conn.cursor() as cur:
			cur.execute("DELETE FROM rag_docs")
			
			for i, (chunk, embedding) in enumerate(zip(chunks, batch_embeddings)):
				cur.execute(
					"INSERT INTO rag_docs (content, embedding, source, page) VALUES (%s, %s, %s, %s)",
					(
						chunk.page_content,
						embedding,
						chunk.metadata.get("source", "unknown"),
						chunk.metadata.get("page", 0)
					)
				)
				if (i + 1) % 100 == 0:
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
	"""Vector search from Supabase."""
	retriever = SupabaseVectorRetriever(top_k=top_k)
	return retriever.invoke(query)


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
