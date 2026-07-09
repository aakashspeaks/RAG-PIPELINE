from __future__ import annotations
import os
import psycopg
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_classic.retrievers import EnsembleRetriever

try:
	from production_rag.embedding import get_embeddings_model
	from production_rag.document_loader import process_all_pdfs, document_splitter
	from rag_basic.retriver import _load_or_rebuild_bm25
except ModuleNotFoundError:
	from embedding import get_embeddings_model
	from document_loader import process_all_pdfs, document_splitter
	from rag_basic.retriver import _load_or_rebuild_bm25

load_dotenv()


def init_supabase():
	"""One-time setup: Create table and enable pgvector."""
	db_uri = os.getenv("SUPABASE_URL")
	if not db_uri:
		raise ValueError("Add SUPABASE_URL to .env")

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


def store_to_supabase(data_dir: str = "./data"):
	"""Load PDFs, embed them, store in Supabase."""
	db_uri = os.getenv("SUPABASE_URL")
	
	# Load and split documents
	print("📄 Loading PDFs...")
	documents = process_all_pdfs(data_dir)
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
			for chunk in chunks:
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
			conn.commit()
	print(f"✅ Stored {len(chunks)} documents")


class SupabaseVectorRetriever(BaseRetriever):
	"""Wrapper for Supabase vector search to work with EnsembleRetriever."""
	
	top_k: int = 4
	
	def _get_relevant_documents(self, query: str) -> list[Document]:
		"""Search Supabase vectors."""
		db_uri = os.getenv("SUPABASE_URL")
		embeddings = get_embeddings_model()
		
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


def search_supabase(query: str, top_k: int = 4) -> list[Document]:
	"""Hybrid search: Supabase vector (60%) + BM25 keyword (40%)."""
	# 1. Vector search from Supabase
	vector_retriever = SupabaseVectorRetriever(top_k=top_k)
	
	# 2. BM25 keyword search locally
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


if __name__ == "__main__":
	init_supabase()
	store_to_supabase()
	
	# Test search
	results = search_supabase("attention mechanism")
	print(f"\n🔍 Found {len(results)} results:")
	for i, doc in enumerate(results, 1):
		print(f"{i}. {doc.page_content[:100]}...")
