from __future__ import annotations

import os
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()


def get_supabase_client() -> Client:
	"""Create and return a Supabase client from environment variables."""
	supabase_url = os.getenv("SUPABASE_URL")
	supabase_key = os.getenv("SUPABASE_API_KEY")

	if not supabase_url:
		raise ValueError("SUPABASE_URL is missing. Add it to your .env file.")
	if not supabase_key:
		raise ValueError("SUPABASE_API_KEY is missing. Add it to your .env file.")

	return create_client(supabase_url, supabase_key)


def test_supabase_connection() -> bool:
	"""Run a lightweight connectivity check against Supabase REST endpoint."""
	supabase_url = os.getenv("SUPABASE_URL")
	supabase_key = os.getenv("SUPABASE_API_KEY")

	if not supabase_url or not supabase_key:
		raise ValueError("SUPABASE_URL or SUPABASE_API_KEY is missing.")

	rest_url = f"{supabase_url.rstrip('/')}/rest/v1/"
	request = Request(
		rest_url,
		headers={
			"apikey": supabase_key,
			"Authorization": f"Bearer {supabase_key}",
		},
	)

	with urlopen(request, timeout=10) as response:
		return 200 <= response.status < 300


def test_postgres_uri_connection() -> bool:
	"""Test direct Postgres connection using SUPABASE_DATABASE_URL."""
	db_uri = os.getenv("SUPABASE_DATABASE_URL")
	if not db_uri:
		raise ValueError("SUPABASE_DATABASE_URL is missing. Add it to your .env file.")

	# Lazy import so REST mode still works even if psycopg is not installed.
	try:
		import psycopg
	except ImportError as exc:
		raise ImportError("psycopg is required for URI mode. Install with: uv add 'psycopg[binary]>=3.2.0'") from exc

	with psycopg.connect(db_uri, connect_timeout=10) as conn:
		with conn.cursor() as cur:
			cur.execute("select 1")
			row = cur.fetchone()
			return bool(row and row[0] == 1)


def test_connection() -> bool:
	"""Auto-select connection mode: URI first, then REST."""
	if os.getenv("SUPABASE_DATABASE_URL"):
		return test_postgres_uri_connection()
	return test_supabase_connection()


def store_documents_in_supabase(
	documents: list[dict],
	table_name: str = "documents",
) -> None:
	"""
	Store documents with embeddings in Supabase.
	
	Expected document structure:
	{
		"content": str,
		"embedding": list[float],
		"source_file": str,
		"page": int,
		"metadata": dict
	}
	"""
	client = get_supabase_client()
	
	try:
		response = client.table(table_name).insert(documents).execute()
		print(f"Stored {len(documents)} documents in '{table_name}' table")
		return response.data
	except Exception as exc:
		print(f"Error storing documents: {exc}")
		raise


def retrieve_similar_documents(
	query_embedding: list[float],
	table_name: str = "documents",
	top_k: int = 4,
	similarity_threshold: float = 0.5,
) -> list[dict]:
	"""
	Retrieve documents from Supabase using pgvector similarity search.
	Uses cosine similarity by default.
	"""
	client = get_supabase_client()
	
	try:
		# Call the RPC function for vector similarity search
		response = client.rpc(
			"search_documents",
			{
				"query_embedding": query_embedding,
				"similarity_threshold": similarity_threshold,
				"match_count": top_k,
			}
		).execute()
		
		return response.data
	except Exception as exc:
		print(f"Error retrieving documents: {exc}")
		raise


def setup_pgvector_table(table_name: str = "documents") -> bool:
	"""
	Set up pgvector extension and documents table in Supabase.
	Run this once to initialize your vector store.
	"""
	import psycopg
	
	db_uri = os.getenv("SUPABASE_DATABASE_URL")
	if not db_uri:
		raise ValueError("SUPABASE_DATABASE_URL is missing.")
	
	try:
		with psycopg.connect(db_uri, connect_timeout=10) as conn:
			with conn.cursor() as cur:
				# Enable pgvector extension
				cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
				
				# Create documents table with vector column
				cur.execute(f"""
					CREATE TABLE IF NOT EXISTS {table_name} (
						id BIGSERIAL PRIMARY KEY,
						content TEXT NOT NULL,
						embedding vector(1536),
						source_file TEXT,
						page INTEGER,
						metadata JSONB,
						created_at TIMESTAMP DEFAULT NOW()
					)
				""")
				
				# Create index for faster similarity search
				cur.execute(f"""
					CREATE INDEX IF NOT EXISTS {table_name}_embedding_idx
					ON {table_name}
					USING ivfflat (embedding vector_cosine_ops)
				""")
				
				conn.commit()
				print(f"✓ pgvector table '{table_name}' set up successfully")
				return True
	except Exception as exc:
		print(f"Error setting up pgvector table: {exc}")
		raise


if __name__ == "__main__":
	try:
		ok = test_connection()
		mode = "Postgres URI" if os.getenv("SUPABASE_DATABASE_URL") else "Supabase REST"
		print(f"{mode} connection successful." if ok else f"{mode} connection check failed.")
	except Exception as exc:
		print(f"Supabase connection error: {exc}")
