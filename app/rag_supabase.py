"""
Supabase Vector Search Retrieval
Pure vector-based RAG using pgvector.
"""

from __future__ import annotations
import os
import re
import string
import psycopg
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAIEmbeddings

from app.config import get_settings
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
	
	top_k: int = 8
	
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


def _safe_normalize(value: float, min_value: float, max_value: float) -> float:
	"""Normalize score to [0,1] with divide-by-zero safety."""
	if max_value <= min_value:
		return 0.0
	return (value - min_value) / (max_value - min_value)


def _query_terms(query: str) -> list[str]:
	"""Extract lowercase lexical terms for lightweight term-coverage scoring."""
	stopwords = {
		"what", "is", "the", "a", "an", "of", "about", "define", "explain",
		"tell", "me", "in", "to", "for", "and", "on", "with",
	}
	return [
		t for t in re.findall(r"[a-zA-Z0-9]+", query.lower())
		if len(t) >= 3 and t not in stopwords
	]


def _normalize_text(value: str) -> str:
	"""Normalize text for robust lexical comparisons across punctuation/hyphens."""
	value = value.lower()
	value = value.replace("-", " ").replace("–", " ").replace("—", " ")
	value = value.translate(str.maketrans("", "", string.punctuation))
	value = re.sub(r"\s+", " ", value).strip()
	return value


def _query_phrase_candidates(query: str) -> list[str]:
	"""Generate phrase aliases for lexical matching and rerank boosts."""
	norm = _normalize_text(query)
	phrases = [norm] if norm else []
	terms = set(_query_terms(query))

	# GPT-focused aliases to catch common user phrasing variants.
	if "gpt" in terms or {"generative", "pre", "trained", "transformer"}.issubset(terms):
		phrases.extend([
			"generative pre trained transformer",
			"generative pretrained transformer",
			"gpt",
			"gpt 1",
			"gpt 2",
			"gpt 3",
			"gpt 4",
		])

	# Keep unique order and drop short/noisy entries.
	seen = set()
	result = []
	for phrase in phrases:
		if len(phrase) < 3:
			continue
		if phrase in seen:
			continue
		seen.add(phrase)
		result.append(phrase)
	return result


def _term_coverage(text: str, terms: list[str]) -> float:
	"""Fraction of query terms found in candidate text."""
	if not terms:
		return 0.0
	text_lower = _normalize_text(text)
	hits = sum(1 for term in terms if term in text_lower)
	return hits / len(terms)


def _hybrid_search_supabase(query: str, top_k: int) -> list[Document]:
	"""Hybrid retrieval: vector + BM25-like lexical ranking + weighted rerank."""
	db_uri = os.getenv("SUPABASE_DATABASE_URL")
	if not db_uri:
		print("Error: SUPABASE_DATABASE_URL not set")
		return []

	settings = get_settings()
	pool_size = max(top_k, settings.retrieval_candidate_pool)

	# Normalize weights if env overrides make them sum differently.
	vector_w = float(settings.retrieval_vector_weight)
	bm25_w = float(settings.retrieval_bm25_weight)
	phrase_w = float(settings.retrieval_phrase_weight)
	term_w = float(settings.retrieval_term_weight)
	weight_total = vector_w + bm25_w + phrase_w + term_w
	if weight_total <= 0:
		vector_w, bm25_w, phrase_w, term_w = 0.55, 0.35, 0.07, 0.03
		weight_total = 1.0
	vector_w /= weight_total
	bm25_w /= weight_total
	phrase_w /= weight_total
	term_w /= weight_total

	embeddings = get_embeddings_model()
	query_embedding = embeddings.embed_query(query)
	phrase_candidates = _query_phrase_candidates(query)
	query_lexical = " OR ".join(
		f'"{phrase}"' if " " in phrase else phrase
		for phrase in phrase_candidates
	)
	if not query_lexical:
		query_lexical = query

	vector_rows = []
	lexical_rows = []
	try:
		with psycopg.connect(db_uri, connect_timeout=10) as conn:
			with conn.cursor() as cur:
				cur.execute(
					"""
					SELECT content, source, page,
					       1 - (embedding <=> %s::vector) AS vector_score
					FROM rag_docs
					ORDER BY embedding <=> %s::vector
					LIMIT %s
					""",
					(str(query_embedding), str(query_embedding), pool_size),
				)
				vector_rows = cur.fetchall()

				cur.execute(
					"""
					SELECT content, source, page,
					       ts_rank_cd(
					           to_tsvector('english', content),
					           websearch_to_tsquery('english', %s)
					       ) AS bm25_score
					FROM rag_docs
					WHERE to_tsvector('english', content) @@ websearch_to_tsquery('english', %s)
					ORDER BY bm25_score DESC
					LIMIT %s
					""",
					(query_lexical, query_lexical, pool_size),
				)
				lexical_rows = cur.fetchall()
	except Exception as e:
		print(f"Hybrid retrieval failed, falling back to vector search: {e}")
		return SupabaseVectorRetriever(top_k=top_k).invoke(query)

	# Merge candidates by exact content+source+page identity.
	candidates: dict[tuple[str, str, int], dict] = {}
	for content, source, page, vector_score in vector_rows:
		key = (content, source, int(page or 0))
		candidates[key] = {
			"content": content,
			"source": source,
			"page": int(page or 0),
			"vector_score": float(vector_score or 0.0),
			"bm25_score": 0.0,
		}

	for content, source, page, bm25_score in lexical_rows:
		key = (content, source, int(page or 0))
		if key not in candidates:
			candidates[key] = {
				"content": content,
				"source": source,
				"page": int(page or 0),
				"vector_score": 0.0,
				"bm25_score": float(bm25_score or 0.0),
			}
		else:
			candidates[key]["bm25_score"] = float(bm25_score or 0.0)

	if not candidates:
		return []

	query_lower = _normalize_text(query)
	terms = _query_terms(query)
	vector_values = [c["vector_score"] for c in candidates.values()]
	bm25_values = [c["bm25_score"] for c in candidates.values()]
	min_vec, max_vec = min(vector_values), max(vector_values)
	min_bm25, max_bm25 = min(bm25_values), max(bm25_values)

	reranked = []
	for c in candidates.values():
		vec_norm = _safe_normalize(c["vector_score"], min_vec, max_vec)
		bm25_norm = _safe_normalize(c["bm25_score"], min_bm25, max_bm25)
		normalized_content = _normalize_text(c["content"])
		phrase_bonus = 0.0
		for phrase in phrase_candidates:
			if phrase and phrase in normalized_content:
				phrase_bonus = 1.0
				break
		if query_lower and query_lower in normalized_content:
			phrase_bonus = 1.0
		term_ratio = _term_coverage(c["content"], terms)
		final_score = (
			(vector_w * vec_norm)
			+ (bm25_w * bm25_norm)
			+ (phrase_w * phrase_bonus)
			+ (term_w * term_ratio)
		)
		reranked.append(
			{
				**c,
				"final_score": final_score,
				"vector_norm": vec_norm,
				"bm25_norm": bm25_norm,
				"phrase_bonus": phrase_bonus,
				"term_ratio": term_ratio,
			}
		)

	reranked.sort(key=lambda row: row["final_score"], reverse=True)
	selected = reranked[:top_k]

	docs = []
	for row in selected:
		docs.append(
			Document(
				page_content=row["content"],
				metadata={
					"source": row["source"],
					"page": row["page"],
					"similarity": row["final_score"],
					"vector_score": row["vector_score"],
					"bm25_score": row["bm25_score"],
					"phrase_bonus": row["phrase_bonus"],
					"term_ratio": row["term_ratio"],
					"retrieval_mode": "hybrid",
				},
			)
		)

	return docs


def search_supabase(query: str, top_k: int = 8) -> list[Document]:
	"""Hybrid semantic + lexical search from Supabase with reranking."""
	return _hybrid_search_supabase(query=query, top_k=top_k)


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
