from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from dotenv import load_dotenv
load_dotenv()

try:
	from app.embedding import load_and_split_chunks
	from basic_rag.vector_store import build_chroma_vector_store
except ModuleNotFoundError:
	from app.embedding import load_and_split_chunks
	from basic_rag.vector_store import build_chroma_vector_store


def _compute_pdf_corpus_signature(data_dir: str) -> str:
	"""Build a stable signature from PDF file path + size + mtime."""
	base = Path(data_dir)
	pdf_files = sorted(base.rglob("*.pdf"))

	hasher = hashlib.sha256()
	for pdf_path in pdf_files:
		stat = pdf_path.stat()
		record = f"{pdf_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}\n"
		hasher.update(record.encode("utf-8"))

	return hasher.hexdigest()


def _load_or_rebuild_bm25(
	data_dir: str,
	bm25_k: int,
	cache_dir: str,
) -> BM25Retriever:
	"""Load BM25 from cache, rebuild only when source PDFs change."""
	cache_path = Path(cache_dir)
	cache_path.mkdir(parents=True, exist_ok=True)

	state_file = cache_path / "bm25_state.json"
	bm25_file = cache_path / "bm25.pkl"

	current_signature = _compute_pdf_corpus_signature(data_dir)
	cached_signature = None

	if state_file.exists():
		try:
			state = json.loads(state_file.read_text(encoding="utf-8"))
			cached_signature = state.get("pdf_corpus_signature")
		except (json.JSONDecodeError, OSError):
			cached_signature = None

	if bm25_file.exists() and cached_signature == current_signature:
		with bm25_file.open("rb") as f:
			bm25_retriever: BM25Retriever = pickle.load(f)
		bm25_retriever.k = bm25_k
		print("BM25 cache hit: loaded existing index")
		return bm25_retriever

	print("BM25 cache miss: rebuilding index")
	chunks = load_and_split_chunks(data_dir)
	bm25_retriever = BM25Retriever.from_documents(chunks)
	bm25_retriever.k = bm25_k

	with bm25_file.open("wb") as f:
		pickle.dump(bm25_retriever, f)

	state_file.write_text(
		json.dumps({"pdf_corpus_signature": current_signature}, indent=2),
		encoding="utf-8",
	)

	return bm25_retriever


def build_hybrid_retriever(
	data_dir: str = "./data",
	persist_directory: str = "./chroma_db",
	bm25_cache_directory: str = "./bm25_cache",
	collection_name: str = "pdf_chunks",
	bm25_k: int = 4,
	vector_k: int = 4,
	bm25_weight: float = 0.6,
	vector_weight: float = 0.4,
) -> EnsembleRetriever:
	"""Create a hybrid retriever (BM25 + vector similarity)."""

	bm25_retriever = _load_or_rebuild_bm25(
		data_dir=data_dir,
		bm25_k=bm25_k,
		cache_dir=bm25_cache_directory,
	)

	vector_store = build_chroma_vector_store(
		data_dir=data_dir,
		persist_directory=persist_directory,
		collection_name=collection_name,
	)
	vector_retriever = vector_store.as_retriever(
		search_type="similarity",
		search_kwargs={"k": vector_k},
	)

	hybrid_retriever = EnsembleRetriever(
		retrievers=[bm25_retriever, vector_retriever],
		weights=[bm25_weight, vector_weight],
	)

	return hybrid_retriever


if __name__ == "__main__":
	retriever = build_hybrid_retriever()
	query = "What is attention mechanism?"
	results = retriever.invoke(query)

	print(f"Hybrid retriever returned {len(results)} documents")
	for i, doc in enumerate(results[:3], start=1):
		print(f"[{i}] {doc.metadata.get('source_file', 'unknown')} - page {doc.metadata.get('page_label', doc.metadata.get('page', '?'))}")
		print(doc.page_content[:200])
		print("-" * 60)