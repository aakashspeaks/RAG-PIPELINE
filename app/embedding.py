from __future__ import annotations

import os

import numpy as np
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from app.document_loader import process_all_pdfs, document_splitter

load_dotenv()


def get_embeddings_model(model: str = "text-embedding-3-small") -> OpenAIEmbeddings:
	"""Create one embeddings client for reuse."""
	api_key = os.getenv("OPENAI_API_KEY")
	if not api_key:
		raise ValueError("OPENAI_API_KEY is not set. Add it to your environment or .env file.")
	return OpenAIEmbeddings(model=model)


def load_and_split_chunks(data_dir: str = "./data") -> list[Document]:
	"""Load PDFs and split them into chunked LangChain documents."""
	documents = process_all_pdfs(data_dir)
	return document_splitter(documents, strategy="hybrid")


def embed_chunk_documents(
	chunk_docs: list[Document],
	embeddings_model: OpenAIEmbeddings | None = None,
	batch_size: int = 128,
) -> list[list[float]]:
	"""Embed chunk documents with simple batching."""
	if not chunk_docs:
		return []

	texts = [doc.page_content.strip() for doc in chunk_docs if doc.page_content and doc.page_content.strip()]
	if not texts:
		return []

	model = embeddings_model or get_embeddings_model()
	vectors: list[list[float]] = []
	for i in range(0, len(texts), batch_size):
		batch = texts[i : i + batch_size]
		vectors.extend(model.embed_documents(batch))

	vector_dim = len(next((v for v in vectors if v), []))
	avg_norm = float(np.mean([np.linalg.norm(v) for v in vectors])) if vectors else 0.0

	print(f"Embedded {len(texts)} chunks")
	print(f"Unique chunk texts embedded: {len(texts)}")
	print(f"Vector dimensions: {vector_dim}")
	print(f"Average vector norm: {avg_norm:.4f}")

	return vectors


def embed_chunks_from_data(
	data_dir: str = "./data",
	embeddings_model: OpenAIEmbeddings | None = None,
	batch_size: int = 128,
	chunking_strategy: str = "hybrid",
) -> tuple[list[Document], list[list[float]]]:
	"""One-call helper: load PDFs, split to chunks, and embed those chunks."""
	documents = process_all_pdfs(data_dir)
	chunk_docs = document_splitter(documents, strategy=chunking_strategy)
	vectors = embed_chunk_documents(chunk_docs, embeddings_model=embeddings_model, batch_size=batch_size)
	return chunk_docs, vectors
