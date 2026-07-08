from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma

try:
	from app.document_loader import process_all_pdfs, document_splitter
	from app.embedding import get_embeddings_model
except ModuleNotFoundError:
	from document_loader import process_all_pdfs, document_splitter
	from embedding import get_embeddings_model

load_dotenv()


def build_chroma_vector_store(
	data_dir: str = "./data",
	persist_directory: str = "./chroma_db",
	collection_name: str = "pdf_chunks",
) -> Chroma:
	"""Load PDFs, split into chunks, and persist them in Chroma."""
	documents = process_all_pdfs(data_dir)
	chunks = document_splitter(documents)

	embeddings_model = get_embeddings_model()
	persist_path = Path(persist_directory)
	persist_path.mkdir(parents=True, exist_ok=True)

	vector_store = Chroma.from_documents(
		documents=chunks,
		embedding=embeddings_model,
		collection_name=collection_name,
		persist_directory=str(persist_path),
	)

	print(f"Stored {len(chunks)} chunks in Chroma")
	print(f"Collection name: {collection_name}")
	print(f"Persist directory: {persist_path.resolve()}")

	return vector_store


if __name__ == "__main__":
	build_chroma_vector_store()
