from __future__ import annotations

from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

try:
	from app.retriver import build_hybrid_retriever
except ModuleNotFoundError:
	from retriver import build_hybrid_retriever

load_dotenv()


@dataclass
class RAGResult:
	query: str
	answer: str
	source_count: int
	sources: list[dict]


def _format_context(docs: list) -> str:
	"""Convert retrieved docs into a compact prompt context."""
	blocks: list[str] = []

	for i, doc in enumerate(docs, start=1):
		source = doc.metadata.get("source_file", "unknown")
		page = doc.metadata.get("page_label", doc.metadata.get("page", "?"))
		blocks.append(
			f"[Source {i}] file={source}, page={page}\n{doc.page_content.strip()}"
		)

	return "\n\n".join(blocks)


def generate_augmented_answer(
	query: str,
	model_name: str = "gpt-4o-mini",
	top_k: int = 4,
) -> RAGResult:
	"""Run retrieval-augmented generation using the existing hybrid retriever."""

	retriever = build_hybrid_retriever(vector_k=top_k, bm25_k=top_k)
	docs = retriever.invoke(query)

	if not docs:
		return RAGResult(
			query=query,
			answer="I could not find relevant context in the indexed documents.",
			source_count=0,
			sources=[],
		)

	context = _format_context(docs)
	llm = ChatOpenAI(model=model_name, temperature=0)

	prompt = (
		"You are a helpful assistant. Answer the user question using only the provided context. "
		"If context is insufficient, clearly say so. Keep the answer concise and factual.\n\n"
		f"Question:\n{query}\n\n"
		f"Context:\n{context}\n\n"
		"Return a short answer followed by a bullet list of source references in this format: "
		"file name + page."
	)

	response = llm.invoke(prompt)

	sources = [
		{
			"source_file": doc.metadata.get("source_file", "unknown"),
			"page": doc.metadata.get("page_label", doc.metadata.get("page", "?")),
		}
		for doc in docs
	]

	return RAGResult(
		query=query,
		answer=response.content,
		source_count=len(sources),
		sources=sources,
	)


if __name__ == "__main__":
	query = "What is the attention mechanism in transformers?"
	result = generate_augmented_answer(query)

	print("\n=== Augmented Generation Output ===")
	print(f"Query: {result.query}")
	print(f"Retrieved sources: {result.source_count}")
	print("\nAnswer:")
	print(result.answer)
