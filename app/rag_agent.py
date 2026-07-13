"""
RAG-Enhanced LangGraph Agent
Integrates Supabase vector search + BM25 with LLM generation.
"""

from typing import Optional
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langsmith import traceable

from app.config import get_settings
from app.rag_supabase import search_supabase
from app.monitoring import get_logger

logger = get_logger()


# === RAG Agent State ===

class RAGAgentState(TypedDict):
	"""State for RAG-enhanced agent."""
	messages: Annotated[list[BaseMessage], add_messages]
	query: str
	retrieved_docs: list
	context: str
	error: Optional[str]
	retry_count: int
	model_used: str


# === RAG Agent Builder ===

class RAGAgent:
	"""
	Production RAG agent with:
	- Hybrid retrieval (vector + keyword search)
	- Context injection into LLM prompt
	- Retry with fallback
	- LangSmith tracing
	"""

	def __init__(self):
		settings = get_settings()

		if not settings.openai_api_key:
			raise ValueError(
				"OPENAI_API_KEY is not set. "
				"Add it to your environment variables on Render."
			)

		self.primary_llm = ChatOpenAI(
			model=settings.primary_model,
			temperature=0,
			timeout=30,
			max_retries=0,
			api_key=settings.openai_api_key,
		)
		self.fallback_llm = ChatOpenAI(
			model=settings.fallback_model,
			temperature=0,
			timeout=30,
			max_retries=0,
			api_key=settings.openai_api_key,
		)
		self.max_retries = settings.max_retries
		self.retrieval_top_k = settings.retrieval_top_k
		self.graph = self._build_graph()

	def _build_graph(self):
		"""Build the RAG-aware LangGraph state machine."""

		def retrieve_documents(state: RAGAgentState) -> dict:
			"""Retrieve documents from Supabase using hybrid search."""
			try:
				query = state.get("query") or state["messages"][-1].content
				logger.info(f"RAG: Retrieving documents for query: {query[:50]}...")
				
				docs = search_supabase(query, top_k=self.retrieval_top_k)

				# Prioritize chunks that have lexical evidence for the current query.
				evidence_docs = [
					doc for doc in docs
					if float(doc.metadata.get("phrase_bonus", 0.0)) > 0.0
					or float(doc.metadata.get("bm25_score", 0.0)) > 0.0
					or float(doc.metadata.get("term_ratio", 0.0)) >= 0.45
				]
				if evidence_docs:
					docs = evidence_docs[: self.retrieval_top_k]
				
				if not docs:
					logger.warning("RAG: No documents retrieved from Supabase. Check if SUPABASE_DATABASE_URL is set and documents exist.")
				else:
					logger.info(f"RAG: Retrieved {len(docs)} documents")
				
				# Format context from docs
				context_blocks = []
				for i, doc in enumerate(docs, 1):
					source = doc.metadata.get("source", "unknown")
					page = doc.metadata.get("page", "?")
					context_blocks.append(
						f"[Doc {i}] {source} (page {page})\n{doc.page_content}"
					)
				
				context = "\n\n".join(context_blocks)
				
				return {
					"retrieved_docs": docs,
					"context": context,
					"error": None,
				}
			except Exception as e:
				logger.error(f"RAG: Retrieval failed: {str(e)}")
				return {
					"retrieved_docs": [],
					"context": "",
					"error": f"Retrieval failed: {str(e)}",
				}

		def generate_with_context(state: RAGAgentState) -> dict:
			"""Generate response using LLM with retrieved context."""
			try:
				query = state.get("query") or state["messages"][-1].content
				context = state.get("context", "")
				retrieved_docs = state.get("retrieved_docs", [])
				
				if context and retrieved_docs:
					# RAG mode: use context
					logger.info("RAG: Using retrieved context for generation (RAG mode)")
					prompt = (
						"You are a helpful assistant. Answer the user's question using ONLY the provided context. "
						"Prefer direct definitions when present. Quote short supporting phrases from context where useful. "
						"If the context truly lacks relevant information, say so clearly.\n\n"
						f"Context:\n{context}\n\n"
						f"Question: {query}\n\n"
						"Answer:"
					)
					response = self.primary_llm.invoke(prompt)
				else:
					# No documents found: return information not available message
					logger.warning("RAG: No documents retrieved, returning 'not found' response")
					response = AIMessage(
						content=f"I apologize, but I wasn't able to find any relevant information about '{query}' in our available documents. "
						"Please try asking a different question or check if the document contains information about this topic."
					)
				
				return {
					"messages": [response],
					"error": None,
					"model_used": "primary_rag",
				}
			except Exception as e:
				logger.error(f"RAG: Generation failed: {str(e)}")
				return {
					"error": str(e),
					"retry_count": state["retry_count"] + 1,
					"model_used": "",
				}

		def try_fallback(state: RAGAgentState) -> dict:
			"""Fallback: generate without context if primary fails."""
			try:
				query = state.get("query") or state["messages"][-1].content
				response = self.fallback_llm.invoke(query)
				
				return {
					"messages": [response],
					"error": None,
					"model_used": "fallback_rag",
				}
			except Exception as e:
				return {
					"error": str(e),
					"model_used": "",
				}

		def handle_error(state: RAGAgentState) -> dict:
			"""Return graceful error message."""
			return {
				"messages": [
					AIMessage(content=(
						"I apologize, but I encountered an error processing your request. "
						"Please try again in a moment."
					))
				],
				"model_used": "error_handler",
			}

		def route_after_generation(state: RAGAgentState) -> str:
			"""Route after primary generation attempt."""
			if state.get("error") is None:
				return "done"
			elif state["retry_count"] < self.max_retries:
				return "fallback"
			else:
				return "error"

		def route_after_fallback(state: RAGAgentState) -> str:
			"""Route after fallback attempt."""
			if state.get("error") is None:
				return "done"
			else:
				return "error"

		# Build graph
		graph = StateGraph(RAGAgentState)

		graph.add_node("retrieve", retrieve_documents)
		graph.add_node("generate", generate_with_context)
		graph.add_node("fallback", try_fallback)
		graph.add_node("error", handle_error)

		graph.add_edge(START, "retrieve")
		graph.add_edge("retrieve", "generate")
		graph.add_conditional_edges(
			"generate",
			route_after_generation,
			{"done": END, "fallback": "fallback", "error": "error"},
		)
		graph.add_conditional_edges(
			"fallback",
			route_after_fallback,
			{"done": END, "error": "error"},
		)
		graph.add_edge("error", END)

		return graph.compile()

	@traceable(name="rag_agent_invoke")
	def invoke(self, message: str) -> dict:
		"""
		Invoke the RAG agent.
		Returns: {"response": str, "model_used": str, "rag_mode": bool, "sources": list[str], "error": str | None}
		"""
		result = self.graph.invoke({
			"messages": [HumanMessage(content=message)],
			"query": message,
			"retrieved_docs": [],
			"context": "",
			"error": None,
			"retry_count": 0,
			"model_used": "",
		})

		# Extract sources from retrieved documents
		retrieved_docs = result.get("retrieved_docs", [])
		sources = list(set(
			doc.metadata.get("source", "unknown")
			for doc in retrieved_docs
		))
		
		rag_mode = len(retrieved_docs) > 0
		
		return {
			"response": result["messages"][-1].content,
			"model_used": result.get("model_used", "unknown"),
			"rag_mode": rag_mode,
			"sources": sources,
			"error": result.get("error"),
		}
