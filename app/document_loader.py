from __future__ import annotations

import math
import os
import re
from typing import Protocol

from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 180

# Preserve section boundaries and sentence structure common in technical papers.
TECHNICAL_SEPARATORS = [
    "\n## ",
    "\n### ",
    "\n\n",
    "\n",
    ". ",
    " ",
    "",
]

DEFAULT_CHUNK_STRATEGY = "hybrid"
DEFAULT_SEMANTIC_THRESHOLD = 0.78
DEFAULT_SEMANTIC_MIN_CHARS = 320


class EmbeddingsLike(Protocol):
    """Minimal interface required for semantic chunk refinement."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity without adding extra dependencies."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentence-like units while preserving formula-heavy lines."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []

    # First preserve line boundaries often used in papers (equations, bullets, captions).
    units: list[str] = []
    for line in lines:
        # Split common sentence endings, but keep abbreviations mostly intact.
        pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])", line)
        units.extend(piece.strip() for piece in pieces if piece and piece.strip())
    return units


def _build_semantic_chunks(
    text: str,
    embeddings_model: EmbeddingsLike,
    similarity_threshold: float,
    min_chunk_chars: int,
    max_chunk_chars: int,
) -> list[str]:
    """Group adjacent sentence units by semantic continuity."""
    sentence_units = _split_into_sentences(text)
    if len(sentence_units) <= 1:
        return [text.strip()] if text.strip() else []

    vectors = embeddings_model.embed_documents(sentence_units)
    semantic_chunks: list[str] = []

    current_units = [sentence_units[0]]
    current_chars = len(sentence_units[0])

    for idx in range(1, len(sentence_units)):
        candidate = sentence_units[idx]
        similarity = _cosine_similarity(vectors[idx - 1], vectors[idx])

        should_split = False
        if current_chars >= max_chunk_chars:
            should_split = True
        elif current_chars >= min_chunk_chars and similarity < similarity_threshold:
            should_split = True

        if should_split:
            semantic_chunks.append(" ".join(current_units).strip())
            current_units = [candidate]
            current_chars = len(candidate)
        else:
            current_units.append(candidate)
            current_chars += len(candidate)

    if current_units:
        semantic_chunks.append(" ".join(current_units).strip())

    return [chunk for chunk in semantic_chunks if chunk]


def _build_recursive_chunks(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Create first-pass recursive chunks optimized for technical documents."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=TECHNICAL_SEPARATORS,
        add_start_index=True,
        strip_whitespace=True,
    )
    return splitter.split_documents(documents)


def _get_semantic_model() -> OpenAIEmbeddings | None:
    """Create semantic embeddings model if API key is available."""
    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        return OpenAIEmbeddings(model="text-embedding-3-small")
    except Exception:
        return None


def _semantic_refine_recursive_chunks(
    recursive_chunks: list[Document],
    embeddings_model: EmbeddingsLike,
    similarity_threshold: float,
    min_chunk_chars: int,
    max_chunk_chars: int,
) -> list[Document]:
    """Run semantic segmentation inside each recursive chunk."""
    refined_docs: list[Document] = []
    for parent_idx, chunk in enumerate(recursive_chunks):
        semantic_chunks = _build_semantic_chunks(
            text=chunk.page_content,
            embeddings_model=embeddings_model,
            similarity_threshold=similarity_threshold,
            min_chunk_chars=min_chunk_chars,
            max_chunk_chars=max_chunk_chars,
        )

        if not semantic_chunks:
            continue

        for semantic_idx, semantic_text in enumerate(semantic_chunks):
            metadata = dict(chunk.metadata)
            metadata["parent_chunk_index"] = parent_idx
            metadata["semantic_chunk_in_parent"] = semantic_idx
            metadata["chunking_strategy"] = "hybrid"
            refined_docs.append(Document(page_content=semantic_text, metadata=metadata))

    return refined_docs
    

### Read all the pdf's inside the directory
def process_all_pdfs(pdf_directory):
    """Process all PDF files in a directory using PyMuPDF (faster than pypdf)"""
    all_documents = []
    pdf_dir = Path(pdf_directory)
    
    # Find all PDF files recursively
    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")
        try:
            # Use PyMuPDF for faster PDF loading
            doc = fitz.open(str(pdf_file))
            documents = []
            
            for page_num, page in enumerate(doc):
                text = page.get_text()
                
                metadata = {
                    'source': pdf_file.name,
                    'page': page_num + 1,
                    'file_type': 'pdf'
                }
                
                documents.append(Document(
                    page_content=text,
                    metadata=metadata
                ))
            
            all_documents.extend(documents)
            print(f"  ✓ Loaded {len(documents)} pages")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print(f"\nTotal documents loaded: {len(all_documents)}")
    return all_documents

## Chunking the documents into smaller pieces
def document_splitter(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    strategy: str = DEFAULT_CHUNK_STRATEGY,
    semantic_similarity_threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
    semantic_min_chunk_chars: int = DEFAULT_SEMANTIC_MIN_CHARS,
    semantic_model: EmbeddingsLike | None = None,
) -> list[Document]:
    """Split loaded documents using recursive or hybrid recursive+semantic chunking."""
    if not documents:
        print("No documents to split.")
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    if strategy not in {"recursive", "hybrid"}:
        raise ValueError("strategy must be either 'recursive' or 'hybrid'")
    if not 0.0 <= semantic_similarity_threshold <= 1.0:
        raise ValueError("semantic_similarity_threshold must be between 0 and 1")
    if semantic_min_chunk_chars <= 0:
        raise ValueError("semantic_min_chunk_chars must be greater than 0")
    if semantic_min_chunk_chars >= chunk_size:
        raise ValueError("semantic_min_chunk_chars must be smaller than chunk_size")

    print(f"Loaded {len(documents)} documents from PDF.")
    print(f"Chunking strategy: {strategy}")

    recursive_chunks = _build_recursive_chunks(
        documents=documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    split_docs = recursive_chunks
    if strategy == "hybrid":
        model = semantic_model or _get_semantic_model()
        if model is None:
            print("Semantic model unavailable. Falling back to recursive chunking.")
        else:
            try:
                split_docs = _semantic_refine_recursive_chunks(
                    recursive_chunks=recursive_chunks,
                    embeddings_model=model,
                    similarity_threshold=semantic_similarity_threshold,
                    min_chunk_chars=semantic_min_chunk_chars,
                    max_chunk_chars=chunk_size,
                )
            except Exception as e:
                print(f"Semantic chunking failed ({e}). Falling back to recursive chunking.")
                split_docs = recursive_chunks

    for chunk_idx, chunk in enumerate(split_docs):
        chunk.metadata["chunk_index"] = chunk_idx
        chunk.metadata.setdefault("chunking_strategy", strategy)

    print(f"Split into {len(split_docs)} chunks")
    print(f"Chunk size: {chunk_size}")
    print(f"Chunk overlap: {chunk_overlap}")
    print(f"\nFirst chunk metadata: {split_docs[0].metadata}")
    print(f"First chunk content: {split_docs[0].page_content[:200]}...")
    print(f"\nLast chunk metadata: {split_docs[-1].metadata}")

    return split_docs
