from __future__ import annotations

from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
    

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
def document_splitter(documents: list[Document]) -> list[Document]:
    """Split loaded documents into chunks."""
    if not documents:
        print("No documents to split.")
        return []

    print(f"Loaded {len(documents)} documents from PDF.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    # split the docs
    split_docs = splitter.split_documents(documents)

    print(f"Split into {len(split_docs)} chunks")
    print(f"\nFirst chunk metadata: {split_docs[0].metadata}")
    print(f"First chunk content: {split_docs[0].page_content[:200]}...")
    print(f"\nLast chunk metadata: {split_docs[-1].metadata}")

    return split_docs
