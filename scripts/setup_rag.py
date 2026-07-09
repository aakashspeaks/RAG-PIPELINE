#!/usr/bin/env python3
"""
Setup script for populating Supabase with RAG documents.

Usage:
    uv run python scripts/setup_rag.py ./data

This script will:
1. Check environment variables (OPENAI_API_KEY, SUPABASE_DATABASE_URL)
2. Initialize Supabase table with pgvector extension
3. Load PDFs from ./data directory
4. Embed documents with OpenAI embeddings
5. Store them in Supabase rag_docs table
6. Test the search functionality
"""

import sys
import os
from pathlib import Path

# Add parent directory to path so we can import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.rag_supabase import store_to_supabase, search_supabase


def main():
    print("=" * 60)
    print("🚀 RAG Supabase Setup")
    print("=" * 60)
    
    # Check environment variables
    print("\n📋 Checking environment variables...")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set")
        return False
    print("✓ OPENAI_API_KEY set")
    
    if not os.getenv("SUPABASE_DATABASE_URL"):
        print("❌ SUPABASE_DATABASE_URL not set")
        return False
    print("✓ SUPABASE_DATABASE_URL set")
    
    # Get data directory from CLI or use default
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./data"
    
    if not Path(data_dir).exists():
        print(f"\n⚠️  Warning: {data_dir} directory not found")
        print(f"📁 Creating {data_dir}...")
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        print(f"   Please add PDF files to {data_dir} and run this script again")
        return False
    
    pdf_count = len(list(Path(data_dir).glob("**/*.pdf")))
    if pdf_count == 0:
        print(f"\n⚠️  No PDF files found in {data_dir}")
        print("   Please add PDFs to the data directory and try again")
        return False
    
    print(f"✓ Found {pdf_count} PDF file(s) in {data_dir}")
    
    # Initialize and populate Supabase
    print("\n" + "=" * 60)
    print("📊 Populating Supabase with documents...")
    print("=" * 60)
    
    try:
        store_to_supabase(data_dir)
    except Exception as e:
        print(f"\n❌ Error during setup: {e}")
        return False
    
    # Test search
    print("\n" + "=" * 60)
    print("🧪 Testing RAG search...")
    print("=" * 60)
    
    try:
        results = search_supabase("machine learning", top_k=2)
        
        if not results:
            print("⚠️  No results found (Supabase might be empty)")
            return False
        
        print(f"\n✅ Search works! Found {len(results)} result(s)")
        
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            similarity = doc.metadata.get("similarity", 0)
            print(f"\n[Result {i}]")
            print(f"  Source: {source} (page {page})")
            print(f"  Similarity: {similarity:.3f}")
            print(f"  Content: {doc.page_content[:100]}...")
        
    except Exception as e:
        print(f"\n❌ Error during search test: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ RAG setup complete!")
    print("=" * 60)
    print("\nYour API is ready for RAG queries:")
    print("  POST /chat")
    print("  {\"message\": \"Your question here\", \"thread_id\": \"user-123\"}")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
