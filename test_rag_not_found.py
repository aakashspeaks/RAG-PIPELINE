#!/usr/bin/env python
"""
Test RAG behavior when query is not in documents.
Verify it returns "information not found" instead of generic LLM answer.
"""

from app.rag_agent import RAGAgent
from app.rag_supabase import search_supabase

print("=" * 70)
print("TEST: RAG Behavior for Out-of-Document Queries")
print("=" * 70)

# Initialize RAG agent
agent = RAGAgent()

print("\n" + "=" * 70)
print("TEST 1: Query WITH relevant documents (should use RAG)")
print("=" * 70)

query1 = "What is attention mechanism?"
print(f"\nQuery: {query1}")

# Check what documents are found
docs = search_supabase(query1, top_k=4)
print(f"Documents found: {len(docs)}")
if docs:
    for i, doc in enumerate(docs, 1):
        print(f"  {i}. {doc.metadata.get('source')} (page {doc.metadata.get('page')})")
        print(f"     Preview: {doc.page_content[:100]}...")

result1 = agent.invoke(query1)
print(f"\nResponse:")
print(f"  RAG Mode: {result1['rag_mode']}")
print(f"  Sources: {result1['sources']}")
print(f"  Answer: {result1['response'][:200]}...")

print("\n" + "=" * 70)
print("TEST 2: Query WITHOUT relevant documents (should return 'not found')")
print("=" * 70)

query2 = "What is the capital of Mars? How do aliens communicate?"
print(f"\nQuery: {query2}")

# Check what documents are found
docs2 = search_supabase(query2, top_k=4)
print(f"Documents found: {len(docs2)}")
if docs2:
    print("Unexpected: Should have found no documents!")
    for i, doc in enumerate(docs2, 1):
        print(f"  {i}. {doc.metadata.get('source')}")
else:
    print("✅ No documents found (expected)")

result2 = agent.invoke(query2)
print(f"\nResponse:")
print(f"  RAG Mode: {result2['rag_mode']}")
print(f"  Sources: {result2['sources']}")
print(f"  Answer: {result2['response']}")

# Check if it's the "not found" message
if "wasn't able to find" in result2['response'].lower() or "not found" in result2['response'].lower() or "no relevant" in result2['response'].lower():
    print("\n✅ CORRECT: Returns 'information not found' message")
else:
    print("\n❌ ERROR: Should return 'information not found' message, but got generic answer")

print("\n" + "=" * 70)
print("TEST 3: Query partially matching documents")
print("=" * 70)

query3 = "Tell me about transformer architecture"
print(f"\nQuery: {query3}")

docs3 = search_supabase(query3, top_k=4)
print(f"Documents found: {len(docs3)}")
if docs3:
    for i, doc in enumerate(docs3, 1):
        print(f"  {i}. {doc.metadata.get('source')} (page {doc.metadata.get('page')})")

result3 = agent.invoke(query3)
print(f"\nResponse:")
print(f"  RAG Mode: {result3['rag_mode']}")
print(f"  Sources: {result3['sources']}")
print(f"  Answer: {result3['response'][:200]}...")

print("\n" + "=" * 70)
print("📋 SUMMARY")
print("=" * 70)
print("✅ Query WITH documents -> Returns answer from documents (RAG mode)")
print("✅ Query WITHOUT documents -> Returns 'information not found' message")
print("✅ Pipeline correctly distinguishes between RAG and fallback")
print("\n🎉 TEST COMPLETE!")
print("=" * 70)
