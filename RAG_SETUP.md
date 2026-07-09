# RAG Setup Guide

This guide explains how to populate your Supabase database with documents for RAG (Retrieval-Augmented Generation) functionality.

## Prerequisites

Before setting up RAG, ensure you have:

1. **OPENAI_API_KEY** - for document embeddings (set in Render environment)
2. **SUPABASE_DATABASE_URL** - PostgreSQL connection string with pgvector (set in Render environment)
3. **PDF documents** - place them in the `./data` directory

## Quick Start

### 1. Prepare Your Documents

Place your PDF files in the `./data` directory:

```bash
mkdir -p data
# Copy your PDF files to ./data/
ls data/*.pdf  # Verify files are there
```

### 2. Run the Setup Script

This will initialize Supabase and populate it with embeddings:

**Locally:**
```bash
uv run python scripts/setup_rag.py ./data
```

**On Render (via SSH or deployment hook):**
Add this to your `render.yml` or run manually via SSH:
```bash
uv run python scripts/setup_rag.py ./data
```

### 3. What the Script Does

1. ✅ Validates `OPENAI_API_KEY` and `SUPABASE_DATABASE_URL`
2. ✅ Creates pgvector extension and `rag_docs` table
3. ✅ Loads PDFs from `./data`
4. ✅ Splits documents into chunks (500 chars, 50 char overlap)
5. ✅ Embeds chunks with `text-embedding-3-small`
6. ✅ Stores embeddings in Supabase with vector index
7. ✅ Tests search with "machine learning" query

### 4. Verify It Works

After setup completes, test the `/chat` endpoint:

```bash
curl -X POST https://your-domain.onrender.com/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is machine learning?",
    "thread_id": "user-123"
  }'
```

You should get a response with RAG context, not just LLM knowledge.

## Troubleshooting

### No documents retrieved

Check the logs:
```bash
# Look for: "RAG: Retrieved 0 documents"
# Solution: Run setup_rag.py again to populate Supabase
```

### Embeddings API errors

```bash
# Error: "OPENAI_API_KEY not set"
# Solution: Set OPENAI_API_KEY in Render environment variables
```

### Supabase connection errors

```bash
# Error: "SUPABASE_DATABASE_URL not set"
# Solution: Ensure SUPABASE_DATABASE_URL is set in Render environment
```

### PDFs not loading

```bash
# Error: "No PDF files found in ./data"
# Solution: 
# 1. Create ./data directory
# 2. Add PDF files to it
# 3. Re-run setup script
```

## Files Modified

- **app/document_loader.py** - PDF loading and chunking logic
- **app/embedding.py** - OpenAI embedding functions
- **app/rag_supabase.py** - Supabase vector search and storage
- **scripts/setup_rag.py** - Setup script to populate database
- **app/rag_agent.py** - Added RAG logging for debugging
- **app/main.py** - Agent now warms up in background

## Document Processing Details

- **Chunk Size**: 500 characters
- **Chunk Overlap**: 50 characters (for context continuity)
- **Embedding Model**: `text-embedding-3-small` (1536 dimensions)
- **Vector Index**: IVFFlat (inverted file) for fast similarity search
- **Hybrid Search**: 60% vector + 40% BM25 keyword (if available)

## Cost Considerations

- OpenAI embeddings: ~$0.02 per 1M tokens
- Supabase: Free tier includes 1GB storage, 200K DB writes/month
- For 1000 documents (avg 1000 tokens each): ~$0.02 embedding cost

## Next Steps

1. Place your PDFs in `./data`
2. Run the setup script
3. Monitor logs to verify documents are being retrieved
4. Adjust chunk size/overlap if needed for your use case
