# RAG Setup - Simple

## How It Works

1. **On Startup**: App automatically loads PDFs from `./data`, embeds them, stores in Supabase
2. **On Chat**: Retrieve relevant documents from Supabase, inject into LLM prompt

## Setup

### Step 1: Add Your PDFs

```bash
mkdir -p data
# Copy your PDFs to ./data/
ls data/*.pdf
```

### Step 2: Add Environment Variables

In Render (or .env locally):
```
OPENAI_API_KEY=sk-...
SUPABASE_DATABASE_URL=postgresql://...
```

### Step 3: Start the App

```bash
uv run uvicorn app.main:app --reload
```

App will automatically:
- ✅ Load PDFs from `./data`
- ✅ Embed chunks with OpenAI
- ✅ Store embeddings in Supabase
- ✅ Initialize RAG retriever

### Step 4: Test It

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is the main topic in your documents?",
    "thread_id": "user-123"
  }'
```

**Check logs for**: `RAG: Retrieved X documents` → This means it's working!

## That's It

No setup scripts. No manual steps. Just:
1. Add PDFs
2. Set env vars
3. Start app
4. Chat

The app handles the rest automatically.

## Troubleshooting

### No documents retrieved
- Check logs for: "RAG: Retrieved 0 documents"
- Verify PDFs are in `./data`
- Check `OPENAI_API_KEY` is valid
- Check `SUPABASE_DATABASE_URL` is set correctly

### Build fails locally
```bash
uv run pip install -e .
uv run pytest tests/
```
