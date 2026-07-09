# RAG Pipeline Chat UI

A modern, responsive chat interface for the RAG Pipeline API.

## Features

✨ **Clean Chat Interface** - Modern chat UI with real-time messaging
📊 **Response Metadata** - View processing time, model, cache status, and RAG mode
📚 **Source Attribution** - See which documents were used to generate responses
⚡ **Fast & Responsive** - Smooth animations and instant feedback
📱 **Mobile Friendly** - Works great on tablets and phones

## Quick Start

### Option 1: Open Directly in Browser
Simply open `index.html` in your web browser:
```bash
open index.html
# or on Linux:
xdg-open index.html
```

### Option 2: Local HTTP Server (Recommended)
To avoid CORS issues, serve it locally:

```bash
# Using Python (3.x)
python3 -m http.server 8000

# Using Node.js
npx http-server

# Using Python (2.x)
python -m SimpleHTTPServer 8000
```

Then open: **http://localhost:8000**

### Option 3: Deploy to Netlify/Vercel
Since this is a static HTML file, you can deploy it anywhere:
- Drag & drop to Netlify
- Push to GitHub and connect to Vercel
- Upload to any static hosting service

## How to Use

1. **Type a Question** - Enter your question in the input field
2. **Press Send** - Click the button or press Enter
3. **View Response** - The AI responds with sources and metadata
4. **Review Metadata** - Check processing time, model used, etc.
5. **See Sources** - View the documents used to generate the response

## Example Questions

Try these questions with the provided sample data:

- "What are the applications of GPTs in tourism?"
- "How do GPTs help with pest control in agriculture?"
- "What is the attention mechanism?"
- "What are the benefits of GPTs in agriculture?"
- "What challenges are mentioned in using GPTs?"

## API Connection

The UI connects to: `https://rag-pipeline-4t6i.onrender.com`

To use with a different API endpoint, edit the `API_URL` variable in the JavaScript:

```javascript
const API_URL = 'https://your-api-url.com';
```

## Metadata Explained

| Field | Meaning |
|-------|---------|
| **Status** | ✅ RAG (used documents) or ⚠️ LLM (no documents found) |
| **Time (ms)** | How long the response took to generate |
| **Model** | Which LLM model was used (gpt-4o-mini, etc.) |
| **Cached** | Whether the response was served from cache |
| **RAG Mode** | Whether the RAG pipeline was activated |
| **Thread ID** | Unique conversation thread identifier |

## Browser Support

Works on all modern browsers:
- Chrome/Chromium
- Firefox
- Safari
- Edge
- Mobile browsers

## Technical Details

- **No Backend Required** - Pure frontend (HTML/CSS/JS)
- **No Dependencies** - Vanilla JavaScript, no frameworks
- **CORS Compatible** - Works with CORS-enabled APIs
- **Production Ready** - Can be deployed as-is

## Troubleshooting

### "CORS error"
- Run a local HTTP server (see Option 2 above)
- Or ensure your API has CORS headers enabled

### "API Connection Failed"
- Verify the API URL is correct
- Check that the API is running and accessible
- Look at browser console (F12) for error details

### "Rate limit exceeded"
- Wait a moment between requests
- Check your API rate limit settings

## License

MIT - Feel free to use and modify!
