#!/usr/bin/env python3
"""
Simple HTTP server to serve the RAG Pipeline Chat UI locally.
Usage: python3 serve.py
Then open http://localhost:8000 in your browser.
"""

import http.server
import socketserver
import os
import webbrowser
from pathlib import Path

PORT = 8000
HANDLER = http.server.SimpleHTTPRequestHandler

def start_server():
    """Start the local HTTP server."""
    # Change to the UI directory
    ui_dir = Path(__file__).parent
    os.chdir(ui_dir)
    
    with socketserver.TCPServer(("", PORT), HANDLER) as httpd:
        print(f"🚀 RAG Pipeline Chat UI Server")
        print(f"=" * 50)
        print(f"📍 Open this URL in your browser:")
        print(f"   http://localhost:{PORT}")
        print(f"")
        print(f"✋ Press Ctrl+C to stop the server")
        print(f"=" * 50)
        print()
        
        # Try to open browser automatically
        try:
            webbrowser.open(f"http://localhost:{PORT}")
        except:
            pass
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n✌️  Server stopped.")

if __name__ == "__main__":
    start_server()
