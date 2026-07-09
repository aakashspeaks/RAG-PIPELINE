"""
Production-Ready FastAPI + LangGraph Application

Wires together:
- Security pipeline (input sanitization, PII masking)
- Response caching
- Rate limiting (slowapi)
- LangGraph agent (with retries + fallback)
- Structured logging + metrics
- LangSmith tracing
- Health checks
"""

import time
import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from langsmith import traceable
from dotenv import load_dotenv

from app.config import get_settings
from app.models import (
    ChatRequest, ChatResponse,
    HealthResponse, MetricsResponse, ErrorResponse,
)
from app.security import SecurityPipeline
from app.cache import ResponseCache
from app.monitoring import get_logger, MetricsCollector, RequestTimer

metrics: MetricsCollector = None
security: SecurityPipeline = None
cache: ResponseCache = None
agent = None  # Lazy-loaded, type is RAGAgent but imported on demand
logger = get_logger()


def get_agent():
    """Return the global RAGAgent, initializing it if needed."""
    global agent
    if agent is None:
        logger.info("Initializing RAGAgent...")
        from app.rag_agent import RAGAgent
        agent = RAGAgent()
        logger.info("RAGAgent ready!")
    return agent


def _populate_rag_documents():
    """Load and store documents from ./data into Supabase (one time)."""
    try:
        from pathlib import Path
        data_dir = Path("./data")
        
        if not data_dir.exists() or not list(data_dir.glob("**/*.pdf")):
            logger.info("No PDFs found in ./data - skipping RAG population")
            return
        
        logger.info("📚 Populating RAG documents from ./data...")
        from app.rag_supabase import store_to_supabase
        store_to_supabase("./data")
        logger.info("✅ RAG documents stored in Supabase")
    except Exception as e:
        logger.warning(f"RAG population failed: {e}")


def _warmup_agent():
    """Initialize RAGAgent in a background thread at startup."""
    try:
        get_agent()
    except Exception as e:
        logger.warning(f"Agent warmup failed (will retry on first request): {e}")


# === Lifespan (startup/shutdown) ===

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize lightweight components at startup.
    RAGAgent (heavy) will lazy-load on first request.
    """
    global security, cache, metrics
    
    settings = get_settings()

    logger.info("🚀 API starting...", extra={"extra_data": {
        "environment": settings.app_env,
        "primary_model": settings.primary_model,
    }})

    # Initialize lightweight components only
    security = SecurityPipeline()
    cache = ResponseCache(ttl_seconds=settings.cache_ttl_seconds)
    metrics = MetricsCollector()

    # Auto-populate RAG from ./data on startup
    _populate_rag_documents()

    # Warm up agent in background — first /chat won't block on initialization
    threading.Thread(target=_warmup_agent, daemon=True).start()
    logger.info("✅ Core components ready. RAGAgent warming up in background.")
    
    yield
    
    logger.info("🛑 API shutting down", extra={"extra_data": metrics.summary if metrics else {}})


# === Rate Limiter Setup ===
limiter = Limiter(key_func=get_remote_address)

# === FastAPI App ===
app = FastAPI(
    title="Production LangGraph API",
    description="A production-ready chat API with security, caching, and observability.",
    version="1.0.0",
    lifespan=lifespan,
)
app.state.limiter = limiter


# === Exception Handlers ===

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """Handle rate limit exceeded errors."""
    logger.warning("Rate limit exceeded", extra={"extra_data": {
        "client_ip": get_remote_address(request),
    }})
    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate limit exceeded",
            "detail": "Too many requests. Please slow down.",
        },
    )
    

# =============================================
# ENDPOINTS
# =============================================

@app.post("/chat", response_model=ChatResponse)
@limiter.limit(get_settings().rate_limit)
@traceable(name="chat_endpoint")
async def chat(request: Request, body: ChatRequest):
    """
    Main chat endpoint.

    Flow:
    1. Security check (injection + PII masking)
    2. Cache lookup
    3. LangGraph agent invoke (if cache miss)
    4. Output validation
    5. Cache store
    6. Return response
    """
    with RequestTimer() as timer:
        security_notes = []

        # ---- Step 1: Security Check ----
        is_allowed, cleaned_message, notes = security.check_input(body.message)
        security_notes.extend(notes)

        if not is_allowed:
            logger.warning("Request blocked by security", extra={"extra_data": {
                "reason": notes,
                "thread_id": body.thread_id,
            }})
            metrics.record_request(latency_ms=0, error=True)
            raise HTTPException(
                status_code=400,
                detail="Your message was blocked by our security filters."
            )

        # ---- Step 2: Cache Lookup ----
        cached_response = cache.get(cleaned_message)
        if cached_response is not None:
            metrics.record_request(latency_ms=0, cache_hit=True)
            logger.info("Cache hit", extra={"extra_data": {
                "thread_id": body.thread_id,
            }})
            return ChatResponse(
                response=cached_response,
                thread_id=body.thread_id,
                model_used="cache",
                rag_mode=False,  # Cached responses don't have RAG info
                sources=[],
                cached=True,
                processing_time_ms=0,
            )

        # ---- Step 3: Invoke LangGraph Agent (lazy-loaded) ----
        try:
            rag_agent = get_agent()  # Lazy-load on first request
            result = rag_agent.invoke(cleaned_message)
        except Exception as e:
            logger.error(f"Agent invocation failed: {e}", extra={"extra_data": {
                "thread_id": body.thread_id,
                "error": str(e),
            }})
            metrics.record_request(latency_ms=0, error=True)
            raise HTTPException(
                status_code=500,
                detail="An error occurred while processing your request."
            )

        response_text = result["response"]
        model_used = result["model_used"]
        rag_mode = result.get("rag_mode", False)
        sources = result.get("sources", [])

        # ---- Step 4: Output Validation ----
        validated_response, output_warnings = security.check_output(response_text)
        security_notes.extend(output_warnings)

        # ---- Step 5: Cache Store ----
        cache.set(cleaned_message, validated_response)

    # ---- Step 6: Log & Record Metrics ----
    input_tokens = int(len(cleaned_message.split()) * 1.3)
    output_tokens = int(len(validated_response.split()) * 1.3)

    metrics.record_request(
        latency_ms=timer.elapsed_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_hit=False,
    )

    if security_notes:
        logger.info("Security notes", extra={"extra_data": {
            "notes": security_notes,
            "thread_id": body.thread_id,
        }})

    logger.info("Request completed", extra={"extra_data": {
        "thread_id": body.thread_id,
        "model_used": model_used,
        "latency_ms": round(timer.elapsed_ms, 2),
    }})

    return ChatResponse(
        response=validated_response,
        thread_id=body.thread_id,
        model_used=model_used,
        rag_mode=rag_mode,
        sources=sources,
        cached=False,
        processing_time_ms=round(timer.elapsed_ms, 2),
        security_notes=security_notes,
    )
    
    
    
    
@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check for Docker/Kubernetes."""
    settings = get_settings()

    checks = {
        "agent": agent is not None,  # lazy-loaded; false until first request
        "security": security is not None,
        "cache": cache is not None,
    }

    # Agent is lazy-loaded by design — don't penalise health status for it
    core_healthy = checks["security"] and checks["cache"]

    return HealthResponse(
        status="healthy" if core_healthy else "degraded",
        environment=settings.app_env,
        checks=checks,
    )


@app.get("/metrics", response_model=MetricsResponse)
async def metrics_endpoint():
    """Metrics for monitoring dashboards."""
    summary = metrics.summary
    return MetricsResponse(**summary)


@app.get("/cache/stats")
async def cache_stats():
    """Cache performance statistics."""
    return cache.stats