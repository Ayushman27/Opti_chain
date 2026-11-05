"""
OptiChain AI - FastAPI Application
Exposes validated supply chain simulation, forecasting, benchmarks,
and MILP optimization endpoints under /api/v1.
"""

import os
import logging
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from optichain.api.routes import (
    health,
    dashboard,
    benchmarks,
    forecasts,
    inventory,
    optimization,
    simulation,
)
from optichain.agent.reasoning_agent import SupplyChainReasoningAgent
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional

class ChatRequest(BaseModel):
    prompt: str = Field(..., description="User query or instruction")
    context_kpis: Optional[Dict[str, Any]] = None

_reasoning_agent = SupplyChainReasoningAgent()

logger = logging.getLogger("optichain.api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(
    title="OptiChain AI REST API",
    description="Autonomous Supply Chain Digital Twin & Optimization Platform API (Phase 5)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# -------------------------------------------------------------------------
# CORS Configuration
# -------------------------------------------------------------------------
# Configurable via environment variables; automatically permits *.vercel.app for Vercel preview & production deployments
env_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,http://127.0.0.1:3000")
allowed_origins = [o.strip() for o in env_origins.split(",") if o.strip()]

# Regex allowing any vercel.app domain (including preview branches) and localhost ports
allow_origin_regex = os.getenv(
    "ALLOW_ORIGIN_REGEX",
    r"https://.*\.vercel\.app|http://localhost(:\d+)?|http://127\.0\.0\.1(:\d+)?"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=allow_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------------------------------
# Global Exception Handling
# -------------------------------------------------------------------------
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catches unhandled server exceptions, logs full stack trace internally,
    and returns a clean, sanitized JSON error response to client.
    """
    logger.exception("Unhandled server exception at %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred while processing the request. Contact system administrator.",
            "path": request.url.path,
        },
    )


# -------------------------------------------------------------------------
# Mount /api/v1 Router Hierarchy
# -------------------------------------------------------------------------
app.include_router(health.router, prefix="/api/v1")
app.include_router(dashboard.router, prefix="/api/v1")
app.include_router(benchmarks.router, prefix="/api/v1")
app.include_router(forecasts.router, prefix="/api/v1")
app.include_router(inventory.router, prefix="/api/v1")
app.include_router(optimization.router, prefix="/api/v1")
app.include_router(simulation.router, prefix="/api/v1")

# AI Copilot Reasoning Agent routes
@app.post("/api/agent/chat")
@app.post("/api/v1/agent/chat")
def chat_with_agent(req: ChatRequest):
    return _reasoning_agent.generate_chat_response(
        prompt=req.prompt,
        kpis=req.context_kpis or {},
    )

# Convenience root /health alias
@app.get("/health", include_in_schema=False)
def root_health():
    return health.get_health()
