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
# Local frontend origins configured via environment variable; no unrestricted wildcard in production.
env_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,http://127.0.0.1:3000")
allowed_origins = [o.strip() for o in env_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
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

# Convenience root /health alias
@app.get("/health", include_in_schema=False)
def root_health():
    return health.get_health()
