"""
Health check route
GET /api/v1/health
"""

from datetime import datetime, timezone
from fastapi import APIRouter
from optichain.api.schemas import HealthResponse
from optichain.api.dependencies import (
    check_cbc_solver,
    get_optimization_benchmark_path,
    get_forecast_benchmark_path,
)

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """
    Returns service health status, API version, and verifies actual dependent components
    (CBC MILP solver availability and benchmark artifacts presence).
    """
    cbc_ok = check_cbc_solver()
    benchmark_ok = (
        get_optimization_benchmark_path().exists()
        and get_forecast_benchmark_path().exists()
    )

    # Honest reporting: degrade status if dependent components are missing
    if cbc_ok and benchmark_ok:
        status = "healthy"
    elif benchmark_ok or cbc_ok:
        status = "degraded"
    else:
        status = "unhealthy"

    return HealthResponse(
        status=status,
        version="1.0.0",
        service="OptiChain AI Backend",
        cbc_solver_available=cbc_ok,
        benchmark_artifact_present=benchmark_ok,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
