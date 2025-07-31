"""
API v1 router for GreenCast application
"""

from fastapi import APIRouter

from .endpoints import (
    auth,
    users,
    fields,
    disease_detection,
    yield_prediction,
    alerts,
    dashboard,
    field_logs
)

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(fields.router, prefix="/fields", tags=["fields"])
api_router.include_router(disease_detection.router, prefix="/disease-detection", tags=["disease-detection"])
api_router.include_router(yield_prediction.router, prefix="/yield-prediction", tags=["yield-prediction"])
api_router.include_router(alerts.router, prefix="/alerts", tags=["alerts"])
api_router.include_router(dashboard.router, prefix="/dashboard", tags=["dashboard"])
api_router.include_router(field_logs.router, prefix="/field-logs", tags=["field-logs"])
