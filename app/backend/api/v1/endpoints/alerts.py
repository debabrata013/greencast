"""
Alerts API endpoints
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from pydantic import BaseModel, Field
from loguru import logger

# Local imports
from backend.models.database import (
    Alert, User, Field, CropType, AlertSeverity, 
    WeatherData, Location
)
from backend.services.ml_service import alert_service
from backend.utils.auth import get_current_user
from backend.utils.weather_service import get_weather_forecast

router = APIRouter()


# Pydantic models for request/response
class AlertRequest(BaseModel):
    """Alert generation request model"""
    field_id: Optional[str] = Field(None, description="Field ID for location-specific alerts")
    crop_type: CropType = Field(..., description="Type of crop")
    latitude: float = Field(..., ge=-90, le=90, description="Latitude coordinate")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude coordinate")
    forecast_days: int = Field(default=7, ge=1, le=14, description="Number of forecast days")


class AlertResponse(BaseModel):
    """Alert response model"""
    id: str
    alert_type: str
    severity: AlertSeverity
    title: str
    message: str
    recommendations: List[str]
    is_forecast: bool
    forecast_date: Optional[datetime]
    location: Optional[Dict[str, float]]
    created_at: datetime
    expires_at: Optional[datetime]
    is_read: bool


class AlertSummaryResponse(BaseModel):
    """Alert summary response model"""
    total_alerts: int
    unread_alerts: int
    critical_alerts: int
    high_alerts: int
    medium_alerts: int
    low_alerts: int
    recent_alerts: int  # Last 24 hours


class WeatherAlertRequest(BaseModel):
    """Weather-based alert request model"""
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    crop_type: CropType
    field_id: Optional[str] = None


@router.post("/generate", response_model=List[AlertResponse])
async def generate_alerts(
    request: AlertRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """
    Generate alerts based on weather conditions and crop type
    
    This endpoint generates current and forecast alerts for the specified location.
    """
    try:
        # Validate field if provided
        field = None
        if request.field_id:
            field = await Field.get(request.field_id)
            if not field or field.user_id != str(current_user.id):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied to this field"
                )
        
        # Get current weather and forecast
        weather_data = await get_weather_forecast(request.latitude, request.longitude, request.forecast_days)
        
        if not weather_data:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Weather service unavailable"
            )
        
        current_weather = weather_data["current"]
        forecast = weather_data["forecast"]
        
        # Create location object
        location = Location(
            latitude=request.latitude,
            longitude=request.longitude
        )
        
        # Generate current alerts
        current_weather_obj = WeatherData(**current_weather)
        current_alerts = await alert_service.generate_alerts(
            weather_data=current_weather_obj,
            crop_type=request.crop_type
        )
        
        # Generate forecast alerts
        forecast_alerts = []
        for day_forecast in forecast[:request.forecast_days]:
            forecast_weather = WeatherData(**day_forecast["weather"])
            day_alerts = await alert_service.generate_alerts(
                weather_data=forecast_weather,
                crop_type=request.crop_type
            )
            
            # Mark as forecast alerts
            for alert in day_alerts:
                alert["is_forecast"] = True
                alert["forecast_date"] = datetime.fromisoformat(day_forecast["date"])
            
            forecast_alerts.extend(day_alerts)
        
        # Combine all alerts
        all_alerts = current_alerts + forecast_alerts
        
        # Save alerts to database
        saved_alerts = []
        for alert_data in all_alerts:
            # Set expiration time
            expires_at = datetime.utcnow() + timedelta(hours=24)
            if alert_data.get("is_forecast"):
                expires_at = alert_data["forecast_date"] + timedelta(hours=12)
            
            alert = Alert(
                user_id=str(current_user.id),
                field_id=request.field_id,
                alert_type=alert_data["alert_type"],
                severity=AlertSeverity(alert_data["severity"]),
                title=alert_data["title"],
                message=alert_data["message"],
                recommendations=alert_data["recommendations"],
                weather_data=current_weather_obj if not alert_data.get("is_forecast") else forecast_weather,
                location=location,
                is_forecast=alert_data.get("is_forecast", False),
                forecast_date=alert_data.get("forecast_date"),
                expires_at=expires_at
            )
            
            await alert.insert()
            saved_alerts.append(alert)
        
        # Schedule background cleanup of old alerts
        background_tasks.add_task(cleanup_expired_alerts, str(current_user.id))
        
        logger.info(f"Generated {len(saved_alerts)} alerts for user {current_user.email}")
        
        # Convert to response format
        alert_responses = []
        for alert in saved_alerts:
            alert_responses.append(AlertResponse(
                id=str(alert.id),
                alert_type=alert.alert_type,
                severity=alert.severity,
                title=alert.title,
                message=alert.message,
                recommendations=alert.recommendations,
                is_forecast=alert.is_forecast,
                forecast_date=alert.forecast_date,
                location={"latitude": location.latitude, "longitude": location.longitude},
                created_at=alert.created_at,
                expires_at=alert.expires_at,
                is_read=alert.is_read
            ))
        
        return alert_responses
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Alert generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Alert generation failed"
        )


@router.get("/", response_model=List[AlertResponse])
async def get_alerts(
    unread_only: bool = False,
    severity: Optional[AlertSeverity] = None,
    alert_type: Optional[str] = None,
    field_id: Optional[str] = None,
    page: int = 1,
    page_size: int = 50,
    current_user: User = Depends(get_current_user)
):
    """
    Get alerts for the current user
    
    Returns paginated list of alerts with filtering options.
    """
    try:
        # Build query
        query = Alert.user_id == str(current_user.id)
        
        if unread_only:
            query = query & (Alert.is_read == False)
        
        if severity:
            query = query & (Alert.severity == severity)
        
        if alert_type:
            query = query & (Alert.alert_type == alert_type)
        
        if field_id:
            query = query & (Alert.field_id == field_id)
        
        # Filter out expired alerts
        query = query & ((Alert.expires_at == None) | (Alert.expires_at > datetime.utcnow()))
        
        # Get paginated results
        skip = (page - 1) * page_size
        alerts = await Alert.find(query).sort(-Alert.created_at).skip(skip).limit(page_size).to_list()
        
        # Convert to response format
        alert_responses = []
        for alert in alerts:
            location_dict = None
            if alert.location:
                location_dict = {
                    "latitude": alert.location.latitude,
                    "longitude": alert.location.longitude
                }
            
            alert_responses.append(AlertResponse(
                id=str(alert.id),
                alert_type=alert.alert_type,
                severity=alert.severity,
                title=alert.title,
                message=alert.message,
                recommendations=alert.recommendations,
                is_forecast=alert.is_forecast,
                forecast_date=alert.forecast_date,
                location=location_dict,
                created_at=alert.created_at,
                expires_at=alert.expires_at,
                is_read=alert.is_read
            ))
        
        return alert_responses
        
    except Exception as e:
        logger.error(f"Failed to get alerts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve alerts"
        )


@router.get("/summary", response_model=AlertSummaryResponse)
async def get_alert_summary(
    current_user: User = Depends(get_current_user)
):
    """
    Get alert summary statistics for the current user
    
    Returns summary of alert counts by severity and status.
    """
    try:
        # Get all active alerts for user
        alerts = await Alert.find(
            (Alert.user_id == str(current_user.id)) &
            ((Alert.expires_at == None) | (Alert.expires_at > datetime.utcnow()))
        ).to_list()
        
        # Calculate statistics
        total_alerts = len(alerts)
        unread_alerts = len([a for a in alerts if not a.is_read])
        
        # Count by severity
        critical_alerts = len([a for a in alerts if a.severity == AlertSeverity.CRITICAL])
        high_alerts = len([a for a in alerts if a.severity == AlertSeverity.HIGH])
        medium_alerts = len([a for a in alerts if a.severity == AlertSeverity.MEDIUM])
        low_alerts = len([a for a in alerts if a.severity == AlertSeverity.LOW])
        
        # Recent alerts (last 24 hours)
        recent_date = datetime.utcnow() - timedelta(hours=24)
        recent_alerts = len([a for a in alerts if a.created_at >= recent_date])
        
        return AlertSummaryResponse(
            total_alerts=total_alerts,
            unread_alerts=unread_alerts,
            critical_alerts=critical_alerts,
            high_alerts=high_alerts,
            medium_alerts=medium_alerts,
            low_alerts=low_alerts,
            recent_alerts=recent_alerts
        )
        
    except Exception as e:
        logger.error(f"Failed to get alert summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve alert summary"
        )


@router.put("/{alert_id}/read")
async def mark_alert_as_read(
    alert_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Mark an alert as read
    
    Updates the alert status to read.
    """
    try:
        alert = await Alert.get(alert_id)
        
        if not alert:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Alert not found"
            )
        
        # Check if user owns this alert
        if alert.user_id != str(current_user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        # Mark as read
        alert.is_read = True
        await alert.save()
        
        return {"message": "Alert marked as read"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to mark alert as read: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to mark alert as read"
        )


@router.put("/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Acknowledge an alert
    
    Marks the alert as acknowledged by the user.
    """
    try:
        alert = await Alert.get(alert_id)
        
        if not alert:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Alert not found"
            )
        
        # Check if user owns this alert
        if alert.user_id != str(current_user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        # Acknowledge alert
        alert.is_acknowledged = True
        alert.acknowledged_at = datetime.utcnow()
        alert.is_read = True  # Also mark as read
        await alert.save()
        
        return {"message": "Alert acknowledged successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to acknowledge alert: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to acknowledge alert"
        )


@router.delete("/{alert_id}")
async def delete_alert(
    alert_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Delete an alert
    
    Removes the alert from the user's alert list.
    """
    try:
        alert = await Alert.get(alert_id)
        
        if not alert:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Alert not found"
            )
        
        # Check if user owns this alert
        if alert.user_id != str(current_user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        # Delete alert
        await alert.delete()
        
        return {"message": "Alert deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete alert: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete alert"
        )


@router.post("/bulk-read")
async def mark_all_alerts_as_read(
    field_id: Optional[str] = None,
    alert_type: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """
    Mark multiple alerts as read
    
    Bulk operation to mark alerts as read with optional filtering.
    """
    try:
        # Build query
        query = (Alert.user_id == str(current_user.id)) & (Alert.is_read == False)
        
        if field_id:
            query = query & (Alert.field_id == field_id)
        
        if alert_type:
            query = query & (Alert.alert_type == alert_type)
        
        # Update alerts
        alerts = await Alert.find(query).to_list()
        
        updated_count = 0
        for alert in alerts:
            alert.is_read = True
            await alert.save()
            updated_count += 1
        
        return {
            "message": f"Marked {updated_count} alerts as read",
            "updated_count": updated_count
        }
        
    except Exception as e:
        logger.error(f"Failed to mark alerts as read: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to mark alerts as read"
        )


@router.get("/types")
async def get_alert_types():
    """
    Get available alert types
    
    Returns list of all available alert types and their descriptions.
    """
    return {
        "alert_types": [
            {
                "type": "fungal_risk",
                "name": "Fungal Disease Risk",
                "description": "High temperature and humidity conditions favor fungal diseases",
                "severity_levels": ["high", "critical"]
            },
            {
                "type": "pest_risk",
                "name": "Pest Activity Risk",
                "description": "Weather conditions favorable for pest activity",
                "severity_levels": ["medium", "high"]
            },
            {
                "type": "soil_temperature",
                "name": "Soil Temperature Alert",
                "description": "Soil temperature outside optimal range for crop growth",
                "severity_levels": ["medium", "high"]
            },
            {
                "type": "excessive_rainfall",
                "name": "Excessive Rainfall",
                "description": "Unusual high rainfall patterns detected",
                "severity_levels": ["high", "critical"]
            },
            {
                "type": "drought_conditions",
                "name": "Drought Conditions",
                "description": "Low rainfall patterns detected",
                "severity_levels": ["high", "critical"]
            }
        ]
    }


# Background task functions
async def cleanup_expired_alerts(user_id: str):
    """Background task to clean up expired alerts"""
    try:
        expired_alerts = await Alert.find(
            (Alert.user_id == user_id) &
            (Alert.expires_at != None) &
            (Alert.expires_at < datetime.utcnow())
        ).to_list()
        
        for alert in expired_alerts:
            await alert.delete()
        
        if expired_alerts:
            logger.info(f"Cleaned up {len(expired_alerts)} expired alerts for user {user_id}")
            
    except Exception as e:
        logger.error(f"Failed to cleanup expired alerts: {e}")
