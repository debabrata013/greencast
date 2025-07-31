"""
Yield prediction API endpoints
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, validator
from loguru import logger

# Local imports
from backend.models.database import (
    YieldPrediction, User, Field, CropType, WeatherData, 
    SoilData, CropManagement, PredictionStatus
)
from backend.services.ml_service import yield_service
from backend.utils.auth import get_current_user

router = APIRouter()


# Pydantic models for request/response
class YieldPredictionRequest(BaseModel):
    """Yield prediction request model"""
    field_id: str = Field(..., description="Field ID for prediction")
    crop_type: CropType = Field(..., description="Type of crop")
    
    # Weather data
    temperature: float = Field(..., ge=-50, le=60, description="Temperature in Celsius")
    humidity: float = Field(..., ge=0, le=100, description="Humidity percentage")
    rainfall: float = Field(..., ge=0, description="Rainfall in mm")
    wind_speed: float = Field(..., ge=0, description="Wind speed in m/s")
    pressure: Optional[float] = Field(None, description="Atmospheric pressure in hPa")
    
    # Soil data
    soil_ph: float = Field(..., ge=0, le=14, description="Soil pH level")
    soil_nitrogen: float = Field(..., ge=0, description="Nitrogen content in ppm")
    soil_phosphorus: float = Field(..., ge=0, description="Phosphorus content in ppm")
    soil_potassium: float = Field(..., ge=0, description="Potassium content in ppm")
    soil_moisture: Optional[float] = Field(None, ge=0, le=100, description="Soil moisture percentage")
    
    # Management data
    fertilizer_amount: float = Field(..., ge=0, description="Fertilizer amount in kg/hectare")
    irrigation_hours: float = Field(..., ge=0, description="Irrigation hours per week")
    pesticide_applications: int = Field(default=0, ge=0, description="Number of pesticide applications")
    planting_date: Optional[datetime] = Field(None, description="Planting date")
    expected_harvest_date: Optional[datetime] = Field(None, description="Expected harvest date")
    
    # Additional parameters
    elevation: Optional[float] = Field(None, description="Elevation in meters")
    notes: Optional[str] = Field(None, description="Additional notes")
    
    @validator('expected_harvest_date')
    def validate_harvest_date(cls, v, values):
        if v and 'planting_date' in values and values['planting_date']:
            if v <= values['planting_date']:
                raise ValueError('Harvest date must be after planting date')
        return v


class YieldPredictionResponse(BaseModel):
    """Yield prediction response model"""
    id: str
    field_id: str
    crop_type: CropType
    predicted_yield: float
    confidence_interval: Dict[str, float]
    feature_importance: Dict[str, float]
    model_used: str
    prediction_date: datetime
    harvest_date: Optional[datetime]
    status: PredictionStatus
    created_at: datetime


class YieldHistoryResponse(BaseModel):
    """Yield prediction history response model"""
    predictions: List[YieldPredictionResponse]
    total_count: int
    page: int
    page_size: int


class YieldComparisonResponse(BaseModel):
    """Yield comparison response model"""
    field_id: str
    field_name: str
    predictions: List[Dict[str, Any]]
    average_yield: float
    yield_trend: str  # "increasing", "decreasing", "stable"


@router.post("/predict", response_model=YieldPredictionResponse)
async def predict_yield(
    request: YieldPredictionRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Predict crop yield based on weather, soil, and management data
    
    This endpoint accepts agricultural parameters and returns yield prediction.
    """
    try:
        # Validate field ownership
        field = await Field.get(request.field_id)
        if not field:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Field not found"
            )
        
        if field.user_id != str(current_user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this field"
            )
        
        # Prepare data models
        weather_data = WeatherData(
            temperature=request.temperature,
            humidity=request.humidity,
            rainfall=request.rainfall,
            wind_speed=request.wind_speed,
            pressure=request.pressure or 1013.25,
            soil_temperature=request.temperature - 3  # Approximate soil temp
        )
        
        soil_data = SoilData(
            ph=request.soil_ph,
            nitrogen=request.soil_nitrogen,
            phosphorus=request.soil_phosphorus,
            potassium=request.soil_potassium,
            moisture=request.soil_moisture
        )
        
        management_data = CropManagement(
            fertilizer_amount=request.fertilizer_amount,
            irrigation_hours=request.irrigation_hours,
            pesticide_applications=request.pesticide_applications,
            planting_date=request.planting_date,
            expected_harvest_date=request.expected_harvest_date
        )
        
        # Get field elevation or use default
        elevation = request.elevation or field.location.elevation or 300.0
        
        # Perform yield prediction
        prediction_result = await yield_service.predict_yield(
            crop_type=request.crop_type,
            weather_data=weather_data,
            soil_data=soil_data,
            management_data=management_data,
            field_size=field.size,
            elevation=elevation
        )
        
        # Save prediction to database
        prediction = YieldPrediction(
            user_id=str(current_user.id),
            field_id=request.field_id,
            crop_type=request.crop_type,
            weather_data=weather_data,
            soil_data=soil_data,
            management_data=management_data,
            predicted_yield=prediction_result["predicted_yield"],
            confidence_interval=prediction_result["confidence_interval"],
            feature_importance=prediction_result["feature_importance"],
            model_used=prediction_result["model_used"],
            harvest_date=request.expected_harvest_date,
            notes=request.notes
        )
        
        await prediction.insert()
        
        logger.info(f"Yield prediction completed for user {current_user.email}: {prediction_result['predicted_yield']:.2f} tons/hectare")
        
        return YieldPredictionResponse(
            id=str(prediction.id),
            field_id=prediction.field_id,
            crop_type=prediction.crop_type,
            predicted_yield=prediction.predicted_yield,
            confidence_interval=prediction.confidence_interval,
            feature_importance=prediction.feature_importance,
            model_used=prediction.model_used,
            prediction_date=prediction.prediction_date,
            harvest_date=prediction.harvest_date,
            status=prediction.status,
            created_at=prediction.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Yield prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Yield prediction processing failed"
        )


@router.get("/history", response_model=YieldHistoryResponse)
async def get_yield_prediction_history(
    page: int = 1,
    page_size: int = 20,
    field_id: Optional[str] = None,
    crop_type: Optional[CropType] = None,
    current_user: User = Depends(get_current_user)
):
    """
    Get yield prediction history for the current user
    
    Returns paginated list of previous yield predictions with filtering options.
    """
    try:
        # Build query
        query = YieldPrediction.user_id == str(current_user.id)
        
        if field_id:
            query = query & (YieldPrediction.field_id == field_id)
        
        if crop_type:
            query = query & (YieldPrediction.crop_type == crop_type)
        
        # Get total count
        total_count = await YieldPrediction.find(query).count()
        
        # Get paginated results
        skip = (page - 1) * page_size
        predictions = await YieldPrediction.find(query).sort(-YieldPrediction.created_at).skip(skip).limit(page_size).to_list()
        
        # Convert to response format
        prediction_responses = []
        for prediction in predictions:
            prediction_responses.append(YieldPredictionResponse(
                id=str(prediction.id),
                field_id=prediction.field_id,
                crop_type=prediction.crop_type,
                predicted_yield=prediction.predicted_yield,
                confidence_interval=prediction.confidence_interval,
                feature_importance=prediction.feature_importance,
                model_used=prediction.model_used,
                prediction_date=prediction.prediction_date,
                harvest_date=prediction.harvest_date,
                status=prediction.status,
                created_at=prediction.created_at
            ))
        
        return YieldHistoryResponse(
            predictions=prediction_responses,
            total_count=total_count,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        logger.error(f"Failed to get yield prediction history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve yield prediction history"
        )


@router.get("/{prediction_id}")
async def get_yield_prediction(
    prediction_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get specific yield prediction by ID
    
    Returns detailed information about a specific yield prediction.
    """
    try:
        prediction = await YieldPrediction.get(prediction_id)
        
        if not prediction:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Yield prediction not found"
            )
        
        # Check if user owns this prediction
        if prediction.user_id != str(current_user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        # Get field information
        field = await Field.get(prediction.field_id)
        
        return {
            "id": str(prediction.id),
            "field_id": prediction.field_id,
            "field_name": field.name if field else "Unknown Field",
            "crop_type": prediction.crop_type,
            "predicted_yield": prediction.predicted_yield,
            "confidence_interval": prediction.confidence_interval,
            "feature_importance": prediction.feature_importance,
            "model_used": prediction.model_used,
            "prediction_date": prediction.prediction_date,
            "harvest_date": prediction.harvest_date,
            "actual_yield": prediction.actual_yield,
            "weather_data": prediction.weather_data,
            "soil_data": prediction.soil_data,
            "management_data": prediction.management_data,
            "notes": prediction.notes,
            "status": prediction.status,
            "created_at": prediction.created_at
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get yield prediction {prediction_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve yield prediction"
        )


@router.put("/{prediction_id}/actual-yield")
async def update_actual_yield(
    prediction_id: str,
    actual_yield: float = Field(..., ge=0, description="Actual yield in tons/hectare"),
    current_user: User = Depends(get_current_user)
):
    """
    Update actual yield after harvest
    
    Allows farmers to record the actual yield for model improvement.
    """
    try:
        prediction = await YieldPrediction.get(prediction_id)
        
        if not prediction:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Yield prediction not found"
            )
        
        # Check if user owns this prediction
        if prediction.user_id != str(current_user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        # Update actual yield
        prediction.actual_yield = actual_yield
        await prediction.save()
        
        # Calculate accuracy
        accuracy = 1 - abs(prediction.predicted_yield - actual_yield) / prediction.predicted_yield
        
        logger.info(f"Actual yield updated for prediction {prediction_id}: {actual_yield} tons/hectare")
        
        return {
            "message": "Actual yield updated successfully",
            "predicted_yield": prediction.predicted_yield,
            "actual_yield": actual_yield,
            "accuracy": accuracy,
            "difference": actual_yield - prediction.predicted_yield
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update actual yield for {prediction_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update actual yield"
        )


@router.get("/comparison/fields")
async def compare_field_yields(
    current_user: User = Depends(get_current_user)
):
    """
    Compare yield predictions across user's fields
    
    Returns yield comparison data for all user fields.
    """
    try:
        # Get user's fields
        fields = await Field.find(Field.user_id == str(current_user.id)).to_list()
        
        if not fields:
            return {"message": "No fields found"}
        
        comparisons = []
        
        for field in fields:
            # Get predictions for this field
            predictions = await YieldPrediction.find(
                YieldPrediction.field_id == str(field.id)
            ).sort(-YieldPrediction.prediction_date).to_list()
            
            if predictions:
                # Calculate statistics
                yields = [p.predicted_yield for p in predictions]
                average_yield = sum(yields) / len(yields)
                
                # Determine trend
                if len(yields) >= 2:
                    recent_avg = sum(yields[:len(yields)//2]) / (len(yields)//2)
                    older_avg = sum(yields[len(yields)//2:]) / (len(yields) - len(yields)//2)
                    
                    if recent_avg > older_avg * 1.1:
                        trend = "increasing"
                    elif recent_avg < older_avg * 0.9:
                        trend = "decreasing"
                    else:
                        trend = "stable"
                else:
                    trend = "insufficient_data"
                
                # Prepare prediction data
                prediction_data = []
                for p in predictions[-10:]:  # Last 10 predictions
                    prediction_data.append({
                        "date": p.prediction_date.isoformat(),
                        "predicted_yield": p.predicted_yield,
                        "actual_yield": p.actual_yield,
                        "crop_type": p.crop_type.value
                    })
                
                comparisons.append(YieldComparisonResponse(
                    field_id=str(field.id),
                    field_name=field.name,
                    predictions=prediction_data,
                    average_yield=average_yield,
                    yield_trend=trend
                ))
        
        return {"field_comparisons": comparisons}
        
    except Exception as e:
        logger.error(f"Failed to compare field yields: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to compare field yields"
        )


@router.get("/statistics/summary")
async def get_yield_statistics(
    current_user: User = Depends(get_current_user)
):
    """
    Get yield prediction statistics for the current user
    
    Returns summary statistics about yield predictions.
    """
    try:
        # Get all predictions for user
        predictions = await YieldPrediction.find(
            YieldPrediction.user_id == str(current_user.id)
        ).to_list()
        
        if not predictions:
            return {
                "total_predictions": 0,
                "average_predicted_yield": 0,
                "crop_distribution": {},
                "model_accuracy": None,
                "recent_predictions": 0
            }
        
        # Calculate statistics
        total_predictions = len(predictions)
        average_yield = sum(p.predicted_yield for p in predictions) / total_predictions
        
        # Crop distribution
        crop_counts = {}
        for prediction in predictions:
            crop = prediction.crop_type.value
            crop_counts[crop] = crop_counts.get(crop, 0) + 1
        
        # Model accuracy (where actual yield is available)
        predictions_with_actual = [p for p in predictions if p.actual_yield is not None]
        accuracy = None
        if predictions_with_actual:
            accuracies = []
            for p in predictions_with_actual:
                acc = 1 - abs(p.predicted_yield - p.actual_yield) / p.predicted_yield
                accuracies.append(max(0, acc))  # Ensure non-negative
            accuracy = sum(accuracies) / len(accuracies)
        
        # Recent predictions (last 30 days)
        recent_date = datetime.utcnow() - timedelta(days=30)
        recent_predictions = len([p for p in predictions if p.created_at >= recent_date])
        
        return {
            "total_predictions": total_predictions,
            "average_predicted_yield": average_yield,
            "crop_distribution": crop_counts,
            "model_accuracy": accuracy,
            "recent_predictions": recent_predictions,
            "predictions_with_actual": len(predictions_with_actual)
        }
        
    except Exception as e:
        logger.error(f"Failed to get yield statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve yield statistics"
        )
