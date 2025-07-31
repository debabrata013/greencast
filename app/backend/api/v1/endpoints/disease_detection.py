"""
Disease detection API endpoints
"""

import os
import uuid
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from loguru import logger

# Local imports
from backend.models.database import DiseaseDetection, User, Field, CropType, AlertSeverity, ImageMetadata
from backend.services.ml_service import disease_service
from backend.core.config import FileConfig
from backend.utils.auth import get_current_user
from backend.utils.file_handler import save_uploaded_file, validate_image_file

router = APIRouter()


# Pydantic models for request/response
class DiseaseDetectionRequest(BaseModel):
    """Disease detection request model"""
    field_id: Optional[str] = None
    crop_type: CropType
    notes: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class DiseaseDetectionResponse(BaseModel):
    """Disease detection response model"""
    id: str
    predicted_disease: str
    confidence_score: float
    severity_level: AlertSeverity
    treatment_recommendations: List[str]
    processing_time: float
    created_at: datetime
    image_url: str


class DiseaseHistoryResponse(BaseModel):
    """Disease detection history response model"""
    detections: List[DiseaseDetectionResponse]
    total_count: int
    page: int
    page_size: int


@router.post("/upload", response_model=DiseaseDetectionResponse)
async def upload_crop_image(
    image: UploadFile = File(..., description="Plant image for disease detection"),
    field_id: Optional[str] = Form(None, description="Associated field ID"),
    crop_type: CropType = Form(..., description="Type of crop"),
    notes: Optional[str] = Form(None, description="Additional notes"),
    latitude: Optional[float] = Form(None, description="Latitude coordinate"),
    longitude: Optional[float] = Form(None, description="Longitude coordinate"),
    current_user: User = Depends(get_current_user)
):
    """
    Upload crop image for disease detection
    
    This endpoint accepts a plant image and returns disease prediction results.
    """
    try:
        # Validate image file
        if not validate_image_file(image):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid image file. Supported formats: JPG, JPEG, PNG, GIF"
            )
        
        # Check file size
        if image.size > FileConfig.get_max_file_size():
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size exceeds maximum limit of {FileConfig.get_max_file_size() / (1024*1024):.1f}MB"
            )
        
        # Save uploaded file
        file_info = await save_uploaded_file(image, "disease_detection")
        
        # Create image metadata
        image_metadata = ImageMetadata(
            filename=image.filename,
            file_path=file_info["file_path"],
            file_size=image.size,
            mime_type=image.content_type,
            width=file_info.get("width"),
            height=file_info.get("height")
        )
        
        # Perform disease detection
        prediction_result = await disease_service.predict_disease(file_info["file_path"])
        
        # Create location data if provided
        location = None
        if latitude is not None and longitude is not None:
            from backend.models.database import Location
            location = Location(latitude=latitude, longitude=longitude)
        
        # Save detection result to database
        detection = DiseaseDetection(
            user_id=str(current_user.id),
            field_id=field_id,
            image_metadata=image_metadata,
            crop_type=crop_type,
            predicted_disease=prediction_result["predicted_disease"],
            confidence_score=prediction_result["confidence_score"],
            all_probabilities=prediction_result["all_probabilities"],
            treatment_recommendations=prediction_result["treatment_recommendations"],
            severity_level=prediction_result["severity_level"],
            location=location,
            notes=notes,
            processing_time=prediction_result["processing_time"]
        )
        
        await detection.insert()
        
        # Log the detection
        logger.info(f"Disease detection completed for user {current_user.email}: {prediction_result['predicted_disease']}")
        
        # Return response
        return DiseaseDetectionResponse(
            id=str(detection.id),
            predicted_disease=detection.predicted_disease,
            confidence_score=detection.confidence_score,
            severity_level=detection.severity_level,
            treatment_recommendations=detection.treatment_recommendations,
            processing_time=detection.processing_time,
            created_at=detection.created_at,
            image_url=f"/static/{file_info['filename']}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Disease detection failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Disease detection processing failed"
        )


@router.get("/history", response_model=DiseaseHistoryResponse)
async def get_disease_detection_history(
    page: int = 1,
    page_size: int = 20,
    field_id: Optional[str] = None,
    crop_type: Optional[CropType] = None,
    severity: Optional[AlertSeverity] = None,
    current_user: User = Depends(get_current_user)
):
    """
    Get disease detection history for the current user
    
    Returns paginated list of previous disease detections with filtering options.
    """
    try:
        # Build query
        query = DiseaseDetection.user_id == str(current_user.id)
        
        if field_id:
            query = query & (DiseaseDetection.field_id == field_id)
        
        if crop_type:
            query = query & (DiseaseDetection.crop_type == crop_type)
        
        if severity:
            query = query & (DiseaseDetection.severity_level == severity)
        
        # Get total count
        total_count = await DiseaseDetection.find(query).count()
        
        # Get paginated results
        skip = (page - 1) * page_size
        detections = await DiseaseDetection.find(query).sort(-DiseaseDetection.created_at).skip(skip).limit(page_size).to_list()
        
        # Convert to response format
        detection_responses = []
        for detection in detections:
            detection_responses.append(DiseaseDetectionResponse(
                id=str(detection.id),
                predicted_disease=detection.predicted_disease,
                confidence_score=detection.confidence_score,
                severity_level=detection.severity_level,
                treatment_recommendations=detection.treatment_recommendations,
                processing_time=detection.processing_time,
                created_at=detection.created_at,
                image_url=f"/static/{detection.image_metadata.filename}"
            ))
        
        return DiseaseHistoryResponse(
            detections=detection_responses,
            total_count=total_count,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        logger.error(f"Failed to get disease detection history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve disease detection history"
        )


@router.get("/{detection_id}")
async def get_disease_detection(
    detection_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get specific disease detection by ID
    
    Returns detailed information about a specific disease detection.
    """
    try:
        detection = await DiseaseDetection.get(detection_id)
        
        if not detection:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Disease detection not found"
            )
        
        # Check if user owns this detection
        if detection.user_id != str(current_user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        return {
            "id": str(detection.id),
            "predicted_disease": detection.predicted_disease,
            "confidence_score": detection.confidence_score,
            "all_probabilities": detection.all_probabilities,
            "severity_level": detection.severity_level,
            "treatment_recommendations": detection.treatment_recommendations,
            "crop_type": detection.crop_type,
            "field_id": detection.field_id,
            "location": detection.location,
            "notes": detection.notes,
            "processing_time": detection.processing_time,
            "created_at": detection.created_at,
            "image_metadata": detection.image_metadata,
            "image_url": f"/static/{detection.image_metadata.filename}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get disease detection {detection_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve disease detection"
        )


@router.delete("/{detection_id}")
async def delete_disease_detection(
    detection_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Delete a disease detection record
    
    Removes the detection record and associated image file.
    """
    try:
        detection = await DiseaseDetection.get(detection_id)
        
        if not detection:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Disease detection not found"
            )
        
        # Check if user owns this detection
        if detection.user_id != str(current_user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        # Delete image file
        try:
            if os.path.exists(detection.image_metadata.file_path):
                os.remove(detection.image_metadata.file_path)
        except Exception as e:
            logger.warning(f"Failed to delete image file: {e}")
        
        # Delete database record
        await detection.delete()
        
        logger.info(f"Disease detection {detection_id} deleted by user {current_user.email}")
        
        return {"message": "Disease detection deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete disease detection {detection_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete disease detection"
        )


@router.get("/statistics/summary")
async def get_disease_statistics(
    current_user: User = Depends(get_current_user)
):
    """
    Get disease detection statistics for the current user
    
    Returns summary statistics about disease detections.
    """
    try:
        # Get all detections for user
        detections = await DiseaseDetection.find(
            DiseaseDetection.user_id == str(current_user.id)
        ).to_list()
        
        if not detections:
            return {
                "total_detections": 0,
                "disease_distribution": {},
                "severity_distribution": {},
                "crop_distribution": {},
                "recent_detections": 0
            }
        
        # Calculate statistics
        total_detections = len(detections)
        
        # Disease distribution
        disease_counts = {}
        for detection in detections:
            disease = detection.predicted_disease
            disease_counts[disease] = disease_counts.get(disease, 0) + 1
        
        # Severity distribution
        severity_counts = {}
        for detection in detections:
            severity = detection.severity_level.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Crop distribution
        crop_counts = {}
        for detection in detections:
            crop = detection.crop_type.value
            crop_counts[crop] = crop_counts.get(crop, 0) + 1
        
        # Recent detections (last 7 days)
        from datetime import timedelta
        recent_date = datetime.utcnow() - timedelta(days=7)
        recent_detections = len([d for d in detections if d.created_at >= recent_date])
        
        return {
            "total_detections": total_detections,
            "disease_distribution": disease_counts,
            "severity_distribution": severity_counts,
            "crop_distribution": crop_counts,
            "recent_detections": recent_detections,
            "average_confidence": sum(d.confidence_score for d in detections) / total_detections
        }
        
    except Exception as e:
        logger.error(f"Failed to get disease statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve disease statistics"
        )
