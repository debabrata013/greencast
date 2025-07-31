"""
Machine Learning services for GreenCast application
Integrates trained models for disease detection, yield prediction, and alerts
"""

import os
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import pickle
import joblib
from pathlib import Path

# Image processing
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# ML libraries
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest

# Logging
from loguru import logger

# Local imports
from backend.core.config import MLConfig, settings
from backend.models.database import WeatherData, SoilData, CropManagement, CropType, AlertSeverity


class ModelLoadError(Exception):
    """Custom exception for model loading errors"""
    pass


class PredictionError(Exception):
    """Custom exception for prediction errors"""
    pass


class DiseaseDetectionService:
    """Service for plant disease detection using CNN models"""
    
    def __init__(self):
        self.model = None
        self.class_names = []
        self.input_shape = (224, 224, 3)
        self.is_loaded = False
        
        # Disease treatment recommendations database
        self.treatment_recommendations = {
            "Apple___Apple_scab": [
                "Apply fungicide spray during wet weather",
                "Remove fallen leaves and debris",
                "Improve air circulation around trees",
                "Use resistant apple varieties"
            ],
            "Apple___Black_rot": [
                "Prune infected branches immediately",
                "Apply copper-based fungicide",
                "Remove mummified fruits",
                "Maintain proper tree spacing"
            ],
            "Corn_(maize)___Northern_Leaf_Blight": [
                "Apply foliar fungicide",
                "Use resistant corn varieties",
                "Rotate crops to break disease cycle",
                "Remove crop residue after harvest"
            ],
            "Tomato___Late_blight": [
                "Apply preventive fungicide spray",
                "Improve ventilation in greenhouse",
                "Avoid overhead watering",
                "Remove infected plants immediately"
            ],
            "healthy": [
                "Continue current management practices",
                "Monitor regularly for early detection",
                "Maintain proper nutrition and watering",
                "Keep field records updated"
            ]
        }
    
    async def load_model(self) -> bool:
        """Load the disease detection model"""
        try:
            model_path = MLConfig.get_disease_model_path()
            
            if not os.path.exists(model_path):
                logger.error(f"Disease model not found at: {model_path}")
                return False
            
            # Load TensorFlow model
            self.model = load_model(model_path, compile=False)
            
            # Load class names (you might want to store this separately)
            self.class_names = self._get_disease_classes()
            
            self.is_loaded = True
            logger.info(f"Disease detection model loaded successfully from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load disease detection model: {e}")
            raise ModelLoadError(f"Failed to load disease detection model: {e}")
    
    def _get_disease_classes(self) -> List[str]:
        """Get disease class names"""
        # This should ideally be loaded from a metadata file
        return [
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
            'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
            'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
            'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
            'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
            'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
            'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
            'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
            'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
            'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
            'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
            'Tomato___healthy'
        ]
    
    async def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess image for model prediction"""
        try:
            # Load image
            image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to model input size
            image = image.resize(self.input_shape[:2])
            
            # Convert to array and normalize
            img_array = img_to_array(image)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0
            
            return img_array
            
        except Exception as e:
            logger.error(f"Failed to preprocess image {image_path}: {e}")
            raise PredictionError(f"Image preprocessing failed: {e}")
    
    async def predict_disease(self, image_path: str) -> Dict[str, Any]:
        """Predict disease from plant image"""
        try:
            if not self.is_loaded:
                await self.load_model()
            
            # Preprocess image
            img_array = await self.preprocess_image(image_path)
            
            # Make prediction
            start_time = datetime.now()
            predictions = self.model.predict(img_array, verbose=0)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Get prediction results
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class = self.class_names[predicted_class_idx]
            confidence_score = float(predictions[0][predicted_class_idx])
            
            # Get all probabilities
            all_probabilities = {
                class_name: float(prob) 
                for class_name, prob in zip(self.class_names, predictions[0])
            }
            
            # Determine severity based on disease type and confidence
            severity = self._determine_severity(predicted_class, confidence_score)
            
            # Get treatment recommendations
            recommendations = self.treatment_recommendations.get(
                predicted_class, 
                ["Consult with agricultural expert for specific treatment"]
            )
            
            return {
                "predicted_disease": predicted_class,
                "confidence_score": confidence_score,
                "all_probabilities": all_probabilities,
                "severity_level": severity,
                "treatment_recommendations": recommendations,
                "processing_time": processing_time,
                "model_version": "mobilenetv2_v1.0"
            }
            
        except Exception as e:
            logger.error(f"Disease prediction failed: {e}")
            raise PredictionError(f"Disease prediction failed: {e}")
    
    def _determine_severity(self, disease_class: str, confidence: float) -> AlertSeverity:
        """Determine disease severity based on class and confidence"""
        if "healthy" in disease_class.lower():
            return AlertSeverity.LOW
        
        # High confidence in disease detection
        if confidence > 0.8:
            # Critical diseases
            if any(critical in disease_class.lower() for critical in ["blight", "rot", "virus"]):
                return AlertSeverity.CRITICAL
            else:
                return AlertSeverity.HIGH
        elif confidence > 0.6:
            return AlertSeverity.MEDIUM
        else:
            return AlertSeverity.LOW


class YieldPredictionService:
    """Service for crop yield prediction using ML models"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = []
        self.is_loaded = False
        
        # Crop-specific yield ranges (tons/hectare)
        self.yield_ranges = {
            CropType.WHEAT: (2.0, 6.0),
            CropType.CORN: (4.0, 12.0),
            CropType.RICE: (3.0, 8.0),
            CropType.SOYBEAN: (1.5, 4.0),
            CropType.BARLEY: (2.5, 5.5),
            CropType.COTTON: (0.8, 2.5)
        }
    
    async def load_model(self) -> bool:
        """Load the yield prediction model"""
        try:
            model_path = MLConfig.get_yield_model_path()
            
            if not os.path.exists(model_path):
                logger.error(f"Yield model not found at: {model_path}")
                return False
            
            # Load model (assuming it's a pickled scikit-learn model)
            self.model = joblib.load(model_path)
            
            # Load preprocessing components (if available)
            model_dir = Path(model_path).parent
            scaler_path = model_dir / "yield_scaler.pkl"
            encoders_path = model_dir / "yield_encoders.pkl"
            
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
            
            if encoders_path.exists():
                self.label_encoders = joblib.load(encoders_path)
            
            # Set feature names
            self.feature_names = [
                'crop_type', 'soil_ph', 'rainfall_mm', 'humidity_percent',
                'temperature_celsius', 'fertilizer_kg_per_hectare', 'irrigation_hours',
                'sunlight_hours', 'soil_nitrogen', 'soil_phosphorus', 'soil_potassium',
                'elevation_meters', 'field_size_hectares'
            ]
            
            self.is_loaded = True
            logger.info(f"Yield prediction model loaded successfully from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load yield prediction model: {e}")
            raise ModelLoadError(f"Failed to load yield prediction model: {e}")
    
    async def predict_yield(
        self,
        crop_type: CropType,
        weather_data: WeatherData,
        soil_data: SoilData,
        management_data: CropManagement,
        field_size: float,
        elevation: float = 300.0
    ) -> Dict[str, Any]:
        """Predict crop yield based on input parameters"""
        try:
            if not self.is_loaded:
                await self.load_model()
            
            # Prepare input features
            features = self._prepare_features(
                crop_type, weather_data, soil_data, management_data, field_size, elevation
            )
            
            # Make prediction
            start_time = datetime.now()
            predicted_yield = self.model.predict([features])[0]
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Validate prediction against expected ranges
            expected_range = self.yield_ranges.get(crop_type, (1.0, 10.0))
            predicted_yield = np.clip(predicted_yield, expected_range[0], expected_range[1])
            
            # Get feature importance (if available)
            feature_importance = {}
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = dict(zip(
                    self.feature_names,
                    self.model.feature_importances_
                ))
            
            # Calculate confidence interval (simplified)
            confidence_interval = self._calculate_confidence_interval(predicted_yield, crop_type)
            
            return {
                "predicted_yield": float(predicted_yield),
                "confidence_interval": confidence_interval,
                "feature_importance": feature_importance,
                "model_used": type(self.model).__name__,
                "processing_time": processing_time,
                "expected_range": expected_range,
                "model_version": "xgboost_v1.0"
            }
            
        except Exception as e:
            logger.error(f"Yield prediction failed: {e}")
            raise PredictionError(f"Yield prediction failed: {e}")
    
    def _prepare_features(
        self,
        crop_type: CropType,
        weather_data: WeatherData,
        soil_data: SoilData,
        management_data: CropManagement,
        field_size: float,
        elevation: float
    ) -> List[float]:
        """Prepare features for model input"""
        
        # Encode crop type
        crop_mapping = {
            CropType.WHEAT: 0, CropType.CORN: 1, CropType.RICE: 2,
            CropType.SOYBEAN: 3, CropType.BARLEY: 4, CropType.COTTON: 5
        }
        
        features = [
            crop_mapping.get(crop_type, 0),
            soil_data.ph,
            weather_data.rainfall,
            weather_data.humidity,
            weather_data.temperature,
            management_data.fertilizer_amount,
            management_data.irrigation_hours,
            8.0,  # Default sunlight hours
            soil_data.nitrogen,
            soil_data.phosphorus,
            soil_data.potassium,
            elevation,
            field_size
        ]
        
        return features
    
    def _calculate_confidence_interval(self, predicted_yield: float, crop_type: CropType) -> Dict[str, float]:
        """Calculate confidence interval for prediction"""
        # Simplified confidence interval calculation
        # In production, this should be based on model uncertainty
        uncertainty = predicted_yield * 0.15  # 15% uncertainty
        
        return {
            "lower": max(0, predicted_yield - uncertainty),
            "upper": predicted_yield + uncertainty,
            "uncertainty_percent": 15.0
        }


class AlertService:
    """Service for generating agricultural alerts"""
    
    def __init__(self):
        self.ml_model = None
        self.scaler = None
        self.anomaly_detector = None
        self.is_loaded = False
        
        # Alert rule thresholds
        self.thresholds = {
            "fungal_risk": {"temp": 28.0, "humidity": 80.0, "duration_days": 3},
            "pest_risk": {"temp_min": 20.0, "temp_max": 30.0, "humidity": 70.0, "wind_max": 2.0},
            "soil_temp": {
                CropType.WHEAT: (10, 25),
                CropType.CORN: (15, 35),
                CropType.RICE: (20, 35),
                CropType.SOYBEAN: (15, 30),
                CropType.BARLEY: (10, 25),
                CropType.COTTON: (20, 35)
            }
        }
    
    async def load_models(self) -> bool:
        """Load alert prediction models"""
        try:
            # Initialize anomaly detector for rainfall
            self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
            
            # Train with synthetic historical data (in production, use real data)
            historical_rainfall = np.random.gamma(2, 2, 100).reshape(-1, 1)
            self.anomaly_detector.fit(historical_rainfall)
            
            self.is_loaded = True
            logger.info("Alert service models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load alert models: {e}")
            raise ModelLoadError(f"Failed to load alert models: {e}")
    
    async def check_fungal_risk(self, weather_data: WeatherData, history: List[WeatherData] = None) -> Dict[str, Any]:
        """Check for fungal disease risk"""
        thresholds = self.thresholds["fungal_risk"]
        
        current_risk = (
            weather_data.temperature > thresholds["temp"] and
            weather_data.humidity > thresholds["humidity"]
        )
        
        # Check persistence if history is available
        persistent_risk = current_risk
        if history and len(history) >= thresholds["duration_days"]:
            recent_risks = [
                (w.temperature > thresholds["temp"] and w.humidity > thresholds["humidity"])
                for w in history[-thresholds["duration_days"]:]
            ]
            persistent_risk = all(recent_risks)
        
        if persistent_risk:
            return {
                "alert_type": "fungal_risk",
                "severity": AlertSeverity.HIGH,
                "title": "Fungal Disease Risk Alert",
                "message": f"High temperature ({weather_data.temperature:.1f}°C) and humidity ({weather_data.humidity:.1f}%) conditions favor fungal diseases",
                "recommendations": [
                    "Apply preventive fungicide spray",
                    "Improve ventilation around plants",
                    "Reduce irrigation frequency",
                    "Monitor plants closely for early symptoms"
                ]
            }
        
        return None
    
    async def check_pest_risk(self, weather_data: WeatherData) -> Dict[str, Any]:
        """Check for pest activity risk"""
        thresholds = self.thresholds["pest_risk"]
        
        risk_score = 0
        
        # Temperature factor
        if thresholds["temp_min"] <= weather_data.temperature <= thresholds["temp_max"]:
            risk_score += 3
        
        # Humidity factor
        if weather_data.humidity > thresholds["humidity"]:
            risk_score += 2
        
        # Wind factor
        if weather_data.wind_speed < thresholds["wind_max"]:
            risk_score += 2
        
        if risk_score >= 4:
            return {
                "alert_type": "pest_risk",
                "severity": AlertSeverity.MEDIUM,
                "title": "Pest Activity Risk Alert",
                "message": f"Weather conditions favor pest activity (Risk Score: {risk_score}/7)",
                "recommendations": [
                    "Monitor crops for pest activity",
                    "Check and maintain pest traps",
                    "Consider preventive treatments",
                    "Inspect plants regularly"
                ]
            }
        
        return None
    
    async def check_soil_temperature(self, soil_temp: float, crop_type: CropType) -> Dict[str, Any]:
        """Check soil temperature thresholds"""
        temp_range = self.thresholds["soil_temp"].get(crop_type, (15, 30))
        min_temp, max_temp = temp_range
        
        if soil_temp < min_temp:
            return {
                "alert_type": "soil_temperature",
                "severity": AlertSeverity.MEDIUM,
                "title": "Low Soil Temperature Alert",
                "message": f"Soil temperature ({soil_temp:.1f}°C) is below optimal range for {crop_type.value}",
                "recommendations": [
                    "Consider soil warming techniques",
                    "Adjust planting schedule",
                    "Monitor seed germination",
                    "Use mulching to retain heat"
                ]
            }
        elif soil_temp > max_temp:
            return {
                "alert_type": "soil_temperature",
                "severity": AlertSeverity.MEDIUM,
                "title": "High Soil Temperature Alert",
                "message": f"Soil temperature ({soil_temp:.1f}°C) is above optimal range for {crop_type.value}",
                "recommendations": [
                    "Increase irrigation frequency",
                    "Use mulching to cool soil",
                    "Provide shade if possible",
                    "Monitor plant stress"
                ]
            }
        
        return None
    
    async def check_rainfall_anomaly(self, rainfall: float, history: List[float] = None) -> Dict[str, Any]:
        """Check for rainfall anomalies"""
        if not self.is_loaded:
            await self.load_models()
        
        # Check for anomaly
        is_anomaly = self.anomaly_detector.predict([[rainfall]])[0] == -1
        
        if is_anomaly:
            # Determine if it's drought or flood
            recent_avg = np.mean(history[-7:]) if history and len(history) >= 7 else rainfall
            
            if rainfall > recent_avg * 2:
                alert_type = "excessive_rainfall"
                message = f"Excessive rainfall detected ({rainfall:.1f}mm)"
                recommendations = [
                    "Check drainage systems",
                    "Monitor for waterlogging",
                    "Prevent soil erosion",
                    "Delay field operations if necessary"
                ]
            else:
                alert_type = "drought_conditions"
                message = f"Drought conditions detected ({rainfall:.1f}mm)"
                recommendations = [
                    "Increase irrigation frequency",
                    "Monitor soil moisture levels",
                    "Consider drought-resistant practices",
                    "Conserve water resources"
                ]
            
            return {
                "alert_type": alert_type,
                "severity": AlertSeverity.HIGH,
                "title": "Rainfall Anomaly Alert",
                "message": message,
                "recommendations": recommendations
            }
        
        return None
    
    async def generate_alerts(
        self,
        weather_data: WeatherData,
        crop_type: CropType,
        weather_history: List[WeatherData] = None,
        rainfall_history: List[float] = None
    ) -> List[Dict[str, Any]]:
        """Generate all applicable alerts"""
        alerts = []
        
        # Check fungal risk
        fungal_alert = await self.check_fungal_risk(weather_data, weather_history)
        if fungal_alert:
            alerts.append(fungal_alert)
        
        # Check pest risk
        pest_alert = await self.check_pest_risk(weather_data)
        if pest_alert:
            alerts.append(pest_alert)
        
        # Check soil temperature
        soil_temp = weather_data.soil_temperature or weather_data.temperature - 3
        soil_alert = await self.check_soil_temperature(soil_temp, crop_type)
        if soil_alert:
            alerts.append(soil_alert)
        
        # Check rainfall anomaly
        rainfall_alert = await self.check_rainfall_anomaly(weather_data.rainfall, rainfall_history)
        if rainfall_alert:
            alerts.append(rainfall_alert)
        
        return alerts


# Global service instances
disease_service = DiseaseDetectionService()
yield_service = YieldPredictionService()
alert_service = AlertService()


# Service initialization
async def initialize_ml_services():
    """Initialize all ML services"""
    try:
        await disease_service.load_model()
        await yield_service.load_model()
        await alert_service.load_models()
        logger.info("All ML services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize ML services: {e}")
        raise


# Export services
__all__ = [
    "disease_service",
    "yield_service", 
    "alert_service",
    "initialize_ml_services",
    "DiseaseDetectionService",
    "YieldPredictionService",
    "AlertService"
]
