"""
Core configuration module for GreenCast application
"""

import os
from typing import List, Optional
from pydantic import BaseSettings, validator
from decouple import config


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Basic Configuration
    PROJECT_NAME: str = config("PROJECT_NAME", default="GreenCast Agricultural Intelligence")
    VERSION: str = config("VERSION", default="1.0.0")
    DESCRIPTION: str = config("DESCRIPTION", default="AI-powered agricultural platform")
    API_V1_STR: str = config("API_V1_STR", default="/api/v1")
    ENVIRONMENT: str = config("ENVIRONMENT", default="development")
    DEBUG: bool = config("DEBUG", default=True, cast=bool)
    
    # Database Configuration
    MONGODB_URL: str = config("MONGODB_URL", default="mongodb://localhost:27017")
    DATABASE_NAME: str = config("DATABASE_NAME", default="greencast_db")
    
    # Security Configuration
    SECRET_KEY: str = config("SECRET_KEY", default="your-secret-key-change-in-production")
    ALGORITHM: str = config("ALGORITHM", default="HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = config("ACCESS_TOKEN_EXPIRE_MINUTES", default=30, cast=int)
    
    # File Upload Configuration
    MAX_FILE_SIZE: int = config("MAX_FILE_SIZE", default=10485760, cast=int)  # 10MB
    ALLOWED_EXTENSIONS: List[str] = config("ALLOWED_EXTENSIONS", default="jpg,jpeg,png,gif").split(",")
    UPLOAD_DIR: str = config("UPLOAD_DIR", default="uploads")
    
    # ML Model Paths
    DISEASE_MODEL_PATH: str = config("DISEASE_MODEL_PATH", default="../ml_models/trained_models/disease_detection_mobilenetv2_final.h5")
    YIELD_MODEL_PATH: str = config("YIELD_MODEL_PATH", default="../ml_models/trained_models/yield_predictor_xgboost.pkl")
    ALERT_MODEL_PATH: str = config("ALERT_MODEL_PATH", default="../ml_models/trained_models/alert_predictor.pkl")
    
    # External APIs
    WEATHER_API_KEY: Optional[str] = config("WEATHER_API_KEY", default=None)
    WEATHER_API_URL: str = config("WEATHER_API_URL", default="https://api.openweathermap.org/data/2.5")
    
    # Logging Configuration
    LOG_LEVEL: str = config("LOG_LEVEL", default="INFO")
    LOG_FILE: str = config("LOG_FILE", default="logs/greencast.log")
    
    # CORS Configuration
    ALLOWED_ORIGINS: List[str] = config("ALLOWED_ORIGINS", default="http://localhost:3000,http://localhost:8501").split(",")
    
    # Streamlit Configuration
    STREAMLIT_SERVER_PORT: int = config("STREAMLIT_SERVER_PORT", default=8501, cast=int)
    STREAMLIT_SERVER_ADDRESS: str = config("STREAMLIT_SERVER_ADDRESS", default="localhost")
    
    @validator("ALLOWED_EXTENSIONS")
    def validate_extensions(cls, v):
        """Validate file extensions"""
        return [ext.strip().lower() for ext in v]
    
    @validator("UPLOAD_DIR")
    def create_upload_dir(cls, v):
        """Create upload directory if it doesn't exist"""
        os.makedirs(v, exist_ok=True)
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()


class DatabaseConfig:
    """Database configuration and connection settings"""
    
    @staticmethod
    def get_database_url() -> str:
        """Get MongoDB connection URL"""
        return settings.MONGODB_URL
    
    @staticmethod
    def get_database_name() -> str:
        """Get database name"""
        return settings.DATABASE_NAME


class SecurityConfig:
    """Security configuration for authentication and authorization"""
    
    @staticmethod
    def get_secret_key() -> str:
        """Get JWT secret key"""
        return settings.SECRET_KEY
    
    @staticmethod
    def get_algorithm() -> str:
        """Get JWT algorithm"""
        return settings.ALGORITHM
    
    @staticmethod
    def get_token_expire_minutes() -> int:
        """Get token expiration time"""
        return settings.ACCESS_TOKEN_EXPIRE_MINUTES


class MLConfig:
    """Machine Learning model configuration"""
    
    @staticmethod
    def get_disease_model_path() -> str:
        """Get disease detection model path"""
        return settings.DISEASE_MODEL_PATH
    
    @staticmethod
    def get_yield_model_path() -> str:
        """Get yield prediction model path"""
        return settings.YIELD_MODEL_PATH
    
    @staticmethod
    def get_alert_model_path() -> str:
        """Get alert prediction model path"""
        return settings.ALERT_MODEL_PATH
    
    @staticmethod
    def validate_model_paths() -> bool:
        """Validate that all model files exist"""
        paths = [
            settings.DISEASE_MODEL_PATH,
            settings.YIELD_MODEL_PATH,
            settings.ALERT_MODEL_PATH
        ]
        
        for path in paths:
            if not os.path.exists(path):
                return False
        return True


class FileConfig:
    """File upload and processing configuration"""
    
    @staticmethod
    def get_max_file_size() -> int:
        """Get maximum file size in bytes"""
        return settings.MAX_FILE_SIZE
    
    @staticmethod
    def get_allowed_extensions() -> List[str]:
        """Get allowed file extensions"""
        return settings.ALLOWED_EXTENSIONS
    
    @staticmethod
    def get_upload_dir() -> str:
        """Get upload directory path"""
        return settings.UPLOAD_DIR
    
    @staticmethod
    def is_allowed_file(filename: str) -> bool:
        """Check if file extension is allowed"""
        if '.' not in filename:
            return False
        
        extension = filename.rsplit('.', 1)[1].lower()
        return extension in settings.ALLOWED_EXTENSIONS


# Export commonly used configurations
__all__ = [
    "settings",
    "DatabaseConfig", 
    "SecurityConfig",
    "MLConfig",
    "FileConfig"
]
