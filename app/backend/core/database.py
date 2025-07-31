"""
Database connection and initialization for GreenCast application
"""

import asyncio
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
from loguru import logger

from backend.core.config import DatabaseConfig
from backend.models.database import DOCUMENT_MODELS


class DatabaseManager:
    """Database connection manager"""
    
    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.database = None
        
    async def connect_to_database(self) -> None:
        """Create database connection"""
        try:
            # Create MongoDB client
            self.client = AsyncIOMotorClient(
                DatabaseConfig.get_database_url(),
                maxPoolSize=10,
                minPoolSize=1,
                maxIdleTimeMS=45000,
                serverSelectionTimeoutMS=5000,
            )
            
            # Get database
            self.database = self.client[DatabaseConfig.get_database_name()]
            
            # Test connection
            await self.client.admin.command('ping')
            logger.info(f"Connected to MongoDB: {DatabaseConfig.get_database_name()}")
            
            # Initialize Beanie with document models
            await init_beanie(
                database=self.database,
                document_models=DOCUMENT_MODELS
            )
            logger.info("Beanie ODM initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    async def close_database_connection(self) -> None:
        """Close database connection"""
        if self.client:
            self.client.close()
            logger.info("Database connection closed")
    
    async def ping_database(self) -> bool:
        """Ping database to check connection"""
        try:
            if self.client:
                await self.client.admin.command('ping')
                return True
            return False
        except Exception as e:
            logger.error(f"Database ping failed: {e}")
            return False
    
    async def get_database_info(self) -> dict:
        """Get database information"""
        try:
            if not self.database:
                return {}
            
            # Get database stats
            stats = await self.database.command("dbStats")
            
            # Get collection info
            collections = await self.database.list_collection_names()
            
            return {
                "database_name": self.database.name,
                "collections": collections,
                "collections_count": len(collections),
                "data_size": stats.get("dataSize", 0),
                "storage_size": stats.get("storageSize", 0),
                "indexes": stats.get("indexes", 0),
                "objects": stats.get("objects", 0),
            }
        except Exception as e:
            logger.error(f"Failed to get database info: {e}")
            return {}
    
    async def create_indexes(self) -> None:
        """Create additional database indexes for performance"""
        try:
            # Create compound indexes for common queries
            await self.database.disease_detections.create_index([
                ("user_id", 1),
                ("created_at", -1)
            ])
            
            await self.database.yield_predictions.create_index([
                ("user_id", 1),
                ("field_id", 1),
                ("prediction_date", -1)
            ])
            
            await self.database.alerts.create_index([
                ("user_id", 1),
                ("is_read", 1),
                ("severity", 1),
                ("created_at", -1)
            ])
            
            await self.database.field_log_entries.create_index([
                ("user_id", 1),
                ("field_id", 1),
                ("created_at", -1)
            ])
            
            # Create text indexes for search functionality
            await self.database.disease_detections.create_index([
                ("predicted_disease", "text"),
                ("notes", "text")
            ])
            
            await self.database.field_log_entries.create_index([
                ("title", "text"),
                ("description", "text"),
                ("tags", "text")
            ])
            
            logger.info("Database indexes created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create indexes: {e}")
    
    async def cleanup_expired_data(self) -> None:
        """Clean up expired data from database"""
        try:
            from datetime import datetime, timedelta
            
            # Remove expired alerts
            expired_date = datetime.utcnow()
            result = await self.database.alerts.delete_many({
                "expires_at": {"$lt": expired_date}
            })
            
            if result.deleted_count > 0:
                logger.info(f"Cleaned up {result.deleted_count} expired alerts")
            
            # Remove old system metrics (keep last 30 days)
            old_date = datetime.utcnow() - timedelta(days=30)
            result = await self.database.system_metrics.delete_many({
                "timestamp": {"$lt": old_date}
            })
            
            if result.deleted_count > 0:
                logger.info(f"Cleaned up {result.deleted_count} old metrics")
                
        except Exception as e:
            logger.error(f"Failed to cleanup expired data: {e}")


# Global database manager instance
db_manager = DatabaseManager()


async def get_database():
    """Dependency to get database instance"""
    return db_manager.database


async def init_database():
    """Initialize database connection"""
    await db_manager.connect_to_database()
    await db_manager.create_indexes()


async def close_database():
    """Close database connection"""
    await db_manager.close_database_connection()


# Database health check
async def check_database_health() -> dict:
    """Check database health status"""
    try:
        is_connected = await db_manager.ping_database()
        db_info = await db_manager.get_database_info()
        
        return {
            "status": "healthy" if is_connected else "unhealthy",
            "connected": is_connected,
            "database_info": db_info,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "connected": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


# Utility functions for common database operations
async def get_user_by_email(email: str):
    """Get user by email address"""
    from backend.models.database import User
    return await User.find_one(User.email == email)


async def get_user_by_id(user_id: str):
    """Get user by ID"""
    from backend.models.database import User
    return await User.get(user_id)


async def get_user_fields(user_id: str):
    """Get all fields for a user"""
    from backend.models.database import Field
    return await Field.find(Field.user_id == user_id).to_list()


async def get_user_alerts(user_id: str, unread_only: bool = False):
    """Get alerts for a user"""
    from backend.models.database import Alert
    
    query = Alert.user_id == user_id
    if unread_only:
        query = query & (Alert.is_read == False)
    
    return await Alert.find(query).sort(-Alert.created_at).to_list()


async def get_recent_predictions(user_id: str, limit: int = 10):
    """Get recent predictions for a user"""
    from backend.models.database import DiseaseDetection, YieldPrediction
    
    # Get recent disease detections
    disease_predictions = await DiseaseDetection.find(
        DiseaseDetection.user_id == user_id
    ).sort(-DiseaseDetection.created_at).limit(limit).to_list()
    
    # Get recent yield predictions
    yield_predictions = await YieldPrediction.find(
        YieldPrediction.user_id == user_id
    ).sort(-YieldPrediction.created_at).limit(limit).to_list()
    
    return {
        "disease_predictions": disease_predictions,
        "yield_predictions": yield_predictions
    }


# Database migration utilities
async def migrate_database():
    """Run database migrations"""
    try:
        logger.info("Starting database migration...")
        
        # Add any migration logic here
        # For example, updating existing documents with new fields
        
        logger.info("Database migration completed successfully")
        
    except Exception as e:
        logger.error(f"Database migration failed: {e}")
        raise


# Export commonly used functions
__all__ = [
    "db_manager",
    "init_database",
    "close_database",
    "get_database",
    "check_database_health",
    "get_user_by_email",
    "get_user_by_id",
    "get_user_fields",
    "get_user_alerts",
    "get_recent_predictions",
    "migrate_database"
]
