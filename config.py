"""
Configuration settings for Aurora QA System
All values are loaded from .env file
"""

from functools import lru_cache
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings - all loaded from .env file"""
    
    # OpenAI API Configuration
    OPENAI_API_KEY: str
    
    # Aurora API Configuration
    AURORA_API_URL: str
    
    # Server Configuration
    PORT: int = 8000
    DEBUG: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()

