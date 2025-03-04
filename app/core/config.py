import os
from typing import List, Union, Dict, Any
from pydantic import AnyHttpUrl, validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    """Application settings."""
    
    # Application settings
    APP_NAME: str = "FastAPI User Management"
    DEBUG: bool = False
    ENVIRONMENT: str = "production"
    
    # Security settings
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Database settings
    DATABASE_URL: str
    
    # CORS settings
    CORS_ORIGINS: List[str] = ["http://localhost:8000", "http://localhost:3000"]
    
    @validator("CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        """Parse CORS origins from string to list."""
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    # Logging settings
    LOG_LEVEL: str = "INFO"
    
    class Config:
        """Pydantic config."""
        case_sensitive = True
        env_file = ".env"

# Create settings instance
settings = Settings() 