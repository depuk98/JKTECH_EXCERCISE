from typing import Optional
from datetime import datetime
from pydantic import BaseModel, EmailStr, Field, validator

# Shared properties
class UserBase(BaseModel):
    """Base user schema."""
    email: Optional[EmailStr] = None
    username: Optional[str] = None
    age: Optional[int] = 100
    is_active: Optional[bool] = True
    is_superuser: Optional[bool] = False

# Properties to receive via API on creation
class UserCreate(UserBase):
    """User creation schema."""
    email: EmailStr
    username: str
    password: str = Field(..., min_length=8)
    
    @validator('username')
    def username_alphanumeric(cls, v):
        """Validate that username is alphanumeric."""
        assert v.isalnum(), 'Username must be alphanumeric'
        return v

# Properties to receive via API on update
class UserUpdate(UserBase):
    """User update schema."""
    password: Optional[str] = Field(None, min_length=8)

# Properties to return via API
class User(UserBase):
    """User schema."""
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        """Pydantic config."""
        orm_mode = True

# Properties for user authentication
class UserLogin(BaseModel):
    """User login schema."""
    username: str
    password: str

# Token schema
class Token(BaseModel):
    """Token schema."""
    access_token: str
    token_type: str

# Token payload schema
class TokenPayload(BaseModel):
    """Token payload schema."""
    sub: Optional[str] = None 