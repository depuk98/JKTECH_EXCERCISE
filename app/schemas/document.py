from typing import Any, List, Optional, Dict, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator, field_validator
import json

# ---------------
# Request Schemas
# ---------------

class DocumentCreate(BaseModel):
    """Schema for creating a document (used internally)."""
    user_id: int
    filename: str
    content_type: str
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    page_count: Optional[int] = None
    status: Optional[str] = "pending"


class DocumentUpdate(BaseModel):
    """Schema for updating document status and metadata."""
    status: Optional[str] = None
    page_count: Optional[int] = None


class DocumentChunkCreate(BaseModel):
    """Schema for creating a document chunk (used internally)."""
    document_id: int
    chunk_index: int
    text: str
    embedding: Optional[List[float]] = None
    chunk_metadata: Optional[Dict[str, Any]] = None
    
    @field_validator('chunk_metadata')
    @classmethod
    def serialize_metadata(cls, v):
        """Convert metadata dict to JSON string if it's a dict."""
        if isinstance(v, dict):
            return json.dumps(v)
        return v


# ---------------
# Response Schemas
# ---------------

class DocumentChunk(BaseModel):
    """Schema for returning a document chunk."""
    id: int
    document_id: int
    chunk_index: int
    text: str
    chunk_metadata: Optional[Dict[str, Any]] = None
    created_at: datetime
    
    @field_validator('chunk_metadata')
    @classmethod
    def deserialize_metadata(cls, v):
        """Convert JSON string to dict if it's a string."""
        if v and isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return {}
        return v or {}
    
    class Config:
        from_attributes = True


class Document(BaseModel):
    """Schema for returning a document."""
    id: int
    user_id: int
    filename: str
    content_type: str
    file_size: Optional[int] = None
    page_count: Optional[int] = None
    status: str
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    chunks: Optional[List[DocumentChunk]] = []
    
    class Config:
        from_attributes = True


class DocumentList(BaseModel):
    """Schema for returning a list of documents."""
    total: int
    documents: List[Document] 