from sqlalchemy import Boolean, Column, Integer, String, Text, ForeignKey, DateTime, JSON
from sqlalchemy import Boolean, Column, Integer, String, Text, ForeignKey, DateTime
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import ARRAY, FLOAT
from sqlalchemy.orm import relationship
# Import the pgvector type
from pgvector.sqlalchemy import Vector

from app.db.base_class import Base

class Document(Base):
    """Document model for storing uploaded document metadata and extracted text."""
    
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    filename = Column(String, nullable=False)
    content_type = Column(String, nullable=False)
    file_path = Column(String, nullable=True)  # Path to original file if stored locally
    file_size = Column(Integer, nullable=True)  # Size in bytes
    page_count = Column(Integer, nullable=True)  # Number of pages (for PDFs, etc)
    status = Column(String, nullable=False, default="pending")  # pending, processing, processed, error
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="documents")
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Document(id={self.id}, filename={self.filename})>"


class DocumentChunk(Base):
    """Model for storing document chunks with their vector embeddings."""
    
    __tablename__ = "document_chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_index = Column(Integer, nullable=False)  # Position in the document
    text = Column(Text, nullable=False)  # The actual text content of the chunk
    # Use pgvector's Vector type instead of ARRAY(FLOAT)
    embedding = Column(Vector(384), nullable=True)  # Vector embedding with dimension 384
    chunk_metadata = Column(JSON, nullable=True)  # JSON with metadata (page number, section, etc)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    document = relationship("Document", back_populates="chunks")
    
    def __repr__(self):
        return f"<DocumentChunk(id={self.id}, document_id={self.document_id}, chunk_index={self.chunk_index})>" 