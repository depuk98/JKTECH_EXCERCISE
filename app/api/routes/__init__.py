from fastapi import APIRouter

from app.api.routes import auth, users, documents

# Create API router
api_router = APIRouter()

# Include all routes
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(documents.router, prefix="/documents", tags=["documents"]) 