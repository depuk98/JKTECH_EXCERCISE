from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.core.config import settings
from app.api.routes import api_router
from app.db.session import engine
from app.api.deps import verify_csrf_token
# Import Base directly from base_class for database initialization
# Note: model_registry.py is only needed by Alembic for migrations
from app.db.base_class import Base
from app.db.init_db import init_db_sync

# Initialize database tables with synchronous engine
# This will only be used when running the app directly
# For async engine, the initialization is done in run.py
if "asyncpg" not in str(engine.url) and "aiosqlite" not in str(engine.url):
    init_db_sync(engine)

app = FastAPI(
    title=settings.APP_NAME,
    openapi_url="/api/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="app/templates")

# Include API routes with CSRF protection
api_router.dependencies = [Depends(verify_csrf_token)]
app.include_router(api_router, prefix="/api")

@app.get("/", response_class=RedirectResponse)
async def root():
    """Redirect to the landing page."""
    return RedirectResponse(url="/landing")

@app.get("/landing", include_in_schema=False)
async def landing_page(request: Request):
    """Render the landing page."""
    return templates.TemplateResponse("landing.html", {"request": request})

@app.get("/login", include_in_schema=False)
async def login_page(request: Request):
    """Render the login page."""
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/signup", include_in_schema=False)
async def signup_page(request: Request):
    """Render the signup page."""
    return templates.TemplateResponse("signup.html", {"request": request})

@app.get("/dashboard", include_in_schema=False)
async def dashboard_page(request: Request):
    """Render the dashboard page."""
    import time
    return templates.TemplateResponse("dashboard.html", {"request": request, "now": int(time.time())})

@app.get("/qa", include_in_schema=False)
async def qa_page(request: Request):
    """Render the Q&A page."""
    return templates.TemplateResponse("qa.html", {"request": request})

# Explicit logout route
@app.get("/logout", include_in_schema=False)
async def logout():
    """Handle logout by redirecting to landing page."""
    return RedirectResponse(url="/landing")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True) 