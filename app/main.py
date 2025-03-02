from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.core.config import settings
from app.api.routes import api_router
from app.db.session import engine
# Import Base directly from base_class for database initialization
# Note: model_registry.py is only needed by Alembic for migrations
from app.db.base_class import Base

# Create database tables
Base.metadata.create_all(bind=engine)

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

# Include API routes
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
    return templates.TemplateResponse("dashboard.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True) 