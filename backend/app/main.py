import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.analysis import router as analysis_router
from app.api.routes.documents import router as documents_router
from app.core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

settings.validate()

app = FastAPI(
    title="Product AI Assistance API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analysis_router)
app.include_router(documents_router)


_PRODUCTS = [
    "ESP32",
    "ESP32-CAM",
    "MG90S Servo",
    "MG996R Servo",
    "PCA9685 Servo Driver",
    "L298N Motor Driver",
    "Robotic Arm",
    "Warehouse Rover",
]


@app.get("/")
async def root():
    return {
        "message": "Product AI Assistance API is running",
        "version": "1.0.0",
    }


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
    }


@app.get("/api/products")
async def list_products():
    return {
        "products": [{"name": p} for p in _PRODUCTS],
    }
