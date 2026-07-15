from fastapi import APIRouter

from app.schemas.response import HealthResponse, ProductInfo, ProductListResponse

router = APIRouter(tags=["Health"])

PRODUCTS = [
    ProductInfo(id="esp32", name="ESP32"),
    ProductInfo(id="esp32-cam", name="ESP32-CAM"),
    ProductInfo(id="mg90s-servo", name="MG90S Servo"),
    ProductInfo(id="mg996r-servo", name="MG996R Servo"),
    ProductInfo(id="pca9685", name="PCA9685 Servo Driver"),
    ProductInfo(id="l298n", name="L298N Motor Driver"),
    ProductInfo(id="robotic-arm", name="Robotic Arm"),
    ProductInfo(id="warehouse-rover", name="Warehouse Rover"),
]


@router.get("/api/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="healthy")


@router.get("/api/products", response_model=ProductListResponse)
def list_products():
    return ProductListResponse(products=PRODUCTS)
