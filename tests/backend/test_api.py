from fastapi.testclient import TestClient
from starlette.routing import BaseRoute

from app.main import app

client = TestClient(app)


def _all_paths(routes: list[BaseRoute]) -> list[str]:
    paths = []
    for r in routes:
        if hasattr(r, "path") and isinstance(getattr(r, "path", None), str):
            paths.append(r.path)
        if hasattr(r, "routes"):
            paths.extend(_all_paths(getattr(r, "routes", [])))
        if hasattr(r, "original_router"):
            paths.extend(_all_paths(
                getattr(r.original_router, "routes", [])
            ))
    return paths


class TestRoot:
    def test_root_returns_status(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data


class TestHealth:
    def test_health_returns_healthy(self):
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


class TestProducts:
    def test_products_returns_list(self):
        response = client.get("/api/products")
        assert response.status_code == 200
        data = response.json()
        assert "products" in data
        assert len(data["products"]) == 8

    def test_products_contain_expected_entries(self):
        response = client.get("/api/products")
        products = response.json()["products"]
        names = [p["name"] for p in products]
        assert "ESP32" in names
        assert "Warehouse Rover" in names
        assert "MG90S Servo" in names


class TestAnalyse:
    def test_valid_analysis(self):
        response = client.post("/api/analyse", json={
            "question": "Why is my servo overheating?",
            "product": "MG90S Servo",
            "assistance_type": "troubleshooting",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["intent"] == "troubleshooting"
        assert data["product"] == "MG90S Servo"
        assert data["summary"]
        assert isinstance(data["possible_causes"], list)
        assert isinstance(data["steps"], list)
        assert "warning" in data
        assert "escalation_required" in data
        assert "sources" in data

    def test_valid_analysis_default_assistance_type(self):
        response = client.post("/api/analyse", json={
            "question": "How do I set up the ESP32?",
            "product": "ESP32",
        })
        assert response.status_code == 200

    def test_empty_question_returns_422(self):
        response = client.post("/api/analyse", json={
            "question": "",
            "product": "ESP32",
        })
        assert response.status_code == 422

    def test_empty_product_returns_422(self):
        response = client.post("/api/analyse", json={
            "question": "How do I fix this?",
            "product": "",
        })
        assert response.status_code == 422

    def test_invalid_assistance_type_returns_422(self):
        response = client.post("/api/analyse", json={
            "question": "Is my motor working?",
            "product": "L298N Motor Driver",
            "assistance_type": "invalid_type",
        })
        assert response.status_code == 422

    def test_missing_question_returns_422(self):
        response = client.post("/api/analyse", json={
            "product": "ESP32",
        })
        assert response.status_code == 422

    def test_only_one_analyse_route_exists(self):
        paths = _all_paths(app.routes)
        assert paths.count("/api/analyse") == 1
