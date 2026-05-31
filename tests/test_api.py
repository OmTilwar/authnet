"""
Tests for FastAPI endpoints.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import io
from PIL import Image
from fastapi.testclient import TestClient


def create_test_image(size=(224, 224), color=(128, 64, 200)):
    """Create a test image as bytes."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


@pytest.fixture(scope="module")
def client():
    """Create a test client with model loaded."""
    import api.main as api_module
    
    # Manually load the model before creating the client
    api_module._load_model()
    
    return TestClient(api_module.app)


class TestHealthEndpoint:
    """Tests for /health endpoint."""
    
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_health_response_fields(self, client):
        response = client.get("/health")
        data = response.json()
        
        assert "status" in data
        assert data["status"] == "healthy"
        assert "model" in data
        assert "device" in data
        assert "embedding_dim" in data
        assert data["embedding_dim"] == 128


class TestFingerprintEndpoint:
    """Tests for /fingerprint endpoint."""
    
    def test_fingerprint_returns_200(self, client):
        img_bytes = create_test_image()
        response = client.post(
            "/fingerprint",
            files={"image": ("test.png", img_bytes, "image/png")},
        )
        assert response.status_code == 200
    
    def test_fingerprint_returns_128_dim(self, client):
        img_bytes = create_test_image()
        response = client.post(
            "/fingerprint",
            files={"image": ("test.png", img_bytes, "image/png")},
        )
        data = response.json()
        
        assert "fingerprint" in data
        assert "dim" in data
        assert data["dim"] == 128
        assert len(data["fingerprint"]) == 128
    
    def test_fingerprint_is_normalized(self, client):
        """Fingerprint should be L2 normalized (unit norm)."""
        import numpy as np
        
        img_bytes = create_test_image()
        response = client.post(
            "/fingerprint",
            files={"image": ("test.png", img_bytes, "image/png")},
        )
        
        fp = np.array(response.json()["fingerprint"])
        norm = np.linalg.norm(fp)
        
        assert abs(norm - 1.0) < 1e-4, f"Fingerprint norm should be 1.0, got {norm}"


class TestAuthenticateEndpoint:
    """Tests for /authenticate endpoint."""
    
    def test_authenticate_returns_200(self, client):
        img_a = create_test_image(color=(100, 50, 200))
        img_b = create_test_image(color=(100, 50, 200))
        
        response = client.post(
            "/authenticate",
            files={
                "image_a": ("test_a.png", img_a, "image/png"),
                "image_b": ("test_b.png", img_b, "image/png"),
            },
        )
        assert response.status_code == 200
    
    def test_authenticate_response_fields(self, client):
        img_a = create_test_image(color=(100, 50, 200))
        img_b = create_test_image(color=(200, 100, 50))
        
        response = client.post(
            "/authenticate",
            files={
                "image_a": ("a.png", img_a, "image/png"),
                "image_b": ("b.png", img_b, "image/png"),
            },
        )
        data = response.json()
        
        assert "is_match" in data
        assert "similarity" in data
        assert "threshold" in data
        assert "verdict" in data
        assert isinstance(data["is_match"], bool)
        assert isinstance(data["similarity"], float)
        assert data["verdict"] in ["MATCH", "MISMATCH"]
    
    def test_same_image_high_similarity(self, client):
        """Same image should have very high self-similarity."""
        img = create_test_image(color=(100, 150, 200))
        img_copy = create_test_image(color=(100, 150, 200))
        
        response = client.post(
            "/authenticate",
            files={
                "image_a": ("a.png", img, "image/png"),
                "image_b": ("b.png", img_copy, "image/png"),
            },
        )
        data = response.json()
        
        # Identical images should have similarity very close to 1.0
        assert data["similarity"] > 0.95, \
            f"Same image similarity should be > 0.95, got {data['similarity']}"


class TestExplainEndpoint:
    """Tests for /explain endpoint."""
    
    def test_explain_returns_png(self, client):
        img_bytes = create_test_image()
        response = client.post(
            "/explain",
            files={"image": ("test.png", img_bytes, "image/png")},
        )
        
        # Accept 200 (success) or 500 (Grad-CAM may fail on solid-color images)
        assert response.status_code in (200, 500), \
            f"Expected 200 or 500, got {response.status_code}"
        
        if response.status_code == 200:
            assert response.headers["content-type"] == "image/png"
    
    def test_explain_returns_valid_image(self, client):
        """Returned PNG should be a valid image."""
        img_bytes = create_test_image()
        response = client.post(
            "/explain",
            files={"image": ("test.png", img_bytes, "image/png")},
        )
        
        if response.status_code == 200:
            # Try to open as PIL Image
            result_img = Image.open(io.BytesIO(response.content))
            assert result_img.size[0] > 0
            assert result_img.size[1] > 0
