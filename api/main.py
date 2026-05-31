"""
AuthNet FastAPI Server
Serves model predictions via REST API with 4 endpoints:
  POST /authenticate — Compare two images for authentication
  POST /fingerprint  — Generate a visual fingerprint for an image
  POST /explain      — Generate Grad-CAM explanation heatmap
  GET  /health       — Health check
"""

import os
import sys
import io
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.model import EmbeddingNet, load_model, build_model
from src.fingerprint import FingerprintEngine
from src.interpretability import generate_heatmap
from src.dataset import get_test_transforms


# ── App Setup ──
from contextlib import asynccontextmanager

# ── Global State ──
model: Optional[EmbeddingNet] = None
engine: Optional[FingerprintEngine] = None
transform = get_test_transforms()


def _load_model():
    """Load the trained model."""
    global model, engine
    
    if os.path.exists(config.BEST_MODEL_PATH):
        print(f"Loading model from {config.BEST_MODEL_PATH}...")
        model = load_model(config.BEST_MODEL_PATH, config.DEVICE)
    else:
        print("No trained model found. Using randomly initialized model for demo.")
        model = build_model(config.DEVICE)
    
    engine = FingerprintEngine(model=model, device=config.DEVICE)
    print("AuthNet API ready!")


@asynccontextmanager
async def lifespan(app):
    """Load model on startup, clean up on shutdown."""
    _load_model()
    yield


app = FastAPI(
    title="AuthNet API",
    description=(
        "Product Authentication & Visual Fingerprinting API. "
        "Uses deep metric learning to authenticate products and generate "
        "unique visual fingerprints from texture analysis."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# ── Response Schemas ──
class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    embedding_dim: int


class FingerprintResponse(BaseModel):
    fingerprint: list
    dim: int


class AuthResponse(BaseModel):
    is_match: bool
    similarity: float
    threshold: float
    verdict: str


# ── Helper ──
async def read_image(file: UploadFile) -> Image.Image:
    """Read an uploaded file into a PIL Image."""
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")


# ── Endpoints ──

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model=config.BACKBONE,
        device=str(config.DEVICE),
        embedding_dim=config.EMBEDDING_DIM,
    )


@app.post("/fingerprint", response_model=FingerprintResponse)
async def create_fingerprint(image: UploadFile = File(...)):
    """
    Generate a unique 128-dim visual fingerprint for an uploaded image.
    
    The fingerprint is an L2-normalized embedding vector that serves as 
    the item's unique digital identity based on its visual characteristics.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    pil_image = await read_image(image)
    fingerprint = engine.create_fingerprint(pil_image)
    
    return FingerprintResponse(
        fingerprint=fingerprint.tolist(),
        dim=len(fingerprint),
    )


@app.post("/authenticate", response_model=AuthResponse)
async def authenticate_pair(
    image_a: UploadFile = File(...),
    image_b: UploadFile = File(...),
):
    """
    Authenticate two images by comparing their visual fingerprints.
    
    Returns a similarity score and verdict:
    - MATCH: similarity >= threshold (images depict the same item)
    - MISMATCH: similarity < threshold (different items)
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    pil_a = await read_image(image_a)
    pil_b = await read_image(image_b)
    
    result = engine.verify_pair(pil_a, pil_b)
    
    return AuthResponse(
        is_match=result['is_match'],
        similarity=result['similarity'],
        threshold=result['threshold'],
        verdict=result['verdict'],
    )


@app.post("/explain")
async def explain_prediction(image: UploadFile = File(...)):
    """
    Generate a Grad-CAM heatmap explanation for an image.
    
    Returns a PNG image showing which regions the model focuses on
    when extracting features for authentication/fingerprinting.
    This demonstrates model interpretability.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    pil_image = await read_image(image)
    
    try:
        original, heatmap, overlay = generate_heatmap(model, pil_image)
        
        # Convert overlay to PNG bytes
        overlay_pil = Image.fromarray(overlay)
        buf = io.BytesIO()
        overlay_pil.save(buf, format="PNG")
        buf.seek(0)
        
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Heatmap generation failed: {e}")


# ── Main ──
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
