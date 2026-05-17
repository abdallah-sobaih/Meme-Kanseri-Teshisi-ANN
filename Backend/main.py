"""
Breast Cancer Diagnosis API
============================
FastAPI backend serving a PyTorch model (Architecture M5) for
binary classification of breast cancer tumors (Benign / Malignant).

Usage:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Dependencies:
    pip install fastapi uvicorn torch pydantic
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Annotated

import joblib
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
)
logger = logging.getLogger("breast_cancer_api")

# ---------------------------------------------------------------------------
# Model Architecture  (must match the saved .pth file exactly — M5)
# ---------------------------------------------------------------------------

class BreastCancerModel(nn.Module):
    """
    Architecture M5 — exactly mirrors the training architecture so that
    the saved state-dict loads without key mismatches.
    """

    def __init__(self) -> None:
        super(BreastCancerModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(30, 64),   nn.BatchNorm1d(64),  nn.ELU(), nn.Dropout(0.15),
            nn.Linear(64, 32),   nn.BatchNorm1d(32),  nn.ELU(), nn.Dropout(0.15),
            nn.Linear(32, 16),   nn.BatchNorm1d(16),  nn.ELU(), nn.Dropout(0.15),
            nn.Linear(16, 1),    nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ---------------------------------------------------------------------------
# Global model state (loaded once at startup via lifespan)
# ---------------------------------------------------------------------------

MODEL_PATH = "en_iyi_meme_kanseri_modeli.pth"
_model: BreastCancerModel | None = None
_device: torch.device = torch.device("cpu")


def _load_model() -> BreastCancerModel:
    """Load and return the model in evaluation mode."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file '{MODEL_PATH}' not found. "
            "Place it in the same directory as main.py."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading model onto device: %s", device)

    model = BreastCancerModel().to(device)
    state = torch.load(MODEL_PATH, map_location=device)

    # Support both raw state-dicts and checkpoint dicts
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    model.load_state_dict(state)
    model.eval()
    logger.info("Model loaded successfully.")
    return model, device


# ---------------------------------------------------------------------------
# Lifespan context (replaces deprecated on_event handlers)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _device
    _model, _device = _load_model()
    yield
    # Teardown (nothing needed for inference-only server)


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Breast Cancer Diagnosis API",
    description=(
        "Binary classification of breast tumors using a trained PyTorch model "
        "(Architecture M5). Input: 30 numerical features from the Wisconsin "
        "Breast Cancer dataset. Output: Benign / Malignant + confidence."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow the frontend (any origin during development) to reach this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Restrict to your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend from the same directory
if os.path.exists("index.html"):
    @app.get("/", response_class=FileResponse)
    async def serve_frontend():
        return FileResponse("index.html")


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class PredictionRequest(BaseModel):
    """
    30 features from the Wisconsin Breast Cancer (Diagnostic) dataset.
    All values must be finite real numbers.
    """

    # ── Mean values ──────────────────────────────────────────────────────────
    mean_radius:             float = Field(..., description="Mean of distances from center to points on the perimeter")
    mean_texture:            float = Field(..., description="Standard deviation of gray-scale values")
    mean_perimeter:          float = Field(..., description="Mean size of the core tumor")
    mean_area:               float = Field(..., description="Mean area of the tumor")
    mean_smoothness:         float = Field(..., description="Mean of local variation in radius lengths")
    mean_compactness:        float = Field(..., description="Mean of perimeter² / area − 1.0")
    mean_concavity:          float = Field(..., description="Mean severity of concave portions of the contour")
    mean_concave_points:     float = Field(..., description="Mean number of concave portions of the contour")
    mean_symmetry:           float = Field(..., description="Mean symmetry")
    mean_fractal_dimension:  float = Field(..., description="Mean 'coastline approximation' − 1")

    # ── Standard error values ─────────────────────────────────────────────────
    se_radius:               float = Field(..., description="Standard error: radius")
    se_texture:              float = Field(..., description="Standard error: texture")
    se_perimeter:            float = Field(..., description="Standard error: perimeter")
    se_area:                 float = Field(..., description="Standard error: area")
    se_smoothness:           float = Field(..., description="Standard error: smoothness")
    se_compactness:          float = Field(..., description="Standard error: compactness")
    se_concavity:            float = Field(..., description="Standard error: concavity")
    se_concave_points:       float = Field(..., description="Standard error: concave points")
    se_symmetry:             float = Field(..., description="Standard error: symmetry")
    se_fractal_dimension:    float = Field(..., description="Standard error: fractal dimension")

    # ── Worst (largest) values ────────────────────────────────────────────────
    worst_radius:            float = Field(..., description="Worst: radius")
    worst_texture:           float = Field(..., description="Worst: texture")
    worst_perimeter:         float = Field(..., description="Worst: perimeter")
    worst_area:              float = Field(..., description="Worst: area")
    worst_smoothness:        float = Field(..., description="Worst: smoothness")
    worst_compactness:       float = Field(..., description="Worst: compactness")
    worst_concavity:         float = Field(..., description="Worst: concavity")
    worst_concave_points:    float = Field(..., description="Worst: concave points")
    worst_symmetry:          float = Field(..., description="Worst: symmetry")
    worst_fractal_dimension: float = Field(..., description="Worst: fractal dimension")

    @field_validator("*", mode="before")
    @classmethod
    def must_be_finite(cls, v):
        import math
        if isinstance(v, (int, float)) and not math.isfinite(v):
            raise ValueError("Feature values must be finite real numbers.")
        return v

    def to_tensor(self) -> torch.Tensor:
        """Return a (1, 30) float32 tensor in the canonical feature order."""
        values = [
            self.mean_radius,            self.mean_texture,
            self.mean_perimeter,         self.mean_area,
            self.mean_smoothness,        self.mean_compactness,
            self.mean_concavity,         self.mean_concave_points,
            self.mean_symmetry,          self.mean_fractal_dimension,
            self.se_radius,              self.se_texture,
            self.se_perimeter,           self.se_area,
            self.se_smoothness,          self.se_compactness,
            self.se_concavity,           self.se_concave_points,
            self.se_symmetry,            self.se_fractal_dimension,
            self.worst_radius,           self.worst_texture,
            self.worst_perimeter,        self.worst_area,
            self.worst_smoothness,       self.worst_compactness,
            self.worst_concavity,        self.worst_concave_points,
            self.worst_symmetry,         self.worst_fractal_dimension,
        ]
        return torch.tensor([values], dtype=torch.float32)


class PredictionResponse(BaseModel):
    diagnosis:   str   = Field(..., description="'Benign' or 'Malignant'")
    confidence:  float = Field(..., description="Model confidence in [0.0, 100.0]")
    probability: float = Field(..., description="Raw sigmoid output in [0.0, 1.0]")
    label:       int   = Field(..., description="0 = Benign, 1 = Malignant")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Utility"])
async def health_check():
    """Simple liveness probe."""
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Run the breast cancer classification model on 30 input features.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    # 1. تحويل البيانات القادمة من الواجهة إلى مصفوفة (Numpy Array)
    raw_data = request.to_tensor().numpy()
    
    # 2. تحميل الـ Scaler لتصغير الأرقام (لكي لا ينفجر الموديل)
    try:
        scaler = joblib.load("scaler.pkl")
        scaled_data = scaler.transform(raw_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scaler Error: {str(e)}")

    # 3. إرسال البيانات المصغرة للموديل
    tensor = torch.tensor(scaled_data, dtype=torch.float32).to(_device)

    with torch.no_grad():
        probability: float = _model(tensor).item()   # scalar in [0, 1]

    label      = int(probability >= 0.5)
    diagnosis  = "Malignant (خبيث)" if label == 1 else "Benign (حميد)"

    # حساب نسبة الثقة بشكل دقيق
    confidence = (abs(probability - 0.5) * 2) * 100
    confidence = max(0.0, min(100.0, confidence))

    logger.info(
        "Prediction: %s (p=%.4f, confidence=%.1f%%)",
        diagnosis, probability, confidence,
    )

    return PredictionResponse(
        diagnosis=diagnosis,
        confidence=round(confidence, 2),
        probability=round(probability, 6),
        label=label,
    )