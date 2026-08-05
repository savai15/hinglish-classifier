"""
FastAPI Backend for Hinglish Complaint Classifier
REST API for predictions, history, stats, and feedback.
"""
import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

from src.active_learning import ComplaintPredictor, ComplaintDB, RetrainManager

predictor = None
db = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor, db
    print("Loading models...")
    predictor = ComplaintPredictor(use_muril=False)
    predictor.load_models()
    db = ComplaintDB()
    print("Models loaded! API ready.")
    yield
    print("Shutting down...")


app = FastAPI(
    title="Hinglish Complaint Classifier API",
    description="Classify Hinglish e-commerce complaints into 9 categories and 3 urgency levels",
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


# ============================================================================
# SCHEMAS
# ============================================================================

class PredictionRequest(BaseModel):
    text: str
    session_id: Optional[str] = None


class BatchPredictionRequest(BaseModel):
    texts: List[str]
    session_id: Optional[str] = None


class CorrectionRequest(BaseModel):
    prediction_id: int
    is_correct_category: bool = True
    is_correct_urgency: bool = True
    corrected_category: Optional[str] = None
    corrected_urgency: Optional[str] = None


class PredictionResponse(BaseModel):
    id: int
    text: str
    cleaned_text: str
    category: str
    category_confidence: float
    category_probabilities: dict
    urgency: str
    urgency_confidence: float
    urgency_probabilities: dict
    source: str
    needs_review: bool


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
def root():
    return {
        "message": "Hinglish Complaint Classifier API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "predict": "POST /predict",
            "batch_predict": "POST /predict/batch",
            "history": "GET /history",
            "stats": "GET /stats",
            "feedback": "POST /feedback",
            "retrain": "POST /retrain",
            "health": "GET /health",
        },
    }


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if not predictor:
        raise HTTPException(status_code=503, detail="Models not loaded")
    result = predictor.predict(request.text, session_id=request.session_id)
    return result


@app.post("/predict/batch")
def predict_batch(request: BatchPredictionRequest):
    if not predictor:
        raise HTTPException(status_code=503, detail="Models not loaded")
    results = []
    for text in request.texts:
        result = predictor.predict(text, session_id=request.session_id)
        results.append(result)
    return {"predictions": results, "count": len(results)}


@app.get("/history")
def get_history(limit: int = 20):
    if not db:
        raise HTTPException(status_code=503, detail="Database not initialized")
    df = db.get_recent_predictions(limit)
    if len(df) == 0:
        return {"predictions": [], "count": 0}
    return {
        "predictions": df.to_dict(orient="records"),
        "count": len(df),
    }


@app.get("/stats")
def get_stats():
    if not db:
        raise HTTPException(status_code=503, detail="Database not initialized")
    return db.get_stats()


@app.post("/feedback")
def submit_feedback(request: CorrectionRequest):
    if not predictor:
        raise HTTPException(status_code=503, detail="Models not loaded")
    predictor.submit_correction(
        pred_id=request.prediction_id,
        is_correct_cat=request.is_correct_category,
        is_correct_urg=request.is_correct_urgency,
        corrected_cat=request.corrected_category,
        corrected_urg=request.corrected_urgency,
    )
    retrain_mgr = RetrainManager()
    should_retrain, correction_count = retrain_mgr.should_retrain()
    return {
        "status": "ok",
        "corrections_total": correction_count,
        "should_retrain": should_retrain,
        "message": f"Feedback recorded. {correction_count} corrections total.",
    }


@app.post("/retrain")
def trigger_retrain():
    retrain_mgr = RetrainManager()
    should_retrain, correction_count = retrain_mgr.should_retrain()
    if not should_retrain:
        return {
            "status": "skipped",
            "message": f"Not enough corrections yet ({correction_count}/{20}). Need {20 - correction_count} more.",
        }

    try:
        accuracy = retrain_mgr.retrain_sklearn()
        return {
            "status": "success",
            "accuracy": accuracy,
            "corrections_used": correction_count,
            "message": f"Retraining complete! Accuracy: {accuracy:.4f}",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")


@app.get("/categories")
def get_categories():
    return {
        "categories": [
            "Account_Technical",
            "Customer_Service",
            "Delivery_Issue",
            "Order_Status",
            "Payment_Invoice",
            "Pricing_Discount",
            "Product_Quality",
            "Returns_Refunds",
            "Wrong_Damaged_Product",
        ],
        "urgency_levels": ["High", "Medium", "Low"],
    }


@app.get("/low-confidence")
def get_low_confidence(limit: int = 20):
    if not db:
        raise HTTPException(status_code=503, detail="Database not initialized")
    df = db.get_low_confidence_predictions()
    if len(df) == 0:
        return {"predictions": [], "count": 0}
    return {
        "predictions": df.head(limit).to_dict(orient="records"),
        "count": len(df),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
