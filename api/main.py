"""
FastAPI Backend for Hinglish Complaint Classifier
REST API for predictions, analytics, batch processing, and feedback.
"""
import os
import sys
import io
import csv
import json
import uuid
import time
import html
import logging
import sqlite3
from pathlib import Path
from contextlib import asynccontextmanager
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, Query, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from datetime import datetime

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

import numpy as np
import requests as http_requests
from src.active_learning import ComplaintPredictor, ComplaintDB, RetrainManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

predictor = None
db = None
compare_models = {}
muril_trainer_instance = None

CATEGORIES = ["Account_Technical", "Customer_Service", "Delivery_Issue", "Order_Status", "Payment_Invoice", "Pricing_Discount", "Product_Quality", "Returns_Refunds", "Wrong_Damaged_Product"]
URGENCY_LEVELS = ["High", "Medium", "Low"]


def groq_request_with_retry(payload, api_key, max_retries=3):
    """Make Groq API request with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            resp = http_requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=15,
            )
            if resp.status_code == 429:
                import time as _time
                _time.sleep(2 ** attempt)
                continue
            return resp
        except http_requests.exceptions.RequestException:
            if attempt < max_retries - 1:
                import time as _time
                _time.sleep(2 ** attempt)
            else:
                raise
    return resp


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor, db, compare_models, muril_trainer_instance
    logger.info("Loading models...")
    predictor = ComplaintPredictor(use_muril=True)
    predictor.load_models()
    db = ComplaintDB()

    # Pre-load individual models for /predict/compare endpoint
    from src.models import load_model
    MODEL_DIR = Path(__file__).parent.parent / "models"
    compare_models = {}
    models_config = {
        'tfidf_svm': {'cat': MODEL_DIR / 'category_tf-idf__svm.pkl', 'urg': MODEL_DIR / 'urgency_tf-idf__svm.pkl', 'label': 'TF-IDF + SVM'},
        'tfidf_lr': {'cat': MODEL_DIR / 'category_tf-idf__lr.pkl', 'urg': MODEL_DIR / 'urgency_tf-idf__lr.pkl', 'label': 'TF-IDF + LR'},
        'ensemble': {'cat': MODEL_DIR / 'category_ensemble.pkl', 'urg': MODEL_DIR / 'urgency_ensemble.pkl', 'label': 'Combined Ensemble'},
    }
    for name, config in models_config.items():
        try:
            compare_models[name] = {
                'label': config['label'],
                'cat_model': load_model(config['cat']),
                'urg_model': load_model(config['urg']),
            }
        except Exception as e:
            logger.warning(f"Failed to load compare model {name}: {e}")

    # Load MuRIL for compare if available
    muril_cat_dir = MODEL_DIR / "muril_classifier" / "category"
    muril_urg_dir = MODEL_DIR / "muril_classifier" / "urgency"
    muril_trainer_instance = None
    if muril_cat_dir.exists() and muril_urg_dir.exists():
        try:
            from src.muril_trainer import MurilTrainer
            muril_cat = MurilTrainer.load(muril_cat_dir)
            muril_urg = MurilTrainer.load(muril_urg_dir)
            muril_trainer_instance = {'cat': muril_cat, 'urg': muril_urg, 'label': 'MuRIL (GPU)'}
            logger.info("MuRIL models loaded for comparison")
        except Exception as e:
            logger.warning(f"Failed to load MuRIL for compare: {e}")

    logger.info("Models loaded! API ready.")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Hinglish Complaint Classifier API",
    description="Classify Hinglish e-commerce complaints into 9 categories and 3 urgency levels",
    version="1.0.0",
    lifespan=lifespan,
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    request_id = getattr(request.state, 'request_id', 'unknown')
    response = await call_next(request)
    duration = round((time.time() - start) * 1000)
    logger.info(f"{request.method} {request.url.path} -> {response.status_code} ({duration}ms) [{request_id}]")
    return response


# ============================================================================
# SCHEMAS
# ============================================================================

class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Complaint text to classify")
    session_id: Optional[str] = None
    model: Optional[str] = Field(default="auto", description="Model to use: 'auto', 'muril', or 'sklearn'")


class BatchPredictionRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=100, description="List of complaints to classify")
    session_id: Optional[str] = None
    model: Optional[str] = Field(default="auto", description="Model to use: 'auto', 'muril', or 'sklearn'")


class CorrectionRequest(BaseModel):
    prediction_id: int
    is_correct_category: bool = True
    is_correct_urgency: bool = True
    corrected_category: Optional[Literal[tuple(CATEGORIES)]] = None
    corrected_urgency: Optional[Literal[tuple(URGENCY_LEVELS)]] = None


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
# CORE ENDPOINTS
# ============================================================================

@app.get("/")
@limiter.exempt
def root():
    return {
        "message": "Hinglish Complaint Classifier API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health")
@limiter.exempt
async def health():
    health_status = {"status": "ok", "timestamp": datetime.now().isoformat()}

    try:
        if db:
            conn = sqlite3.connect(db.db_path)
            conn.execute("SELECT 1")
            conn.close()
            health_status["database"] = "ok"
        else:
            health_status["database"] = "not initialized"
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["database"] = f"error: {str(e)}"
        health_status["status"] = "degraded"

    try:
        if predictor:
            health_status["model"] = "loaded"
        else:
            health_status["model"] = "not loaded"
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["model"] = f"error: {str(e)}"

    groq_key = os.environ.get("GROQ_API_KEY", "")
    health_status["groq_api"] = "configured" if groq_key else "not configured"

    return health_status


@app.get("/categories")
@limiter.exempt
def get_categories():
    return {
        "categories": [
            "Account_Technical", "Customer_Service", "Delivery_Issue",
            "Order_Status", "Payment_Invoice", "Pricing_Discount",
            "Product_Quality", "Returns_Refunds", "Wrong_Damaged_Product",
        ],
        "urgency_levels": ["High", "Medium", "Low"],
        "category_colors": {
            "Account_Technical": "#6366f1",
            "Customer_Service": "#8b5cf6",
            "Delivery_Issue": "#22d3ee",
            "Order_Status": "#14b8a6",
            "Payment_Invoice": "#f59e0b",
            "Pricing_Discount": "#f97316",
            "Product_Quality": "#ef4444",
            "Returns_Refunds": "#ec4899",
            "Wrong_Damaged_Product": "#e879f9",
        },
    }


# ============================================================================
# PREDICTION ENDPOINTS
# ============================================================================

@app.post("/predict", response_model=PredictionResponse)
@limiter.limit("30/minute")
def predict(body: PredictionRequest, request: Request):
    if not predictor:
        raise HTTPException(status_code=503, detail="Models not loaded")
    result = predictor.predict(body.text, session_id=body.session_id, model=body.model)
    return result


@app.post("/predict/batch")
@limiter.limit("10/minute")
def predict_batch(body: BatchPredictionRequest, request: Request):
    if not predictor:
        raise HTTPException(status_code=503, detail="Models not loaded")
    results = []
    for text in body.texts:
        result = predictor.predict(text, session_id=body.session_id, model=body.model)
        results.append(result)
    return {"predictions": results, "count": len(results)}


@app.post("/predict/compare")
@limiter.limit("20/minute")
async def compare_predictions(request: Request, req: PredictionRequest):
    """Run all 3 models and return individual + ensemble predictions."""
    if not predictor:
        raise HTTPException(status_code=503, detail="Models not loaded")

    start = time.time()
    cleaned = predictor.sklearn_preprocessor.preprocess(req.text)

    results = {}
    for name, model_info in compare_models.items():
        try:
            cat_model = model_info['cat_model']
            urg_model = model_info['urg_model']

            cat_pred, cat_conf, cat_probs_dict = None, 0.0, {}
            if hasattr(cat_model, 'predict_proba'):
                probs = cat_model.predict_proba([cleaned])[0]
                classes = cat_model.classes_
                cat_pred = classes[np.argmax(probs)]
                cat_conf = float(np.max(probs))
                cat_probs_dict = {c: round(float(p), 4) for c, p in zip(classes, probs)}
            else:
                cat_pred = cat_model.predict([cleaned])[0]
                cat_conf = 1.0

            urg_pred, urg_conf, urg_probs_dict = None, 0.0, {}
            if hasattr(urg_model, 'predict_proba'):
                u_probs = urg_model.predict_proba([cleaned])[0]
                u_classes = urg_model.classes_
                urg_pred = u_classes[np.argmax(u_probs)]
                urg_conf = float(np.max(u_probs))
                urg_probs_dict = {c: round(float(p), 4) for c, p in zip(u_classes, u_probs)}
            else:
                urg_pred = urg_model.predict([cleaned])[0]
                urg_conf = 1.0

            results[name] = {
                "label": model_info['label'],
                "category": cat_pred,
                "category_confidence": round(cat_conf, 4),
                "category_probabilities": cat_probs_dict,
                "urgency": urg_pred,
                "urgency_confidence": round(urg_conf, 4),
                "urgency_probabilities": urg_probs_dict,
            }
        except Exception as e:
            results[name] = {"label": model_info.get('label', name), "error": str(e)}

    # Add MuRIL predictions if available
    if muril_trainer_instance:
        try:
            cat_result = muril_trainer_instance['cat'].predict([req.text], "category")[0]
            urg_result = muril_trainer_instance['urg'].predict([req.text], "urgency")[0]
            results['muril'] = {
                "label": "MuRIL (GPU)",
                "category": cat_result['label'],
                "category_confidence": cat_result['confidence'],
                "category_probabilities": cat_result['probabilities'],
                "urgency": urg_result['label'],
                "urgency_confidence": urg_result['confidence'],
                "urgency_probabilities": urg_result['probabilities'],
            }
        except Exception as e:
            results['muril'] = {"label": "MuRIL (GPU)", "error": str(e)}

    cat_predictions = [r.get('category') for r in results.values() if r.get('category')]
    urg_predictions = [r.get('urgency') for r in results.values() if r.get('urgency')]

    cat_agreement = len(set(cat_predictions)) == 1 if cat_predictions else False
    urg_agreement = len(set(urg_predictions)) == 1 if urg_predictions else False

    duration = round((time.time() - start) * 1000)

    return {
        "text": req.text,
        "models": results,
        "consensus": {
            "category_agreement": cat_agreement,
            "urgency_agreement": urg_agreement,
            "category": max(set(cat_predictions), key=cat_predictions.count) if cat_predictions else None,
            "urgency": max(set(urg_predictions), key=urg_predictions.count) if urg_predictions else None,
        },
        "inference_time_ms": duration,
    }


# ============================================================================
# HISTORY & SEARCH
# ============================================================================

@app.get("/history")
@limiter.limit("120/minute")
def get_history(
    request: Request,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    category: Optional[str] = None,
    urgency: Optional[str] = None,
    search: Optional[str] = None,
):
    if not db:
        raise HTTPException(status_code=503, detail="Database not initialized")

    import sqlite3
    conn = sqlite3.connect(db.db_path)

    conditions = []
    params = []

    if category:
        conditions.append("predicted_category = ?")
        params.append(category)
    if urgency:
        conditions.append("predicted_urgency = ?")
        params.append(urgency)
    if search:
        conditions.append("text LIKE ?")
        params.append(f"%{search}%")

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    count_query = f"SELECT COUNT(*) FROM predictions {where}"
    cursor = conn.cursor()
    cursor.execute(count_query, params)
    total = cursor.fetchone()[0]

    query = f"""
        SELECT * FROM predictions {where}
        ORDER BY timestamp DESC
        LIMIT ? OFFSET ?
    """
    import pandas as pd
    df = pd.read_sql_query(query, conn, params=params + [limit, offset])
    conn.close()

    return {
        "predictions": [{k: (None if pd.isna(v) else v) for k, v in row.items()} for row in df.to_dict(orient="records")],
        "count": len(df),
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@app.get("/search")
@limiter.limit("120/minute")
def search_predictions(request: Request, q: str = Query(..., min_length=1), limit: int = 50):
    if not db:
        raise HTTPException(status_code=503, detail="Database not initialized")

    import sqlite3
    import pandas as pd
    conn = sqlite3.connect(db.db_path)
    df = pd.read_sql_query("""
        SELECT * FROM predictions
        WHERE text LIKE ?
        ORDER BY timestamp DESC
        LIMIT ?
    """, conn, params=(f"%{q}%", limit))
    conn.close()

    return {"predictions": [{k: (None if pd.isna(v) else v) for k, v in row.items()} for row in df.to_dict(orient="records")], "count": len(df)}


# ============================================================================
# ANALYTICS
# ============================================================================

@app.get("/stats")
@limiter.limit("120/minute")
def get_stats(request: Request):
    if not db:
        raise HTTPException(status_code=503, detail="Database not initialized")
    return db.get_stats()


@app.get("/analytics/timeline")
@limiter.limit("120/minute")
def get_timeline(request: Request, hours: int = Query(24, ge=1, le=168)):
    if not db:
        raise HTTPException(status_code=503, detail="Database not initialized")

    import sqlite3
    import pandas as pd
    from datetime import timedelta

    conn = sqlite3.connect(db.db_path)
    since = (datetime.now() - timedelta(hours=hours)).isoformat()

    df = pd.read_sql_query("""
        SELECT timestamp, predicted_category, predicted_urgency, confidence_category
        FROM predictions
        WHERE timestamp >= ?
        ORDER BY timestamp ASC
    """, conn, params=(since,))
    conn.close()

    if len(df) == 0:
        return {"timeline": [], "hourly_stats": []}

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:00")

    hourly = df.groupby("hour").agg(
        count=("timestamp", "count"),
        avg_confidence=("confidence_category", "mean"),
    ).reset_index()

    category_hourly = df.groupby(["hour", "predicted_category"]).size().reset_index(name="count")

    return {
        "timeline": hourly.to_dict(orient="records"),
        "category_breakdown": category_hourly.to_dict(orient="records"),
        "total_in_period": len(df),
    }


@app.get("/analytics/word-frequency")
@limiter.limit("120/minute")
def get_word_frequency(request: Request, category: Optional[str] = None, limit: int = 20):
    if not db:
        raise HTTPException(status_code=503, detail="Database not initialized")

    import sqlite3
    conn = sqlite3.connect(db.db_path)

    if category:
        rows = conn.execute(
            "SELECT text FROM predictions WHERE predicted_category = ?",
            (category,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT text FROM predictions").fetchall()
    conn.close()

    if not rows:
        return {"words": []}

    stop_words = {
        "hai", "ka", "ki", "ke", "ko", "me", "se", "ne", "ye", "wo",
        "aur", "ya", "par", "pe", "kya", "kaise", "kab", "kyu",
        "mera", "meri", "mere", "mein", "hum", "main", "nahi", "nahin",
        "the", "is", "are", "was", "were", "a", "an", "the", "and",
        "or", "but", "in", "on", "at", "to", "for", "of", "with",
        "my", "your", "his", "her", "its", "our", "their", "this",
        "that", "it", "be", "have", "has", "had", "do", "does",
    }

    word_counts = Counter()
    for row in rows:
        text = row[0].lower()
        words = text.split()
        for w in words:
            w = w.strip(".,!?;:()[]\"'")
            if len(w) > 2 and w not in stop_words:
                word_counts[w] += 1

    top_words = word_counts.most_common(limit)
    return {"words": [{"word": w, "count": c} for w, c in top_words]}


@app.get("/analytics/confidence")
@limiter.limit("120/minute")
def get_confidence_distribution(request: Request):
    if not db:
        raise HTTPException(status_code=503, detail="Database not initialized")

    import sqlite3
    import pandas as pd

    conn = sqlite3.connect(db.db_path)
    df = pd.read_sql_query("""
        SELECT confidence_category, confidence_urgency, predicted_category
        FROM predictions
    """, conn)
    conn.close()

    if len(df) == 0:
        return {"distribution": [], "bins": []}

    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    df["cat_bin"] = pd.cut(df["confidence_category"], bins=bins, labels=[f"{b:.1f}" for b in bins[:-1]])
    dist = df["cat_bin"].value_counts().sort_index().reset_index()
    dist.columns = ["range", "count"]

    cat_avg = df.groupby("predicted_category")["confidence_category"].mean().reset_index()
    cat_avg.columns = ["category", "avg_confidence"]

    return {
        "distribution": dist.to_dict(orient="records"),
        "category_avg": cat_avg.to_dict(orient="records"),
        "overall_avg": float(df["confidence_category"].mean()) if len(df) > 0 else 0,
    }


@app.get("/analytics/patterns")
@limiter.limit("120/minute")
def get_patterns(request: Request):
    if not db:
        raise HTTPException(status_code=503, detail="Database not initialized")

    import sqlite3
    conn = sqlite3.connect(db.db_path)

    total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    needs_review = conn.execute("SELECT COUNT(*) FROM predictions WHERE confidence_category < 0.5").fetchone()[0]
    corrections = conn.execute("SELECT COUNT(*) FROM predictions WHERE is_correct_category = 0 OR is_correct_urgency = 0").fetchone()[0]

    avg_length = conn.execute("SELECT AVG(LENGTH(text)) FROM predictions").fetchone()[0] or 0

    hour_dist = conn.execute("""
        SELECT strftime('%H', timestamp) as hour, COUNT(*) as cnt
        FROM predictions
        GROUP BY hour
        ORDER BY hour
    """).fetchall()

    conn.close()

    return {
        "total_predictions": total,
        "needs_review_count": needs_review,
        "corrections_count": corrections,
        "review_rate": round(needs_review / max(total, 1), 4),
        "correction_rate": round(corrections / max(total, 1), 4),
        "avg_text_length": round(avg_length, 1),
        "hourly_distribution": [{"hour": int(h), "count": c} for h, c in hour_dist],
    }


@app.get("/analytics/suggestions")
@limiter.exempt
async def get_suggestions():
    """Analyze patterns and return actionable suggestions."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not initialized")

    import sqlite3
    conn = sqlite3.connect(db.db_path)
    suggestions = []

    # 1. Low confidence predictions (potential misclassifications)
    low_conf = conn.execute(
        "SELECT COUNT(*) FROM predictions WHERE confidence_category < 0.5"
    ).fetchone()[0]
    if low_conf > 0:
        suggestions.append({
            "type": "warning",
            "title": "Low Confidence Predictions",
            "description": f"{low_conf} predictions have confidence below 50%. Consider reviewing these in the Review Queue.",
            "action": "/review",
            "action_label": "Review Queue",
            "priority": "high",
        })

    # 2. Correction patterns (model confusion)
    confused = conn.execute("""
        SELECT corrected_category, predicted_category, COUNT(*) as cnt
        FROM predictions
        WHERE is_correct_category = 0 AND corrected_category IS NOT NULL
        GROUP BY corrected_category, predicted_category
        HAVING cnt >= 3
        ORDER BY cnt DESC
        LIMIT 5
    """).fetchall()

    for corrected, predicted, count in confused:
        suggestions.append({
            "type": "insight",
            "title": f"Model Confusion: {predicted} → {corrected}",
            "description": f"{count} complaints predicted as '{predicted}' were corrected to '{corrected}'. The model may struggle distinguishing these categories.",
            "priority": "medium",
        })

    # 3. Category imbalance
    imbalance = conn.execute("""
        SELECT predicted_category, COUNT(*) as cnt
        FROM predictions
        GROUP BY predicted_category
        ORDER BY cnt DESC
    """).fetchall()

    if imbalance:
        total = sum(cnt for _, cnt in imbalance)
        top_cat, top_cnt = imbalance[0]
        if top_cnt / max(total, 1) > 0.5:
            suggestions.append({
                "type": "info",
                "title": f"Category Imbalance: {top_cat}",
                "description": f"{top_cat} makes up {round(top_cnt/total*100)}% of all predictions. Consider if this reflects real distribution or model bias.",
                "priority": "low",
            })

    # 4. Time-based pattern
    hourly = conn.execute("""
        SELECT strftime('%H', timestamp) as hour, COUNT(*) as cnt
        FROM predictions
        GROUP BY hour
        ORDER BY cnt DESC
        LIMIT 1
    """).fetchone()

    if hourly:
        suggestions.append({
            "type": "insight",
            "title": f"Peak Hour: {hourly[0]}:00",
            "description": f"Most complaints arrive around {hourly[0]}:00 with {hourly[1]} predictions. Consider staffing accordingly.",
            "priority": "low",
        })

    # 5. Spam detection (very short texts)
    spam = conn.execute(
        "SELECT COUNT(*) FROM predictions WHERE LENGTH(text) < 10"
    ).fetchone()[0]
    if spam > 0:
        suggestions.append({
            "type": "warning",
            "title": "Possible Spam Detected",
            "description": f"{spam} predictions have very short text (< 10 chars). These may be spam or invalid inputs.",
            "priority": "medium",
        })

    # 6. Retrain readiness
    try:
        retrain_mgr = RetrainManager()
        should_retrain, correction_count = retrain_mgr.should_retrain()
        if should_retrain:
            suggestions.append({
                "type": "action",
                "title": "Model Retrain Ready",
                "description": f"{correction_count} corrections collected (threshold: {retrain_mgr.CORRECTION_THRESHOLD}). Retraining may improve accuracy.",
                "action": "/classify",
                "action_label": "Retrain Now",
                "priority": "high",
            })
    except Exception:
        pass

    conn.close()

    # Sort by priority
    priority_order = {"high": 0, "medium": 1, "low": 2}
    suggestions.sort(key=lambda x: priority_order.get(x.get("priority", "low"), 3))

    return {"suggestions": suggestions, "total": len(suggestions)}


# ============================================================================
# EXPORT
# ============================================================================

@app.get("/export/csv")
@limiter.limit("120/minute")
def export_csv(
    request: Request,
    category: Optional[str] = None,
    urgency: Optional[str] = None,
):
    if not db:
        raise HTTPException(status_code=503, detail="Database not initialized")

    import sqlite3
    conn = sqlite3.connect(db.db_path)

    conditions = []
    params = []
    if category:
        conditions.append("predicted_category = ?")
        params.append(category)
    if urgency:
        conditions.append("predicted_urgency = ?")
        params.append(urgency)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    rows = conn.execute(f"""
        SELECT id, text, predicted_category, confidence_category,
               predicted_urgency, confidence_urgency, timestamp
        FROM predictions {where}
        ORDER BY timestamp DESC
        LIMIT 10000
    """, params).fetchall()
    conn.close()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "text", "category", "category_confidence",
                     "urgency", "urgency_confidence", "timestamp"])
    for row in rows:
        writer.writerow(row)

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"},
    )


@app.get("/export/json")
@limiter.limit("120/minute")
def export_json(
    request: Request,
    category: Optional[str] = None,
    urgency: Optional[str] = None,
):
    if not db:
        raise HTTPException(status_code=503, detail="Database not initialized")

    import sqlite3
    import pandas as pd
    conn = sqlite3.connect(db.db_path)

    conditions = []
    params = []
    if category:
        conditions.append("predicted_category = ?")
        params.append(category)
    if urgency:
        conditions.append("predicted_urgency = ?")
        params.append(urgency)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    df = pd.read_sql_query(f"""
        SELECT id, text, predicted_category, confidence_category,
               predicted_urgency, confidence_urgency, timestamp
        FROM predictions {where}
        ORDER BY timestamp DESC
        LIMIT 10000
    """, conn, params=params)
    conn.close()

    return {
        "predictions": [{k: (None if pd.isna(v) else v) for k, v in row.items()} for row in df.to_dict(orient="records")],
        "count": len(df),
        "exported_at": datetime.now().isoformat(),
    }


@app.get("/export/report")
@limiter.exempt
async def export_report():
    """Generate a comprehensive HTML report for PDF export."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not initialized")

    import sqlite3
    conn = sqlite3.connect(db.db_path)

    # Gather data
    stats = db.get_stats()

    # Category distribution
    cat_dist = conn.execute(
        "SELECT predicted_category, COUNT(*) as cnt FROM predictions GROUP BY predicted_category ORDER BY cnt DESC"
    ).fetchall()

    # Urgency distribution
    urg_dist = conn.execute(
        "SELECT predicted_urgency, COUNT(*) as cnt FROM predictions GROUP BY predicted_urgency ORDER BY cnt DESC"
    ).fetchall()

    # Recent predictions
    recent = conn.execute(
        "SELECT text, predicted_category, predicted_urgency, confidence_category, timestamp FROM predictions ORDER BY timestamp DESC LIMIT 20"
    ).fetchall()

    # Correction rate
    total = stats.get('total_predictions', 0)
    corrections = stats.get('correction_count', 0)

    conn.close()

    # Build HTML (escape all user-sourced values to prevent XSS)
    cat_rows = "".join(f"<tr><td>{html.escape(str(c))}</td><td>{n}</td><td>{round(n/max(total,1)*100, 1)}%</td></tr>" for c, n in cat_dist)
    urg_rows = "".join(f"<tr><td>{html.escape(str(u))}</td><td>{n}</td><td>{round(n/max(total,1)*100, 1)}%</td></tr>" for u, n in urg_dist)
    pred_rows = "".join(f"<tr><td>{html.escape(text[:80])}{'...' if len(text)>80 else ''}</td><td>{html.escape(str(cat))}</td><td>{html.escape(str(urg))}</td><td>{round(conf*100, 1)}%</td><td>{html.escape(str(ts))}</td></tr>" for text, cat, urg, conf, ts in recent)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>HinglishAI - Classification Report</title>
<style>
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; margin: 40px; color: #1a1a2e; background: #fff; }}
  h1 {{ color: #6366f1; border-bottom: 3px solid #6366f1; padding-bottom: 10px; }}
  h2 {{ color: #334155; margin-top: 30px; }}
  .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 20px 0; }}
  .stat {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; text-align: center; }}
  .stat .value {{ font-size: 28px; font-weight: bold; color: #6366f1; }}
  .stat .label {{ font-size: 12px; color: #64748b; text-transform: uppercase; margin-top: 4px; }}
  table {{ width: 100%; border-collapse: collapse; margin: 16px 0; }}
  th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #e2e8f0; font-size: 13px; }}
  th {{ background: #f1f5f9; font-weight: 600; color: #334155; }}
  .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #e2e8f0; color: #94a3b8; font-size: 12px; text-align: center; }}
  @media print {{ body {{ margin: 20px; }} }}
</style>
</head>
<body>
<h1>HinglishAI — Classification Report</h1>
<p style="color: #64748b;">Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>

<div class="stats">
  <div class="stat"><div class="value">{total}</div><div class="label">Total Predictions</div></div>
  <div class="stat"><div class="value">{round(stats.get('category_accuracy', 0)*100, 1)}%</div><div class="label">Category Accuracy</div></div>
  <div class="stat"><div class="value">{round(stats.get('urgency_accuracy', 0)*100, 1)}%</div><div class="label">Urgency Accuracy</div></div>
  <div class="stat"><div class="value">{corrections}</div><div class="label">Corrections</div></div>
</div>

<h2>Category Distribution</h2>
<table><thead><tr><th>Category</th><th>Count</th><th>Percentage</th></tr></thead>
<tbody>{cat_rows}</tbody></table>

<h2>Urgency Distribution</h2>
<table><thead><tr><th>Urgency</th><th>Count</th><th>Percentage</th></tr></thead>
<tbody>{urg_rows}</tbody></table>

<h2>Model Performance</h2>
<table><thead><tr><th>Model</th><th>Metric</th><th>Score</th></tr></thead>
<tbody>
<tr><td>TF-IDF + SVM</td><td>Category F1</td><td>99.69%</td></tr>
<tr><td>Combined Ensemble</td><td>Urgency F1</td><td>99.96%</td></tr>
<tr><td>MuRIL (GPU)</td><td>Category F1</td><td>99.87%</td></tr>
<tr><td>MuRIL (GPU)</td><td>Urgency F1</td><td>100.00%</td></tr>
</tbody></table>

<h2>Recent Predictions (Last 20)</h2>
<table><thead><tr><th>Complaint</th><th>Category</th><th>Urgency</th><th>Confidence</th><th>Time</th></tr></thead>
<tbody>{pred_rows}</tbody></table>

<div class="footer">
  <p>HinglishAI — Hinglish E-Commerce Complaint Classifier</p>
  <p>Powered by MuRIL GPU + scikit-learn TF-IDF + SVM | FastAPI Backend | React Frontend</p>
</div>
</body>
</html>"""

    return HTMLResponse(content=html)


# ============================================================================
# FEEDBACK & RETRAIN
# ============================================================================

@app.post("/feedback")
@limiter.limit("60/minute")
def submit_feedback(body: CorrectionRequest, request: Request):
    if not predictor:
        raise HTTPException(status_code=503, detail="Models not loaded")
    predictor.submit_correction(
        pred_id=body.prediction_id,
        is_correct_cat=body.is_correct_category,
        is_correct_urg=body.is_correct_urgency,
        corrected_cat=body.corrected_category,
        corrected_urg=body.corrected_urgency,
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
@limiter.limit("5/hour")
def trigger_retrain(request: Request, background_tasks: BackgroundTasks):
    retrain_mgr = RetrainManager()
    should_retrain, correction_count = retrain_mgr.should_retrain()
    if not should_retrain:
        return {
            "status": "skipped",
            "message": f"Not enough corrections yet ({correction_count}/{20}). Need {20 - correction_count} more.",
        }

    def do_retrain():
        try:
            retrain_mgr.retrain_sklearn()
            logger.info("Background retrain completed successfully")
        except Exception as e:
            logger.error(f"Background retrain failed: {e}")

    background_tasks.add_task(do_retrain)
    return {
        "status": "started",
        "message": f"Retraining started in background with {correction_count} corrections.",
    }


@app.get("/low-confidence")
@limiter.limit("120/minute")
def get_low_confidence(request: Request, limit: int = 20):
    if not db:
        raise HTTPException(status_code=503, detail="Database not initialized")
    df = db.get_low_confidence_predictions()
    if len(df) == 0:
        return {"predictions": [], "count": 0}
    records = [{k: (None if pd.isna(v) else v) for k, v in row.items()} for row in df.to_dict(orient="records")]
    return {
        "predictions": records[:limit],
        "count": len(df),
    }


# ============================================================================
# RETRAIN HISTORY
# ============================================================================

@app.get("/retrain/history")
@limiter.limit("120/minute")
def get_retrain_history(request: Request):
    if not db:
        raise HTTPException(status_code=503, detail="Database not initialized")
    import sqlite3
    conn = sqlite3.connect(db.db_path)
    rows = conn.execute("""
        SELECT id, corrections_count, model_version, accuracy_before, accuracy_after, timestamp
        FROM retrain_log ORDER BY timestamp DESC LIMIT 20
    """).fetchall()
    conn.close()
    return {
        "history": [
            {"id": r[0], "corrections_count": r[1], "model_version": r[2],
             "accuracy_before": r[3], "accuracy_after": r[4], "timestamp": r[5]}
            for r in rows
        ]
    }


@app.get("/retrain/status")
@limiter.exempt
def get_retrain_status(request: Request):
    retrain_mgr = RetrainManager()
    should_retrain, correction_count = retrain_mgr.should_retrain()
    return {
        "corrections_total": correction_count,
        "threshold": 20,
        "should_retrain": should_retrain,
        "remaining": max(0, 20 - correction_count),
    }


# ============================================================================
# AI ASSISTANT (Groq)
# ============================================================================

class AIRequest(BaseModel):
    text: str
    category: Optional[str] = None
    urgency: Optional[str] = None


@app.post("/ai/resolve")
@limiter.limit("10/minute")
def ai_resolve(body: AIRequest, request: Request):
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY not set. Set it as environment variable.")

    cat_info = f"Category: {body.category}. " if body.category else ""
    urg_info = f"Urgency: {body.urgency}. " if body.urgency else ""

    safe_text = body.text.replace('"', "'").replace('\n', ' ')[:2000]
    safe_cat = body.category.replace('"', "'") if body.category else ""
    safe_urg = body.urgency.replace('"', "'") if body.urgency else ""

    prompt = f"""You are an e-commerce customer support expert. Analyze this complaint and provide resolution steps.

<user_complaint>
{safe_text}
</user_complaint>

<Category>{safe_cat}</Category>
<Urgency>{safe_urg}</Urgency>

Provide exactly 3 actionable resolution steps numbered 1-2-3. Be specific and practical. Keep each step under 30 words. Write in English."""

    payload = {"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}],
               "temperature": 0.3, "max_tokens": 300}

    try:
        resp = groq_request_with_retry(payload, api_key)
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Groq API error: {resp.text}")
        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()
        return {"suggestions": content, "model": data.get("model", "llama-3.3-70b-versatile")}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI request failed: {str(e)}")


@app.post("/ai/draft-response")
@limiter.limit("10/minute")
def ai_draft_response(body: AIRequest, request: Request):
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY not set. Set it as environment variable.")

    safe_text = body.text.replace('"', "'").replace('\n', ' ')[:2000]
    safe_cat = body.category.replace('"', "'") if body.category else ""
    safe_urg = body.urgency.replace('"', "'") if body.urgency else ""

    prompt = f"""You are a professional customer service representative for an Indian e-commerce company.
Write a polite, empathetic response to this customer complaint. Write in English.
The response should acknowledge the issue, apologize, and explain next steps.

<customer_complaint>
{safe_text}
</customer_complaint>

<Category>{safe_cat}</Category>
<Urgency>{safe_urg}</Urgency>

Write a professional 3-4 sentence response. Be warm but professional."""

    payload = {"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}],
               "temperature": 0.5, "max_tokens": 300}

    try:
        resp = groq_request_with_retry(payload, api_key)
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Groq API error: {resp.text}")
        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()
        return {"draft": content, "model": data.get("model", "llama-3.3-70b-versatile")}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI request failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
