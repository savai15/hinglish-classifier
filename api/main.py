"""
FastAPI Backend for Hinglish Complaint Classifier
REST API for predictions, analytics, batch processing, and feedback.
"""
import os
import sys
import io
import csv
import json
from pathlib import Path
from contextlib import asynccontextmanager
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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
# CORE ENDPOINTS
# ============================================================================

@app.get("/")
def root():
    return {
        "message": "Hinglish Complaint Classifier API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.get("/categories")
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


# ============================================================================
# HISTORY & SEARCH
# ============================================================================

@app.get("/history")
def get_history(
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
        "predictions": df.to_dict(orient="records"),
        "count": len(df),
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@app.get("/search")
def search_predictions(q: str = Query(..., min_length=1), limit: int = 50):
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

    return {"predictions": df.to_dict(orient="records"), "count": len(df)}


# ============================================================================
# ANALYTICS
# ============================================================================

@app.get("/stats")
def get_stats():
    if not db:
        raise HTTPException(status_code=503, detail="Database not initialized")
    return db.get_stats()


@app.get("/analytics/timeline")
def get_timeline(hours: int = Query(24, ge=1, le=168)):
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
def get_word_frequency(category: Optional[str] = None, limit: int = 20):
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
def get_confidence_distribution():
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
def get_patterns():
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


# ============================================================================
# EXPORT
# ============================================================================

@app.get("/export/csv")
def export_csv(
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
def export_json(
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
    """, conn, params=params)
    conn.close()

    return {
        "predictions": df.to_dict(orient="records"),
        "count": len(df),
        "exported_at": datetime.now().isoformat(),
    }


# ============================================================================
# FEEDBACK & RETRAIN
# ============================================================================

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
