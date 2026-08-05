"""
Active Learning System for Hinglish Complaint Classifier
Stores predictions, tracks corrections, and triggers retraining.
"""
import os
import sys
import json
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = PROJECT_ROOT / "data" / "complaints.db"
ORIGINAL_DATA = PROJECT_ROOT / "data" / "raw" / "hinglish_complaints_30k.csv"

CORRECTION_THRESHOLD = 20
LOW_CONFIDENCE_THRESHOLD = 0.40


# ============================================================================
# DATABASE
# ============================================================================

class ComplaintDB:
    def __init__(self, db_path=None):
        self.db_path = db_path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                predicted_category TEXT,
                predicted_urgency TEXT,
                confidence_category REAL,
                confidence_urgency REAL,
                is_correct_category INTEGER DEFAULT 1,
                is_correct_urgency INTEGER DEFAULT 1,
                corrected_category TEXT,
                corrected_urgency TEXT,
                session_id TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS retrain_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                corrections_count INTEGER,
                model_version TEXT,
                accuracy_before REAL,
                accuracy_after REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_category ON predictions(predicted_category)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_urgency ON predictions(predicted_urgency)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_confidence ON predictions(confidence_category)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_text ON predictions(text)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_correct_cat ON predictions(is_correct_category)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_correct_urg ON predictions(is_correct_urgency)")

        conn.commit()
        conn.close()

    def add_prediction(self, text, pred_cat, pred_urg, conf_cat, conf_urg, session_id=None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions
            (text, predicted_category, predicted_urgency,
             confidence_category, confidence_urgency, session_id)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (text, pred_cat, pred_urg, conf_cat, conf_urg, session_id))
        conn.commit()
        pred_id = cursor.lastrowid
        conn.close()
        return pred_id

    def add_correction(self, pred_id, is_correct_cat, is_correct_urg,
                       corrected_cat=None, corrected_urg=None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE predictions SET
                is_correct_category = ?,
                is_correct_urgency = ?,
                corrected_category = COALESCE(?, corrected_category),
                corrected_urgency = COALESCE(?, corrected_urgency)
            WHERE id = ?
        """, (is_correct_cat, is_correct_urg, corrected_cat, corrected_urg, pred_id))
        conn.commit()
        conn.close()

    def get_pending_corrections(self, limit=100):
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("""
            SELECT * FROM predictions
            WHERE (is_correct_category = 0 OR is_correct_urgency = 0)
            AND corrected_category IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT ?
        """, conn, params=(limit,))
        conn.close()
        return df

    def get_correction_count(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM predictions
            WHERE is_correct_category = 0 OR is_correct_urgency = 0
        """)
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def get_low_confidence_predictions(self, threshold=LOW_CONFIDENCE_THRESHOLD):
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("""
            SELECT * FROM predictions
            WHERE confidence_category < ? OR confidence_urgency < ?
            ORDER BY timestamp DESC
        """, conn, params=(threshold, threshold))
        conn.close()
        return df

    def get_stats(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM predictions")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM predictions WHERE is_correct_category = 0")
        cat_errors = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM predictions WHERE is_correct_urgency = 0")
        urg_errors = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM predictions WHERE confidence_category < 0.5")
        low_conf = cursor.fetchone()[0]

        cursor.execute("""
            SELECT predicted_category, COUNT(*) as cnt
            FROM predictions
            GROUP BY predicted_category
            ORDER BY cnt DESC
        """)
        category_dist = dict(cursor.fetchall())

        cursor.execute("""
            SELECT predicted_urgency, COUNT(*) as cnt
            FROM predictions
            GROUP BY predicted_urgency
            ORDER BY cnt DESC
        """)
        urgency_dist = dict(cursor.fetchall())

        conn.close()

        return {
            "total_predictions": total,
            "category_errors": cat_errors,
            "urgency_errors": urg_errors,
            "category_accuracy": round(1 - (cat_errors / max(total, 1)), 4),
            "urgency_accuracy": round(1 - (urg_errors / max(total, 1)), 4),
            "low_confidence_count": low_conf,
            "correction_count": cat_errors + urg_errors,
            "category_distribution": category_dist,
            "urgency_distribution": urgency_dist,
        }

    def get_recent_predictions(self, limit=20):
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("""
            SELECT * FROM predictions
            ORDER BY timestamp DESC
            LIMIT ?
        """, conn, params=(limit,))
        conn.close()
        return df

    def log_retrain(self, corrections_count, model_version, acc_before, acc_after):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO retrain_log
            (corrections_count, model_version, accuracy_before, accuracy_after)
            VALUES (?, ?, ?, ?)
        """, (corrections_count, model_version, acc_before, acc_after))
        conn.commit()
        conn.close()


# ============================================================================
# RETRAIN MANAGER
# ============================================================================

class RetrainManager:
    def __init__(self):
        self.db = ComplaintDB()

    def should_retrain(self):
        correction_count = self.db.get_correction_count()
        return correction_count >= CORRECTION_THRESHOLD, correction_count

    def get_corrections_as_training_data(self):
        corrections_df = self.db.get_pending_corrections(limit=1000)
        if len(corrections_df) == 0:
            return None

        records = []
        for _, row in corrections_df.iterrows():
            text = row["text"]
            cat = row["corrected_category"] if pd.notna(row["corrected_category"]) else row["predicted_category"]
            urg = row["corrected_urgency"] if pd.notna(row["corrected_urgency"]) else row["predicted_urgency"]
            records.append({"text": text, "category": cat, "urgency": urg})

        return pd.DataFrame(records)

    def prepare_retrain_data(self):
        original_df = pd.read_csv(ORIGINAL_DATA)
        corrections_df = self.get_corrections_as_training_data()

        if corrections_df is not None:
            combined = pd.concat([original_df, corrections_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["text"], keep="last")
        else:
            combined = original_df

        return combined

    def retrain_sklearn(self):
        print("=" * 50)
        print("  Retraining sklearn models with corrections...")
        print("=" * 50)

        from src.preprocessor import HinglishPreprocessor
        from src.models import build_pipelines, EnsembleClassifier, save_model, load_model

        df = self.prepare_retrain_data()
        print(f"  Training data: {len(df)} samples")

        preprocessor = HinglishPreprocessor()
        X = df["text"].apply(preprocessor.preprocess)
        y_cat = df["category"].values
        y_urg = df["urgency"].values

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_cat_train, y_cat_test, y_urg_train, y_urg_test = train_test_split(
            X, y_cat, y_urg, test_size=0.15, random_state=42, stratify=y_cat,
        )

        print("\n  Training category models...")
        cat_pipelines = build_pipelines()
        for name, pipe in cat_pipelines.items():
            pipe.fit(X_train, y_cat_train)
            score = pipe.score(X_test, y_cat_test)
            print(f"    {name}: {score:.4f}")

        best_cat_name = max(cat_pipelines, key=lambda k: cat_pipelines[k].score(X_test, y_cat_test))
        best_cat = cat_pipelines[best_cat_name]
        print(f"  Best category model: {best_cat_name}")

        print("\n  Training urgency models...")
        urg_pipelines = build_pipelines()
        for name, pipe in urg_pipelines.items():
            pipe.fit(X_train, y_urg_train)
            score = pipe.score(X_test, y_urg_test)
            print(f"    {name}: {score:.4f}")

        best_urg_name = max(urg_pipelines, key=lambda k: urg_pipelines[k].score(X_test, y_urg_test))
        best_urg = urg_pipelines[best_urg_name]
        print(f"  Best urgency model: {best_urg_name}")

        preprocessor.save(PROJECT_ROOT / "models" / "preprocessor.pkl")
        save_model(best_cat, PROJECT_ROOT / "models" / "category_ensemble.pkl")
        save_model(best_urg, PROJECT_ROOT / "models" / "urgency_ensemble.pkl")

        self.db.log_retrain(
            corrections_count=len(self.get_corrections_as_training_data() or []),
            model_version=f"sklearn_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            acc_before=0,
            acc_after=best_cat.score(X_test, y_cat_test),
        )

        print("\n  Retraining complete!")
        return best_cat.score(X_test, y_cat_test)

    def retrain_muril(self):
        print("=" * 50)
        print("  Retraining MuRIL with corrections...")
        print("=" * 50)

        from src.muril_trainer import MurilTrainer

        df = self.prepare_retrain_data()
        print(f"  Training data: {len(df)} samples")

        trainer = MurilTrainer()

        print("\n  Training category model...")
        X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(df, "category")
        train_ds, val_ds, test_ds = trainer.tokenize_data(X_train, X_val, X_test, y_train, y_val, y_test)
        trainer.build_model()
        trainer.train(train_ds, val_ds, PROJECT_ROOT / "models" / "muril_classifier" / "category", "category")
        trainer.evaluate(test_ds, "category")
        trainer.save(PROJECT_ROOT / "models" / "muril_classifier" / "category")

        print("\n  Training urgency model...")
        X_train_u, X_val_u, X_test_u, y_train_u, y_val_u, y_test_u = trainer.prepare_data(df, "urgency")
        train_ds_u, val_ds_u, test_ds_u = trainer.tokenize_data(X_train_u, X_val_u, X_test_u, y_train_u, y_val_u, y_test_u)
        trainer.build_model()
        trainer.train(train_ds_u, val_ds_u, PROJECT_ROOT / "models" / "muril_classifier" / "urgency", "urgency")
        trainer.evaluate(test_ds_u, "urgency")
        trainer.save(PROJECT_ROOT / "models" / "muril_classifier" / "urgency")

        print("\n  MuRIL retraining complete!")


# ============================================================================
# PREDICTOR (UNIFIED INTERFACE)
# ============================================================================

class ComplaintPredictor:
    def __init__(self, use_muril=True):
        self.use_muril = use_muril
        self.db = ComplaintDB()
        self.sklearn_cat = None
        self.sklearn_urg = None
        self.sklearn_preprocessor = None
        self.muril_cat = None
        self.muril_urg = None

    def load_models(self):
        from src.models import load_model
        from src.preprocessor import HinglishPreprocessor

        print("  Loading sklearn models...")
        self.sklearn_preprocessor = HinglishPreprocessor.load(PROJECT_ROOT / "models" / "preprocessor.pkl")
        self.sklearn_cat = load_model(PROJECT_ROOT / "models" / "category_ensemble.pkl")
        self.sklearn_urg = load_model(PROJECT_ROOT / "models" / "urgency_ensemble.pkl")

        if self.use_muril:
            muril_cat_dir = PROJECT_ROOT / "models" / "muril_classifier" / "category"
            muril_urg_dir = PROJECT_ROOT / "models" / "muril_classifier" / "urgency"
            if muril_cat_dir.exists() and muril_urg_dir.exists():
                print("  Loading MuRIL models...")
                from src.muril_trainer import MurilTrainer
                self.muril_cat = MurilTrainer.load(muril_cat_dir)
                self.muril_urg = MurilTrainer.load(muril_urg_dir)
            else:
                print("  MuRIL models not found, using sklearn only")
                self.use_muril = False

        print("  Models loaded!")

    def predict(self, text, session_id=None):
        cleaned = self.sklearn_preprocessor.preprocess(text)

        if self.use_muril and self.muril_cat:
            cat_result = self.muril_cat.predict([text], "category")[0]
            urg_result = self.muril_urg.predict([text], "urgency")[0]
            source = "muril"
        else:
            cat_proba = self.sklearn_cat.predict_proba([cleaned])[0]
            urg_proba = self.sklearn_urg.predict_proba([cleaned])[0]
            cat_pred = self.sklearn_cat.predict([cleaned])[0]
            urg_pred = self.sklearn_urg.predict([cleaned])[0]
            cat_result = {
                "label": cat_pred,
                "confidence": float(np.max(cat_proba)),
                "probabilities": {
                    self.sklearn_cat.classes_[i]: round(float(cat_proba[i]), 4)
                    for i in range(len(cat_proba))
                },
            }
            urg_result = {
                "label": urg_pred,
                "confidence": float(np.max(urg_proba)),
                "probabilities": {
                    self.sklearn_urg.classes_[i]: round(float(urg_proba[i]), 4)
                    for i in range(len(urg_proba))
                },
            }
            source = "sklearn"

        pred_id = self.db.add_prediction(
            text=text,
            pred_cat=cat_result["label"],
            pred_urg=urg_result["label"],
            conf_cat=cat_result["confidence"],
            conf_urg=urg_result["confidence"],
            session_id=session_id,
        )

        return {
            "id": pred_id,
            "text": text,
            "cleaned_text": cleaned,
            "category": cat_result["label"],
            "category_confidence": cat_result["confidence"],
            "category_probabilities": cat_result["probabilities"],
            "urgency": urg_result["label"],
            "urgency_confidence": urg_result["confidence"],
            "urgency_probabilities": urg_result["probabilities"],
            "source": source,
            "needs_review": (
                cat_result["confidence"] < LOW_CONFIDENCE_THRESHOLD or
                urg_result["confidence"] < LOW_CONFIDENCE_THRESHOLD
            ),
        }

    def submit_correction(self, pred_id, is_correct_cat, is_correct_urg,
                          corrected_cat=None, corrected_urg=None):
        self.db.add_correction(
            pred_id=pred_id,
            is_correct_cat=1 if is_correct_cat else 0,
            is_correct_urg=1 if is_correct_urg else 0,
            corrected_cat=corrected_cat,
            corrected_urg=corrected_urg,
        )

    def get_stats(self):
        return self.db.get_stats()

    def get_recent(self, limit=20):
        return self.db.get_recent_predictions(limit)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    db = ComplaintDB()
    print(f"Database created at: {DB_PATH}")
    print(f"Stats: {db.get_stats()}")
