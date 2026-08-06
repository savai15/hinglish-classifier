"""
MuRIL Fine-tuning Pipeline for Hinglish Complaint Classification
Fine-tunes google/muril-base-cased on 9-category complaint dataset.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)
from sklearn.model_selection import train_test_split

# ============================================================================
# CONSTANTS
# ============================================================================

MODEL_NAME = "google/muril-base-cased"
MODEL_DIR = PROJECT_ROOT / "models" / "muril_classifier"
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "hinglish_complaints_30k.csv"

CATEGORY_MAP = {
    "Account_Technical": 0,
    "Customer_Service": 1,
    "Delivery_Issue": 2,
    "Order_Status": 3,
    "Payment_Invoice": 4,
    "Pricing_Discount": 5,
    "Product_Quality": 6,
    "Returns_Refunds": 7,
    "Wrong_Damaged_Product": 8,
}
ID_TO_CATEGORY = {v: k for k, v in CATEGORY_MAP.items()}

URGENCY_MAP = {"High": 0, "Medium": 1, "Low": 2}
ID_TO_URGENCY = {v: k for k, v in URGENCY_MAP.items()}

MAX_LENGTH = 128
RANDOM_STATE = 42


# ============================================================================
# DATASET
# ============================================================================

class ComplaintDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    f1_weighted = f1_score(labels, preds, average="weighted")
    precision = precision_score(labels, preds, average="macro")
    recall = recall_score(labels, preds, average="macro")
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision_macro": precision,
        "recall_macro": recall,
    }


# ============================================================================
# TRAINER
# ============================================================================

class MurilTrainer:
    def __init__(self, model_name=MODEL_NAME, num_labels=9):
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = None
        self.model = None
        self.trainer = None

    def load_data(self):
        print(f"Loading dataset from {DATA_PATH}...")
        df = pd.read_csv(DATA_PATH)
        print(f"  Total samples: {len(df)}")
        print(f"  Categories: {df['category'].nunique()}")
        print(f"  Urgency levels: {df['urgency'].nunique()}")
        return df

    def prepare_data(self, df, task="category"):
        if task == "category":
            labels = df["category"].map(CATEGORY_MAP).to_numpy()
        else:
            labels = df["urgency"].map(URGENCY_MAP).to_numpy()

        X_train, X_test, y_train, y_test = train_test_split(
            df["text"].to_numpy(), labels,
            test_size=0.15,
            random_state=RANDOM_STATE,
            stratify=labels,
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=0.1,
            random_state=RANDOM_STATE,
            stratify=y_train,
        )

        print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def tokenize_data(self, X_train, X_val, X_test, y_train, y_val, y_test):
        print(f"  Loading tokenizer: {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        train_dataset = ComplaintDataset(X_train, y_train, self.tokenizer)
        val_dataset = ComplaintDataset(X_val, y_val, self.tokenizer)
        test_dataset = ComplaintDataset(X_test, y_test, self.tokenizer)

        return train_dataset, val_dataset, test_dataset

    def build_model(self):
        print(f"  Loading model: {self.model_name}...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            ignore_mismatched_sizes=True,
        )
        return self.model

    def train(self, train_dataset, val_dataset, output_dir, task="category"):
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            learning_rate=2e-5,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            warmup_steps=500,
            logging_steps=100,
            fp16=True,
            dataloader_num_workers=0,
            report_to="none",
            seed=RANDOM_STATE,
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        print(f"\n  Training {task} model...")
        start_time = datetime.now()
        self.trainer.train()
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"  Training completed in {elapsed:.1f}s")

        return self.trainer

    def evaluate(self, test_dataset, task="category"):
        print(f"\n  Evaluating {task} model...")
        results = self.trainer.evaluate(test_dataset)
        print(f"  Accuracy: {results['eval_accuracy']:.4f}")
        print(f"  F1 (macro): {results['eval_f1_macro']:.4f}")
        print(f"  F1 (weighted): {results['eval_f1_weighted']:.4f}")
        return results

    def predict(self, texts, task="category"):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        if self.model is None:
            self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
            self.model.eval()

        encodings = self.tokenizer(
            texts,
            add_special_tokens=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self.model(**encodings)
            probs = torch.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

        results = []
        for i in range(len(texts)):
            pred_id = preds[i].item()
            confidence = probs[i][pred_id].item()
            if task == "category":
                label = ID_TO_CATEGORY[pred_id]
            else:
                label = ID_TO_URGENCY[pred_id]
            results.append({
                "label": label,
                "confidence": round(confidence, 4),
                "probabilities": {
                    (ID_TO_CATEGORY[j] if task == "category" else ID_TO_URGENCY[j]): round(probs[i][j].item(), 4)
                    for j in range(self.num_labels)
                },
            })
        return results

    def save(self, output_dir=None):
        if output_dir is None:
            output_dir = MODEL_DIR
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  Saving model to {output_dir}...")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        meta = {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "max_length": MAX_LENGTH,
            "category_map": CATEGORY_MAP,
            "urgency_map": URGENCY_MAP,
            "task": "urgency" if self.num_labels == 3 else "category",
            "created_at": datetime.now().isoformat(),
        }
        with open(output_dir / "model_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  Model saved successfully!")

    @classmethod
    def load(cls, model_dir=None):
        if model_dir is None:
            model_dir = MODEL_DIR
        model_dir = Path(model_dir)

        trainer = cls()
        trainer.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        trainer.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        trainer.model.eval()

        with open(model_dir / "model_meta.json") as f:
            meta = json.load(f)
        trainer.num_labels = meta["num_labels"]

        print(f"  MuRIL model loaded from {model_dir}")
        return trainer


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def train_muril():
    print("=" * 70)
    print("  MuRIL Fine-tuning Pipeline")
    print("  google/muril-base-cased for Hinglish Complaint Classification")
    print("=" * 70)

    trainer = MurilTrainer()

    # Load data
    df = trainer.load_data()

    # Train category model
    print("\n" + "=" * 50)
    print("  TASK 1: Category Classification (9 classes)")
    print("=" * 50)
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(df, "category")
    train_ds, val_ds, test_ds = trainer.tokenize_data(X_train, X_val, X_test, y_train, y_val, y_test)
    trainer.build_model()
    cat_dir = MODEL_DIR / "category"
    trainer.train(train_ds, val_ds, cat_dir, "category")
    cat_results = trainer.evaluate(test_ds, "category")

    # Detailed classification report
    preds = trainer.trainer.predict(test_ds)
    pred_labels = [ID_TO_CATEGORY[p] for p in np.argmax(preds.predictions, axis=-1)]
    true_labels = [ID_TO_CATEGORY[l] for l in y_test]
    print("\n  Classification Report:")
    print(classification_report(true_labels, pred_labels, digits=4))

    trainer.save(cat_dir)

    # Train urgency model
    print("\n" + "=" * 50)
    print("  TASK 2: Urgency Classification (3 classes)")
    print("=" * 50)
    X_train_u, X_val_u, X_test_u, y_train_u, y_val_u, y_test_u = trainer.prepare_data(df, "urgency")
    train_ds_u, val_ds_u, test_ds_u = trainer.tokenize_data(X_train_u, X_val_u, X_test_u, y_train_u, y_val_u, y_test_u)
    trainer.build_model()
    urg_dir = MODEL_DIR / "urgency"
    trainer.train(train_ds_u, val_ds_u, urg_dir, "urgency")
    urg_results = trainer.evaluate(test_ds_u, "urgency")

    preds_u = trainer.trainer.predict(test_ds_u)
    pred_labels_u = [ID_TO_URGENCY[p] for p in np.argmax(preds_u.predictions, axis=-1)]
    true_labels_u = [ID_TO_URGENCY[l] for l in y_test_u]
    print("\n  Classification Report:")
    print(classification_report(true_labels_u, pred_labels_u, digits=4))

    trainer.save(urg_dir)

    # Summary
    print("\n" + "=" * 70)
    print("  MuRIL Fine-tuning Results")
    print("=" * 70)
    print(f"  Category Accuracy: {cat_results['eval_accuracy']:.4f}")
    print(f"  Category F1 (macro): {cat_results['eval_f1_macro']:.4f}")
    print(f"  Urgency Accuracy: {urg_results['eval_accuracy']:.4f}")
    print(f"  Urgency F1 (macro): {urg_results['eval_f1_macro']:.4f}")
    print(f"\n  Models saved to: {MODEL_DIR}")
    print("=" * 70)

    return cat_results, urg_results


if __name__ == "__main__":
    train_muril()
