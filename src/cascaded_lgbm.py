"""
Cascaded LightGBM GBDT Architecture for Hinglish Complaint & Urgency Classification

Stage 1: Multi-Class LightGBM GBDT for Category Prediction (10 categories)
Stage 2: Category-Conditioned LightGBM GBDT for Urgency Prediction (High, Medium, Low)
         Uses Stage 1 predicted Category probability distributions as features.
"""

import os
import numpy as np
import scipy.sparse as sp
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, f1_score, classification_report

class CascadedLGBMClassifier(BaseEstimator, ClassifierMixin):
    """
    Cascaded Gradient Boosted Decision Tree (LightGBM) Pipeline
    """
    def __init__(self, n_estimators=100, learning_rate=0.05, max_depth=6, num_leaves=31, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.random_state = random_state
        
        # Feature Extractors
        self.word_vec = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=10000,
            sublinear_tf=True
        )
        self.char_vec = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(2, 5),
            max_features=15000,
            sublinear_tf=True
        )
        
        # Stage 1: Category Model (GBDT)
        self.category_model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1
        )
        
        # Stage 2: Urgency Model (Category-Conditioned GBDT)
        self.urgency_model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1
        )
        
        self.is_fitted = False

    def _extract_features(self, texts, fit=False):
        if fit:
            X_word = self.word_vec.fit_transform(texts)
            X_char = self.char_vec.fit_transform(texts)
        else:
            X_word = self.word_vec.transform(texts)
            X_char = self.char_vec.transform(texts)
            
        return sp.hstack([X_word, X_char], format='csr')

    def fit(self, X_texts, y_category, y_urgency):
        """
        Fits both Stage-1 Category GBDT and Stage-2 Category-Conditioned Urgency GBDT
        """
        print("  [Stage 1] Extracting Word + Subword Char TF-IDF features...")
        X_feat = self._extract_features(X_texts, fit=True)
        
        print("  [Stage 1] Training LightGBM Category Model (10 classes)...")
        self.category_model.fit(X_feat, y_category)
        
        # Predict category probabilities for Stage 2 conditioning
        cat_probs = self.category_model.predict_proba(X_feat)
        
        # Stack text features with Category probability distribution
        X_urgency_feat = sp.hstack([X_feat, cat_probs], format='csr')
        
        print("  [Stage 2] Training LightGBM Category-Conditioned Urgency Model (3 classes)...")
        self.urgency_model.fit(X_urgency_feat, y_urgency)
        
        self.is_fitted = True
        return self

    def predict(self, X_texts):
        """
        Returns predictions for both Category and Urgency tasks
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet!")
            
        X_feat = self._extract_features(X_texts, fit=False)
        
        # Stage 1 Prediction
        cat_preds = self.category_model.predict(X_feat)
        cat_probs = self.category_model.predict_proba(X_feat)
        
        # Stage 2 Prediction (Conditioned on Category probabilities)
        X_urgency_feat = sp.hstack([X_feat, cat_probs], format='csr')
        urg_preds = self.urgency_model.predict(X_urgency_feat)
        
        return cat_preds, urg_preds

    def predict_proba(self, X_texts):
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet!")
            
        X_feat = self._extract_features(X_texts, fit=False)
        cat_probs = self.category_model.predict_proba(X_feat)
        
        X_urgency_feat = sp.hstack([X_feat, cat_probs], format='csr')
        urg_probs = self.urgency_model.predict_proba(X_urgency_feat)
        
        return cat_probs, urg_probs
