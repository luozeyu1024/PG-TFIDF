#!/usr/bin/env python3
"""
Base model class definitions
Base implementations of DNN, RF, and XGBoost models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef
import joblib
from sklearn.ensemble import GradientBoostingClassifier
XGBOOST_AVAILABLE = False
import os
from typing import Dict, Tuple, List, Optional

class DNNModel(nn.Module):
    """Deep neural network supporting single-label and multilabel classification"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout_rate: float = 0.3, 
                 is_multilabel: bool = False, pos_weight: Optional[torch.Tensor] = None):
        super(DNNModel, self).__init__()
        
        self.is_multilabel = is_multilabel
        
        # Fully connected layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.fc4 = nn.Linear(hidden_dim // 4, hidden_dim // 8)
        self.fc5 = nn.Linear(hidden_dim // 8, num_classes)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 4)
        self.bn4 = nn.BatchNorm1d(hidden_dim // 8)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Loss function
        if is_multilabel:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.leaky_relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = F.leaky_relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)
        x = self.fc5(x)
        
        return x
    
    def compute_loss(self, outputs, targets):
        return self.criterion(outputs, targets)
    
    def predict_proba(self, X, device):
        """Predict probabilities"""
        self.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X = torch.FloatTensor(X)
            X = X.to(device)
            outputs = self(X)
            
            if self.is_multilabel:
                # Multilabel: use sigmoid
                probabilities = torch.sigmoid(outputs)
            else:
                # Single-label: use softmax
                probabilities = F.softmax(outputs, dim=1)
                
            return probabilities.cpu().numpy()
    
    def predict(self, X, device, threshold=0.5):
        """Predict classes"""
        probabilities = self.predict_proba(X, device)
        
        if self.is_multilabel:
            # Multilabel: use a fixed threshold
            return (probabilities >= threshold).astype(int)
        else:
            # Single-label: use argmax
            return np.argmax(probabilities, axis=1)

class RFModel:
    """Random forest supporting single-label and multilabel classification"""
    
    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None, 
                 random_state: int = 42, n_jobs: int = -1, is_multilabel: bool = False):
        self.is_multilabel = is_multilabel
        
        if is_multilabel:
            # Multilabel: wrap RandomForestClassifier in MultiOutputClassifier
            from sklearn.multioutput import MultiOutputClassifier
            base_classifier = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=n_jobs,
                class_weight='balanced'
            )
            self.model = MultiOutputClassifier(base_classifier, n_jobs=n_jobs)
            self.label_encoder = None  # Multilabel data do not require an encoder
        else:
            # Single-label: use RandomForestClassifier directly
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=n_jobs,
                class_weight='balanced'
            )
            self.label_encoder = LabelEncoder()
            
        self.is_fitted = False
        
    def fit(self, X, y):
        """Train the model"""
        if self.is_multilabel:
            # Multilabel: use original labels directly
            self.model.fit(X, y)
        else:
            # Single-label: encode labels
            y_encoded = self.label_encoder.fit_transform(y)
            self.model.fit(X, y_encoded)
        
        self.is_fitted = True
        return self
    
    def predict_proba(self, X):
        """Predict probabilities"""
        if not self.is_fitted:
            raise ValueError("Model is not trained; call fit first")
        
        if self.is_multilabel:
            # Multilabel: probabilities from each output classifier
            probas = []
            for i, estimator in enumerate(self.model.estimators_):
                proba_output = estimator.predict_proba(X)
                if proba_output.shape[1] == 2:
                    # Normal case: two classes [negative probability, positive probability]
                    proba = proba_output[:, 1]  # Take the positive-class probability
                elif proba_output.shape[1] == 1:
                    # Edge case: only one class
                    # Check classes in this classifier's training data
                    if hasattr(estimator, 'classes_'):
                        if len(estimator.classes_) == 1:
                            if estimator.classes_[0] == 1:
                                # All training samples are positive
                                proba = np.ones(len(X))
                            else:
                                # All training samples are negative
                                proba = np.zeros(len(X))
                        else:
                            # Defensive fallback; should not occur in theory
                            proba = proba_output[:, 0]
                    else:
                        # Class cannot be determined; use the default
                        proba = proba_output[:, 0]
                else:
                    # Other unexpected cases: use the first column
                    proba = proba_output[:, 0]
                probas.append(proba)
            return np.column_stack(probas)
        else:
            # Single-label: standard probability prediction
            return self.model.predict_proba(X)
    
    def predict(self, X, threshold=0.5):
        """Predict classes"""
        if not self.is_fitted:
            raise ValueError("Model is not trained; call fit first")
        
        if self.is_multilabel:
            # Multilabel: use a fixed threshold
            probabilities = self.predict_proba(X)
            return (probabilities >= threshold).astype(int)
        else:
            # Single-label: standard prediction
            y_pred_encoded = self.model.predict(X)
            return self.label_encoder.inverse_transform(y_pred_encoded)
    
    def get_feature_importance(self):
        """Get feature importance"""
        if not self.is_fitted:
            raise ValueError("Model is not trained; call fit first")
        return self.model.feature_importances_
    
    def save_model(self, filepath: str):
        """Save the model"""
        if not self.is_fitted:
            raise ValueError("Cannot save an untrained model")
        
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'is_fitted': self.is_fitted
        }
        joblib.dump(model_data, filepath)
    
    @classmethod
    def load_model(cls, filepath: str):
        """Load the model"""
        model_data = joblib.load(filepath)
        # Detect whether this is a multilabel model
        is_multilabel = hasattr(model_data['model'], 'estimators_') and len(model_data['model'].estimators_) > 1
        rf_model = cls(is_multilabel=is_multilabel)
        rf_model.model = model_data['model']
        rf_model.label_encoder = model_data['label_encoder']
        rf_model.is_fitted = model_data['is_fitted']
        return rf_model

# class XGBModel:
#     """XGBoost model (implemented with sklearn.GradientBoostingClassifier)"""
#     
#     def __init__(self, n_estimators: int = 100, max_depth: int = 6, 
#                  learning_rate: float = 0.1, random_state: int = 42, n_jobs: int = -1):
#         self.n_estimators = n_estimators
#         self.max_depth = max_depth
#         self.learning_rate = learning_rate
#         self.random_state = random_state
#         self.n_jobs = n_jobs
#         
#         # Use sklearn GradientBoostingClassifier
#         self.model = GradientBoostingClassifier(
#             n_estimators=min(50, n_estimators),  # Reduce the number of trees for speed
#             max_depth=max_depth,
#             learning_rate=learning_rate,
#             random_state=random_state
#         )
#         
#         self.label_encoder = LabelEncoder()
#         self.is_fitted = False
#         
#         print(f"    Implement XGBoost functionality using sklearn.GradientBoostingClassifier")
#         
#     def fit(self, X, y, eval_set=None, verbose=False):
#         """Train the model"""
#         # Encode labels
#         y_encoded = self.label_encoder.fit_transform(y)
#         
#         # Use sklearn GradientBoostingClassifier (ignore eval_set)
#         self.model.fit(X, y_encoded)
#         
#         self.is_fitted = True
#         return self
#     
#     def predict_proba(self, X):
#         """Predict probabilities"""
#         if not self.is_fitted:
#             raise ValueError("Model is not trained; call fit first")
#         
#         return self.model.predict_proba(X)
#     
#     def predict(self, X):
#         """Predict classes"""
#         if not self.is_fitted:
#             raise ValueError("Model is not trained; call fit first")
#             
#         y_pred_encoded = self.model.predict(X)
#         return self.label_encoder.inverse_transform(y_pred_encoded)
#     
#     def get_feature_importance(self):
#         """Get feature importance"""
#         if not self.is_fitted:
#             raise ValueError("Model is not trained; call fit first")
#         return self.model.feature_importances_
#     
#     def save_model(self, filepath: str):
#         """Save the model"""
#         if not self.is_fitted:
#             raise ValueError("Cannot save an untrained model")
#         
#         model_data = {
#             'model': self.model,
#             'scaler': self.scaler,
#             'label_encoder': self.label_encoder,
#             'is_fitted': self.is_fitted,
#             'use_real_xgb': self.use_real_xgb,
#             'n_estimators': self.n_estimators,
#             'max_depth': self.max_depth,
#             'learning_rate': self.learning_rate,
#             'random_state': self.random_state,
#             'n_jobs': self.n_jobs
#         }
#         joblib.dump(model_data, filepath)
#     
#     @classmethod
#     def load_model(cls, filepath: str):
#         """Load the model"""
#         model_data = joblib.load(filepath)
#         
#         # Restore parameters
#         n_estimators = model_data.get('n_estimators', 100)
#         max_depth = model_data.get('max_depth', 6)
#         learning_rate = model_data.get('learning_rate', 0.1)
#         random_state = model_data.get('random_state', 42)
#         n_jobs = model_data.get('n_jobs', -1)
#         
#         xgb_model = cls(n_estimators, max_depth, learning_rate, random_state, n_jobs)
#         xgb_model.model = model_data['model']
#         xgb_model.label_encoder = model_data['label_encoder']
#         xgb_model.is_fitted = model_data['is_fitted']
#         xgb_model.use_real_xgb = model_data.get('use_real_xgb', XGBOOST_AVAILABLE)
#         return xgb_model

def calculate_metrics(y_true, y_pred, y_proba=None, class_names=None, is_multilabel=False) -> Dict:
    """Compute classification metrics for single-label and multilabel data"""
    metrics = {}
    
    if is_multilabel:
        return calculate_multilabel_metrics(y_true, y_pred, y_proba, class_names)
    else:
        # Single-label metrics (original logic)
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Precision, recall, F1 score
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        # Macro and micro averages
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='micro', zero_division=0
        )
        
        metrics['macro_precision'] = macro_precision
        metrics['macro_recall'] = macro_recall
        metrics['macro_f1'] = macro_f1
        metrics['micro_precision'] = micro_precision
        metrics['micro_recall'] = micro_recall
        metrics['micro_f1'] = micro_f1
        
        # MCC (Matthews Correlation Coefficient)
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
        
        # Detailed metrics per class
        if class_names is not None:
            for i, class_name in enumerate(class_names):
                if i < len(precision):
                    metrics[f'{class_name}_precision'] = precision[i]
                    metrics[f'{class_name}_recall'] = recall[i]
                    metrics[f'{class_name}_f1'] = f1[i]
                    metrics[f'{class_name}_support'] = support[i]
        
        return metrics

def calculate_multilabel_metrics(y_true, y_pred, y_proba=None, class_names=None) -> Dict:
    """Compute multilabel classification metrics"""
    from sklearn.metrics import hamming_loss, jaccard_score, f1_score, precision_score, recall_score
    
    metrics = {}
    
    # 1. Exact match (all labels must be correct)
    metrics['subset_accuracy'] = accuracy_score(y_true, y_pred)
    
    # 2. F1 score
    metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_samples'] = f1_score(y_true, y_pred, average='samples', zero_division=0)
    
    # 3. Precision and recall
    metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    # 4. Hamming loss (label-level error rate)
    metrics['hamming_loss'] = hamming_loss(y_true, y_pred)
    
    # 5. Jaccard similarity
    metrics['jaccard_micro'] = jaccard_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['jaccard_macro'] = jaccard_score(y_true, y_pred, average='macro', zero_division=0)
    
    # 6. Detailed metrics per label
    per_label_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_label_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_label_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    if class_names is not None:
        for i, class_name in enumerate(class_names):
            if i < len(per_label_precision):
                metrics[f'{class_name}_precision'] = per_label_precision[i]
                metrics[f'{class_name}_recall'] = per_label_recall[i]
                metrics[f'{class_name}_f1'] = per_label_f1[i]
                # Compute support per label (positive sample count)
                metrics[f'{class_name}_support'] = int(y_true[:, i].sum())
    
    return metrics

def save_metrics(metrics: Dict, filepath: str):
    """Save metrics to file"""
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(filepath, index=False)
    print(f"Metrics saved to: {filepath}")

def load_metrics(filepath: str) -> Dict:
    """Load metrics from file"""
    metrics_df = pd.read_csv(filepath)
    return metrics_df.iloc[0].to_dict()
