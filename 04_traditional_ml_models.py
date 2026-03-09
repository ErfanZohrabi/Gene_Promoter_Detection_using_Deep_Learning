"""
==============================================================================
CELL 4: TRADITIONAL MACHINE LEARNING MODELS
==============================================================================
Objective: Train and evaluate traditional ML models for promoter detection

Models:
1. Support Vector Machine (SVM) - High accuracy with proper features
2. Random Forest - Interpretable, handles non-linear relationships
3. XGBoost - State-of-the-art gradient boosting

Evaluation Metrics (for imbalanced data):
- Accuracy
- Precision, Recall, F1-score
- Matthews Correlation Coefficient (MCC) - Best for imbalanced
- ROC-AUC
- Precision-Recall AUC

Scientific Reference:
- Sigma70Pred (2022): 97.38% accuracy with SVM
- MLDSPP (2024): >95% F1-score with XGBoost
==============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve,
    average_precision_score
)
import xgboost as xgb
import time
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    Comprehensive model evaluation and visualization.
    """
    
    @staticmethod
    def calculate_metrics(y_true, y_pred, y_pred_proba=None):
        """
        Calculate all relevant metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            
        Returns:
            dict: Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'mcc': matthews_corrcoef(y_true, y_pred),
        }
        
        # Calculate specificity manually
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['sensitivity'] = metrics['recall']  # Same as recall
        
        # Add ROC-AUC if probabilities provided
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            metrics['pr_auc'] = average_precision_score(y_true, y_pred_proba)
        
        return metrics
    
    
    @staticmethod
    def print_metrics(metrics, model_name="Model"):
        """
        Print metrics in a formatted table.
        
        Args:
            metrics: Dictionary of metrics
            model_name: Name of the model
        """
        print(f"\n{'='*70}")
        print(f"📊 {model_name} - Performance Metrics")
        print(f"{'='*70}")
        
        # Key metrics
        print(f"{'Metric':<25} {'Value':>10}")
        print(f"{'-'*70}")
        
        metric_order = [
            ('accuracy', 'Accuracy'),
            ('precision', 'Precision'),
            ('recall', 'Recall (Sensitivity)'),
            ('specificity', 'Specificity'),
            ('f1', 'F1-Score'),
            ('mcc', 'MCC'),
            ('roc_auc', 'ROC-AUC'),
            ('pr_auc', 'PR-AUC')
        ]
        
        for key, label in metric_order:
            if key in metrics:
                value = metrics[key]
                print(f"{label:<25} {value:>10.4f}")
        
        print(f"{'='*70}")
    
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, model_name="Model"):
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Non-promoter', 'Promoter'],
                    yticklabels=['Non-promoter', 'Promoter'],
                    cbar_kws={'label': 'Count'})
        
        plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        # Add percentages
        total = cm.sum()
        for i in range(2):
            for j in range(2):
                plt.text(j + 0.5, i + 0.7, f'({cm[i,j]/total*100:.1f}%)',
                        ha='center', va='center', fontsize=10, color='red')
        
        plt.tight_layout()
        plt.show()
    
    
    @staticmethod
    def plot_roc_curve(y_true, y_pred_proba, model_name="Model"):
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'{model_name} (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    
    @staticmethod
    def plot_precision_recall_curve(y_true, y_pred_proba, model_name="Model"):
        """
        Plot Precision-Recall curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = average_precision_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2,
                label=f'{model_name} (AP = {pr_auc:.4f})')
        
        # Baseline (random classifier)
        baseline = sum(y_true) / len(y_true)
        plt.axhline(y=baseline, color='navy', linestyle='--', lw=2, 
                   label=f'Random (AP = {baseline:.4f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curve - {model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc="best")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


# ========== Data Preprocessing ==========

print("="*70)
print("🔄 DATA PREPROCESSING")
print("="*70)

# Scale features (important for SVM)
print("\n📏 Scaling features using StandardScaler...")
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_features)
X_val_scaled = scaler.transform(X_val_features)
X_test_scaled = scaler.transform(X_test_features)

print(f"✅ Features scaled")
print(f"   Mean: {X_train_scaled.mean():.6f}")
print(f"   Std: {X_train_scaled.std():.6f}")

# Handle class imbalance - calculate class weights
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

print(f"\n⚖️  Class weights (for handling imbalance):")
print(f"   Non-promoter (0): {class_weight_dict[0]:.4f}")
print(f"   Promoter (1): {class_weight_dict[1]:.4f}")

print("="*70)


# ========== Model 1: Support Vector Machine (SVM) ==========

print("\n" + "="*70)
print("🤖 MODEL 1: SUPPORT VECTOR MACHINE (SVM)")
print("="*70)
print("\nConfiguration:")
print("  Kernel: RBF (Radial Basis Function)")
print("  C: 10.0 (regularization parameter)")
print("  Gamma: 'scale' (automatically determined)")
print("  Class weight: Balanced (handles imbalance)")
print("\n🔄 Training SVM...")

start_time = time.time()

# Train SVM with RBF kernel
svm_model = SVC(
    kernel='rbf',
    C=10.0,
    gamma='scale',
    class_weight=class_weight_dict,
    probability=True,  # Enable probability estimates
    random_state=42
)

svm_model.fit(X_train_scaled, y_train)

training_time = time.time() - start_time
print(f"✅ SVM trained in {training_time:.2f} seconds")

# Predictions
print("\n🔮 Making predictions...")
y_train_pred_svm = svm_model.predict(X_train_scaled)
y_val_pred_svm = svm_model.predict(X_val_scaled)
y_test_pred_svm = svm_model.predict(X_test_scaled)

y_train_proba_svm = svm_model.predict_proba(X_train_scaled)[:, 1]
y_val_proba_svm = svm_model.predict_proba(X_val_scaled)[:, 1]
y_test_proba_svm = svm_model.predict_proba(X_test_scaled)[:, 1]

# Evaluate
evaluator = ModelEvaluator()

print("\n" + "🎯 VALIDATION SET PERFORMANCE:")
val_metrics_svm = evaluator.calculate_metrics(y_val, y_val_pred_svm, y_val_proba_svm)
evaluator.print_metrics(val_metrics_svm, "SVM")

# Visualizations
evaluator.plot_confusion_matrix(y_val, y_val_pred_svm, "SVM")
evaluator.plot_roc_curve(y_val, y_val_proba_svm, "SVM")
evaluator.plot_precision_recall_curve(y_val, y_val_proba_svm, "SVM")


# ========== Model 2: Random Forest ==========

print("\n" + "="*70)
print("🌲 MODEL 2: RANDOM FOREST")
print("="*70)
print("\nConfiguration:")
print("  Number of trees: 200")
print("  Max depth: 20")
print("  Min samples split: 10")
print("  Class weight: Balanced")
print("\n🔄 Training Random Forest...")

start_time = time.time()

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight=class_weight_dict,
    random_state=42,
    n_jobs=-1  # Use all CPU cores
)

rf_model.fit(X_train_scaled, y_train)

training_time = time.time() - start_time
print(f"✅ Random Forest trained in {training_time:.2f} seconds")

# Predictions
print("\n🔮 Making predictions...")
y_train_pred_rf = rf_model.predict(X_train_scaled)
y_val_pred_rf = rf_model.predict(X_val_scaled)
y_test_pred_rf = rf_model.predict(X_test_scaled)

y_train_proba_rf = rf_model.predict_proba(X_train_scaled)[:, 1]
y_val_proba_rf = rf_model.predict_proba(X_val_scaled)[:, 1]
y_test_proba_rf = rf_model.predict_proba(X_test_scaled)[:, 1]

# Evaluate
print("\n" + "🎯 VALIDATION SET PERFORMANCE:")
val_metrics_rf = evaluator.calculate_metrics(y_val, y_val_pred_rf, y_val_proba_rf)
evaluator.print_metrics(val_metrics_rf, "Random Forest")

# Visualizations
evaluator.plot_confusion_matrix(y_val, y_val_pred_rf, "Random Forest")
evaluator.plot_roc_curve(y_val, y_val_proba_rf, "Random Forest")
evaluator.plot_precision_recall_curve(y_val, y_val_proba_rf, "Random Forest")

# Feature importance
print("\n📊 Top 15 Most Important Features:")
feature_importance = pd.DataFrame({
    'feature': X_train_features.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False).head(15)

print(feature_importance.to_string(index=False))

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(range(15), feature_importance['importance'].values[::-1], color='#2ca02c', alpha=0.7)
plt.yticks(range(15), feature_importance['feature'].values[::-1])
plt.xlabel('Importance', fontsize=12)
plt.title('Top 15 Feature Importances (Random Forest)', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()


# ========== Model 3: XGBoost ==========

print("\n" + "="*70)
print("🚀 MODEL 3: XGBOOST")
print("="*70)
print("\nConfiguration:")
print("  Max depth: 10")
print("  Learning rate: 0.1")
print("  Number of estimators: 200")
print("  Scale pos weight: Handles imbalance")
print("\n🔄 Training XGBoost...")

start_time = time.time()

# Calculate scale_pos_weight for imbalance
scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)

xgb_model = xgb.XGBClassifier(
    max_depth=10,
    learning_rate=0.1,
    n_estimators=200,
    objective='binary:logistic',
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)

xgb_model.fit(X_train_scaled, y_train)

training_time = time.time() - start_time
print(f"✅ XGBoost trained in {training_time:.2f} seconds")

# Predictions
print("\n🔮 Making predictions...")
y_train_pred_xgb = xgb_model.predict(X_train_scaled)
y_val_pred_xgb = xgb_model.predict(X_val_scaled)
y_test_pred_xgb = xgb_model.predict(X_test_scaled)

y_train_proba_xgb = xgb_model.predict_proba(X_train_scaled)[:, 1]
y_val_proba_xgb = xgb_model.predict_proba(X_val_scaled)[:, 1]
y_test_proba_xgb = xgb_model.predict_proba(X_test_scaled)[:, 1]

# Evaluate
print("\n" + "🎯 VALIDATION SET PERFORMANCE:")
val_metrics_xgb = evaluator.calculate_metrics(y_val, y_val_pred_xgb, y_val_proba_xgb)
evaluator.print_metrics(val_metrics_xgb, "XGBoost")

# Visualizations
evaluator.plot_confusion_matrix(y_val, y_val_pred_xgb, "XGBoost")
evaluator.plot_roc_curve(y_val, y_val_proba_xgb, "XGBoost")
evaluator.plot_precision_recall_curve(y_val, y_val_proba_xgb, "XGBoost")


# ========== Model Comparison ==========

print("\n" + "="*70)
print("🏆 MODEL COMPARISON - VALIDATION SET")
print("="*70)

comparison_df = pd.DataFrame({
    'Model': ['SVM', 'Random Forest', 'XGBoost'],
    'Accuracy': [val_metrics_svm['accuracy'], val_metrics_rf['accuracy'], val_metrics_xgb['accuracy']],
    'Precision': [val_metrics_svm['precision'], val_metrics_rf['precision'], val_metrics_xgb['precision']],
    'Recall': [val_metrics_svm['recall'], val_metrics_rf['recall'], val_metrics_xgb['recall']],
    'F1': [val_metrics_svm['f1'], val_metrics_rf['f1'], val_metrics_xgb['f1']],
    'MCC': [val_metrics_svm['mcc'], val_metrics_rf['mcc'], val_metrics_xgb['mcc']],
    'ROC-AUC': [val_metrics_svm['roc_auc'], val_metrics_rf['roc_auc'], val_metrics_xgb['roc_auc']]
})

print("\n")
display(comparison_df)

# Visualize comparison
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'MCC', 'ROC-AUC']

for idx, (metric, ax) in enumerate(zip(metrics, axes.flatten())):
    values = comparison_df[metric].values
    bars = ax.bar(comparison_df['Model'], values, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Traditional ML Models - Performance Comparison', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Find best model
best_model_idx = comparison_df['MCC'].argmax()
best_model_name = comparison_df.loc[best_model_idx, 'Model']
best_mcc = comparison_df.loc[best_model_idx, 'MCC']

print(f"\n🏆 BEST MODEL (by MCC): {best_model_name}")
print(f"   MCC: {best_mcc:.4f}")

print("\n✅ CELL 4 COMPLETE: Traditional ML models trained and evaluated!")
print("\nNext: Deep Learning Models (Cell 5)")
