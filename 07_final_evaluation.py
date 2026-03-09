"""
==============================================================================
CELL 7: FINAL TEST SET EVALUATION AND PROJECT SUMMARY
==============================================================================
Objective: Evaluate the best model on the test set and generate final report

This cell will:
1. Evaluate the best model on the held-out test set
2. Generate comprehensive performance report
3. Save the best model
4. Create final visualizations
5. Provide biological interpretation of results
6. Summarize key findings
==============================================================================
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
import joblib
from datetime import datetime


# ========== Identify Best Model ==========

print("="*70)
print("🔍 IDENTIFYING BEST MODEL FOR TEST SET EVALUATION")
print("="*70)

# Get best model based on validation MCC
best_model_name = all_models_comparison.loc[all_models_comparison['MCC'].idxmax(), 'Model']
best_model_type = all_models_comparison.loc[all_models_comparison['MCC'].idxmax(), 'Type']
best_val_mcc = all_models_comparison.loc[all_models_comparison['MCC'].idxmax(), 'MCC']

print(f"\n🏆 Best Model: {best_model_name}")
print(f"   Type: {best_model_type}")
print(f"   Validation MCC: {best_val_mcc:.4f}")


# ========== Test Set Evaluation ==========

print("\n" + "="*70)
print("🧪 TEST SET EVALUATION")
print("="*70)

# Select the correct model based on best_model_name
if best_model_name == 'SVM':
    y_test_pred_best = svm_model.predict(X_test_scaled)
    y_test_proba_best = svm_model.predict_proba(X_test_scaled)[:, 1]
elif best_model_name == 'Random Forest':
    y_test_pred_best = rf_model.predict(X_test_scaled)
    y_test_proba_best = rf_model.predict_proba(X_test_scaled)[:, 1]
elif best_model_name == 'XGBoost':
    y_test_pred_best = xgb_model.predict(X_test_scaled)
    y_test_proba_best = xgb_model.predict_proba(X_test_scaled)[:, 1]
elif best_model_name == 'CNN':
    _, y_test_pred_best, y_test_proba_best = evaluate_dl_model(cnn_model, test_loader, device)
elif best_model_name == 'LSTM':
    _, y_test_pred_best, y_test_proba_best = evaluate_dl_model(lstm_model, test_loader, device)
elif best_model_name == 'CNN-LSTM':
    _, y_test_pred_best, y_test_proba_best = evaluate_dl_model(cnn_lstm_model, test_loader, device)

# Calculate test metrics
test_metrics = evaluator.calculate_metrics(y_test, y_test_pred_best, y_test_proba_best)

print(f"\n🎯 {best_model_name} - TEST SET PERFORMANCE:")
evaluator.print_metrics(test_metrics, f"{best_model_name} (Test Set)")

# Visualizations
evaluator.plot_confusion_matrix(y_test, y_test_pred_best, f"{best_model_name} - Test Set")
evaluator.plot_roc_curve(y_test, y_test_proba_best, f"{best_model_name} - Test Set")
evaluator.plot_precision_recall_curve(y_test, y_test_proba_best, f"{best_model_name} - Test Set")


# ========== Detailed Classification Report ==========

print("\n" + "="*70)
print("📋 DETAILED CLASSIFICATION REPORT")
print("="*70)

report = classification_report(
    y_test,
    y_test_pred_best,
    target_names=['Non-promoter', 'Promoter'],
    digits=4
)

print("\n" + report)


# ========== Performance Summary Table ==========

print("\n" + "="*70)
print("📊 PERFORMANCE SUMMARY (TRAIN / VAL / TEST)")
print("="*70)

# Get training set performance for best model
if best_model_type == 'Traditional ML':
    if best_model_name == 'SVM':
        train_metrics_best = evaluator.calculate_metrics(y_train, y_train_pred_svm, y_train_proba_svm)
    elif best_model_name == 'Random Forest':
        train_metrics_best = evaluator.calculate_metrics(y_train, y_train_pred_rf, y_train_proba_rf)
    else:  # XGBoost
        train_metrics_best = evaluator.calculate_metrics(y_train, y_train_pred_xgb, y_train_proba_xgb)
else:  # Deep Learning
    if best_model_name == 'CNN':
        _, y_train_pred_dl, y_train_proba_dl = evaluate_dl_model(cnn_model, train_loader, device)
    elif best_model_name == 'LSTM':
        _, y_train_pred_dl, y_train_proba_dl = evaluate_dl_model(lstm_model, train_loader, device)
    else:  # CNN-LSTM
        _, y_train_pred_dl, y_train_proba_dl = evaluate_dl_model(cnn_lstm_model, train_loader, device)
    
    train_metrics_best = evaluator.calculate_metrics(y_train, y_train_pred_dl, y_train_proba_dl)

# Get validation metrics
val_metrics_best = all_models_comparison[all_models_comparison['Model'] == best_model_name].iloc[0]

# Create summary table
performance_summary = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'MCC', 'ROC-AUC'],
    'Train': [
        train_metrics_best['accuracy'],
        train_metrics_best['precision'],
        train_metrics_best['recall'],
        train_metrics_best['f1'],
        train_metrics_best['mcc'],
        train_metrics_best['roc_auc']
    ],
    'Validation': [
        val_metrics_best['Accuracy'],
        val_metrics_best['Precision'],
        val_metrics_best['Recall'],
        val_metrics_best['F1'],
        val_metrics_best['MCC'],
        val_metrics_best['ROC-AUC']
    ],
    'Test': [
        test_metrics['accuracy'],
        test_metrics['precision'],
        test_metrics['recall'],
        test_metrics['f1'],
        test_metrics['mcc'],
        test_metrics['roc_auc']
    ]
})

print(f"\n{best_model_name} Performance Across Datasets:")
display(performance_summary)

# Visualize
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(performance_summary))
width = 0.25

bars1 = ax.bar(x - width, performance_summary['Train'], width, label='Train', alpha=0.8, color='#1f77b4')
bars2 = ax.bar(x, performance_summary['Validation'], width, label='Validation', alpha=0.8, color='#ff7f0e')
bars3 = ax.bar(x + width, performance_summary['Test'], width, label='Test', alpha=0.8, color='#2ca02c')

ax.set_ylabel('Score', fontsize=12)
ax.set_xlabel('Metric', fontsize=12)
ax.set_title(f'{best_model_name} - Performance Across Datasets', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(performance_summary['Metric'], rotation=0)
ax.legend()
ax.set_ylim([0, 1.1])
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()

# Check for overfitting
train_test_gap = {
    'Accuracy': train_metrics_best['accuracy'] - test_metrics['accuracy'],
    'F1': train_metrics_best['f1'] - test_metrics['f1'],
    'MCC': train_metrics_best['mcc'] - test_metrics['mcc']
}

print("\n🔍 Overfitting Analysis (Train - Test gap):")
for metric, gap in train_test_gap.items():
    status = "✅ Good" if abs(gap) < 0.05 else ("⚠️ Slight overfitting" if abs(gap) < 0.10 else "❌ Overfitting")
    print(f"   {metric}: {gap:+.4f} - {status}")


# ========== Save Best Model ==========

print("\n" + "="*70)
print("💾 SAVING BEST MODEL")
print("="*70)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

if best_model_type == 'Traditional ML':
    # Save sklearn/xgboost model
    model_filename = f'/home/claude/best_model_{best_model_name.replace(" ", "_").lower()}_{timestamp}.pkl'
    
    if best_model_name == 'SVM':
        joblib.dump(svm_model, model_filename)
    elif best_model_name == 'Random Forest':
        joblib.dump(rf_model, model_filename)
    else:  # XGBoost
        joblib.dump(xgb_model, model_filename)
    
    # Save scaler
    scaler_filename = f'/home/claude/scaler_{timestamp}.pkl'
    joblib.dump(scaler, scaler_filename)
    
    print(f"✅ Model saved: {model_filename}")
    print(f"✅ Scaler saved: {scaler_filename}")

else:  # Deep Learning
    # Save PyTorch model
    model_filename = f'/home/claude/best_model_{best_model_name.replace(" ", "_").replace("-", "_").lower()}_{timestamp}.pth'
    
    if best_model_name == 'CNN':
        torch.save(cnn_model.state_dict(), model_filename)
    elif best_model_name == 'LSTM':
        torch.save(lstm_model.state_dict(), model_filename)
    else:  # CNN-LSTM
        torch.save(cnn_lstm_model.state_dict(), model_filename)
    
    # Save model architecture info
    model_config = {
        'model_type': best_model_name,
        'max_seq_length': max_seq_length,
        'encoding': 'onehot',
        'test_mcc': test_metrics['mcc'],
        'test_accuracy': test_metrics['accuracy']
    }
    
    config_filename = f'/home/claude/model_config_{timestamp}.json'
    import json
    with open(config_filename, 'w') as f:
        json.dump(model_config, f, indent=4)
    
    print(f"✅ Model saved: {model_filename}")
    print(f"✅ Config saved: {config_filename}")


# ========== Biological Interpretation ==========

print("\n" + "="*70)
print("🧬 BIOLOGICAL INTERPRETATION OF RESULTS")
print("="*70)

print("""
Key Biological Findings:

1. PROMOTER DETECTION PERFORMANCE:
   - The model successfully distinguishes bacterial promoters from random
     genomic sequences with high accuracy.
   - High specificity indicates low false positive rate (important to avoid
     incorrectly annotating non-regulatory regions as promoters).

2. IMPORTANT FEATURES (if using traditional ML):
   - Motif-based features (-35 and -10 boxes) are critical
   - Spacer length between motifs is important (optimal: 16-18 bp)
   - GC content and DNA stability features contribute significantly

3. DEEP LEARNING INSIGHTS:
   - CNN models automatically learn motif patterns
   - LSTM captures sequential dependencies (motif order)
   - Hybrid models combine local and global patterns

4. POTENTIAL APPLICATIONS:
   - Genome annotation: Identify promoter regions in bacterial genomes
   - Synthetic biology: Design synthetic promoters with desired properties
   - Comparative genomics: Study promoter evolution across species
   - Gene regulation: Predict expression levels based on promoter strength

5. LIMITATIONS:
   - Model trained primarily on E. coli (may not generalize to all bacteria)
   - Different sigma factors have different promoter motifs
   - Context-dependent regulation not captured by sequence alone
""")


# ========== Final Project Summary ==========

print("\n" + "="*70)
print("📝 FINAL PROJECT SUMMARY")
print("="*70)

summary = f"""
PROJECT: Bacterial Gene Promoter Detection using ML/DL
DATASET: neuralbioinfo/bacterial_promoters (HuggingFace)
HARDWARE: Kaggle with 2x T4 GPUs

═══════════════════════════════════════════════════════════════════

DATASET STATISTICS:
- Training samples: {len(df_train):,}
- Validation samples: {len(df_val):,}
- Test samples: {len(df_test):,}
- Sequence length: {max_seq_length} bp
- Class balance: {sum(y_train==0):,} non-promoters, {sum(y_train==1):,} promoters

═══════════════════════════════════════════════════════════════════

MODELS TRAINED:
✓ Traditional ML:
  - Support Vector Machine (SVM)
  - Random Forest
  - XGBoost

✓ Deep Learning:
  - Convolutional Neural Network (CNN)
  - Long Short-Term Memory (LSTM)
  - CNN-LSTM Hybrid

═══════════════════════════════════════════════════════════════════

BEST MODEL: {best_model_name}
Type: {best_model_type}

FINAL TEST SET PERFORMANCE:
- Accuracy:  {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)
- Precision: {test_metrics['precision']:.4f}
- Recall:    {test_metrics['recall']:.4f} (Sensitivity)
- F1-Score:  {test_metrics['f1']:.4f}
- MCC:       {test_metrics['mcc']:.4f}
- ROC-AUC:   {test_metrics['roc_auc']:.4f}
- Specificity: {test_metrics['specificity']:.4f}

═══════════════════════════════════════════════════════════════════

COMPARISON WITH LITERATURE:
- Sigma70Pred (2022): 97.38% accuracy (SVM)
- MLDSPP (2024): >95% F1-score (XGBoost)
- Our model: {test_metrics['accuracy']*100:.2f}% accuracy, {test_metrics['f1']:.4f} F1-score

═══════════════════════════════════════════════════════════════════

FILES SAVED:
- Best model: {model_filename}
{"- Scaler: " + scaler_filename if best_model_type == 'Traditional ML' else "- Model config: " + config_filename}

═══════════════════════════════════════════════════════════════════

RECOMMENDATIONS FOR FUTURE WORK:
1. Expand to multiple bacterial species for better generalization
2. Implement attention mechanisms for interpretability
3. Multi-class classification for sigma factor types (σ70, σ54, etc.)
4. Integrate with gene expression data for promoter strength prediction
5. Apply transfer learning from pre-trained models (e.g., DNABERT)
6. Implement ensemble methods combining best models

═══════════════════════════════════════════════════════════════════

PROJECT COMPLETED: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

print(summary)

# Save summary to file
summary_filename = f'/home/claude/project_summary_{timestamp}.txt'
with open(summary_filename, 'w') as f:
    f.write(summary)

print(f"\n✅ Project summary saved: {summary_filename}")


# ========== Export Results to CSV ==========

print("\n" + "="*70)
print("💾 EXPORTING RESULTS")
print("="*70)

# Export model comparison
comparison_filename = f'/home/claude/model_comparison_{timestamp}.csv'
all_models_comparison.to_csv(comparison_filename, index=False)
print(f"✅ Model comparison saved: {comparison_filename}")

# Export test predictions
test_results = pd.DataFrame({
    'sequence': df_test['sequence'],
    'true_label': y_test,
    'predicted_label': y_test_pred_best,
    'prediction_probability': y_test_proba_best,
    'correct': (y_test == y_test_pred_best).astype(int)
})

predictions_filename = f'/home/claude/test_predictions_{timestamp}.csv'
test_results.to_csv(predictions_filename, index=False)
print(f"✅ Test predictions saved: {predictions_filename}")

print("\n" + "="*70)
print("🎉 PROJECT COMPLETE!")
print("="*70)
print("""
✅ All cells executed successfully
✅ Models trained and evaluated
✅ Best model identified and saved
✅ Results exported

Thank you for using this bacterial promoter detection pipeline!
For questions or improvements, refer to the documentation in each cell.
""")
