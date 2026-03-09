"""
==============================================================================
CELL 6: TRAIN AND EVALUATE DEEP LEARNING MODELS
==============================================================================
Objective: Train CNN, LSTM, and CNN-LSTM models and compare performance

This cell will:
1. Train each deep learning model
2. Evaluate on validation and test sets
3. Generate visualizations (loss curves, ROC curves, confusion matrices)
4. Compare all models (traditional ML + DL)
5. Select the best model
==============================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, confusion_matrix,
    classification_report, roc_curve
)
import time


# ========== Evaluation Function ==========

def evaluate_dl_model(model, data_loader, device='cuda'):
    """
    Evaluate a deep learning model.
    
    Args:
        model: Trained PyTorch model
        data_loader: DataLoader for evaluation
        device: Device to evaluate on
        
    Returns:
        tuple: (y_true, y_pred, y_proba)
    """
    model.eval()
    y_true_list = []
    y_pred_list = []
    y_proba_list = []
    
    with torch.no_grad():
        for sequences, labels in data_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(sequences)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            # Store results
            y_true_list.extend(labels.cpu().numpy())
            y_pred_list.extend(predicted.cpu().numpy())
            y_proba_list.extend(probabilities[:, 1].cpu().numpy())
    
    return np.array(y_true_list), np.array(y_pred_list), np.array(y_proba_list)


def plot_training_history(history, model_name):
    """
    Plot training and validation loss/accuracy curves.
    
    Args:
        history: Training history dictionary
        model_name: Name of the model
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'{model_name} - Training History (Loss)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title(f'{model_name} - Training History (Accuracy)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# ========== Training Configuration ==========

NUM_EPOCHS = 25
LEARNING_RATE = 0.001

# Loss function with class weights
class_weights_tensor = torch.FloatTensor([class_weight_dict[0], class_weight_dict[1]]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

print("="*70)
print("⚙️  TRAINING CONFIGURATION")
print("="*70)
print(f"Number of epochs: {NUM_EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Loss function: CrossEntropyLoss with class weights")
print(f"Class weights: {class_weights_tensor.cpu().numpy()}")
print(f"Device: {device}")
print("="*70)


# ========== Train Model 1: CNN ==========

print("\n" + "="*70)
print("🧠 TRAINING MODEL 1: CNN")
print("="*70)

# Initialize model
cnn_model = PromoterCNN(
    input_channels=4,
    num_filters=128,
    dropout=0.5
)

print(f"\n📊 Model Architecture:")
print(cnn_model)

# Count parameters
total_params = sum(p.numel() for p in cnn_model.parameters())
trainable_params = sum(p.numel() for p in cnn_model.parameters() if p.requires_grad)
print(f"\n📈 Model Parameters:")
print(f"   Total: {total_params:,}")
print(f"   Trainable: {trainable_params:,}")

# Optimizer
optimizer_cnn = optim.Adam(cnn_model.parameters(), lr=LEARNING_RATE)

# Train
print(f"\n🚀 Starting training...")
start_time = time.time()

history_cnn, cnn_model = train_model(
    model=cnn_model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer_cnn,
    num_epochs=NUM_EPOCHS,
    device=device
)

training_time = time.time() - start_time
print(f"\n✅ CNN trained in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

# Plot history
plot_training_history(history_cnn, "CNN")

# Evaluate
print("\n🎯 Evaluating CNN on validation set...")
y_val_true, y_val_pred_cnn_dl, y_val_proba_cnn_dl = evaluate_dl_model(cnn_model, val_loader, device)

val_metrics_cnn_dl = evaluator.calculate_metrics(y_val_true, y_val_pred_cnn_dl, y_val_proba_cnn_dl)
evaluator.print_metrics(val_metrics_cnn_dl, "CNN (Deep Learning)")

evaluator.plot_confusion_matrix(y_val_true, y_val_pred_cnn_dl, "CNN")
evaluator.plot_roc_curve(y_val_true, y_val_proba_cnn_dl, "CNN")


# ========== Train Model 2: LSTM ==========

print("\n" + "="*70)
print("🧠 TRAINING MODEL 2: LSTM")
print("="*70)

# Initialize model
lstm_model = PromoterLSTM(
    input_size=4,
    hidden_size=128,
    num_layers=2,
    dropout=0.5
)

print(f"\n📊 Model Architecture:")
print(lstm_model)

# Count parameters
total_params = sum(p.numel() for p in lstm_model.parameters())
trainable_params = sum(p.numel() for p in lstm_model.parameters() if p.requires_grad)
print(f"\n📈 Model Parameters:")
print(f"   Total: {total_params:,}")
print(f"   Trainable: {trainable_params:,}")

# Optimizer
optimizer_lstm = optim.Adam(lstm_model.parameters(), lr=LEARNING_RATE)

# Train
print(f"\n🚀 Starting training...")
start_time = time.time()

history_lstm, lstm_model = train_model(
    model=lstm_model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer_lstm,
    num_epochs=NUM_EPOCHS,
    device=device
)

training_time = time.time() - start_time
print(f"\n✅ LSTM trained in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

# Plot history
plot_training_history(history_lstm, "LSTM")

# Evaluate
print("\n🎯 Evaluating LSTM on validation set...")
y_val_true, y_val_pred_lstm, y_val_proba_lstm = evaluate_dl_model(lstm_model, val_loader, device)

val_metrics_lstm = evaluator.calculate_metrics(y_val_true, y_val_pred_lstm, y_val_proba_lstm)
evaluator.print_metrics(val_metrics_lstm, "LSTM")

evaluator.plot_confusion_matrix(y_val_true, y_val_pred_lstm, "LSTM")
evaluator.plot_roc_curve(y_val_true, y_val_proba_lstm, "LSTM")


# ========== Train Model 3: CNN-LSTM Hybrid ==========

print("\n" + "="*70)
print("🧠 TRAINING MODEL 3: CNN-LSTM HYBRID")
print("="*70)

# Initialize model
cnn_lstm_model = PromoterCNNLSTM(
    input_channels=4,
    num_filters=64,
    lstm_hidden=64,
    dropout=0.5
)

print(f"\n📊 Model Architecture:")
print(cnn_lstm_model)

# Count parameters
total_params = sum(p.numel() for p in cnn_lstm_model.parameters())
trainable_params = sum(p.numel() for p in cnn_lstm_model.parameters() if p.requires_grad)
print(f"\n📈 Model Parameters:")
print(f"   Total: {total_params:,}")
print(f"   Trainable: {trainable_params:,}")

# Optimizer
optimizer_cnn_lstm = optim.Adam(cnn_lstm_model.parameters(), lr=LEARNING_RATE)

# Train
print(f"\n🚀 Starting training...")
start_time = time.time()

history_cnn_lstm, cnn_lstm_model = train_model(
    model=cnn_lstm_model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer_cnn_lstm,
    num_epochs=NUM_EPOCHS,
    device=device
)

training_time = time.time() - start_time
print(f"\n✅ CNN-LSTM trained in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

# Plot history
plot_training_history(history_cnn_lstm, "CNN-LSTM")

# Evaluate
print("\n🎯 Evaluating CNN-LSTM on validation set...")
y_val_true, y_val_pred_cnn_lstm, y_val_proba_cnn_lstm = evaluate_dl_model(cnn_lstm_model, val_loader, device)

val_metrics_cnn_lstm = evaluator.calculate_metrics(y_val_true, y_val_pred_cnn_lstm, y_val_proba_cnn_lstm)
evaluator.print_metrics(val_metrics_cnn_lstm, "CNN-LSTM")

evaluator.plot_confusion_matrix(y_val_true, y_val_pred_cnn_lstm, "CNN-LSTM")
evaluator.plot_roc_curve(y_val_true, y_val_proba_cnn_lstm, "CNN-LSTM")


# ========== Complete Model Comparison ==========

print("\n" + "="*70)
print("🏆 COMPLETE MODEL COMPARISON - ALL MODELS")
print("="*70)

# Compile all results
all_models_comparison = pd.DataFrame({
    'Model': [
        'SVM', 'Random Forest', 'XGBoost',
        'CNN', 'LSTM', 'CNN-LSTM'
    ],
    'Type': [
        'Traditional ML', 'Traditional ML', 'Traditional ML',
        'Deep Learning', 'Deep Learning', 'Deep Learning'
    ],
    'Accuracy': [
        val_metrics_svm['accuracy'],
        val_metrics_rf['accuracy'],
        val_metrics_xgb['accuracy'],
        val_metrics_cnn_dl['accuracy'],
        val_metrics_lstm['accuracy'],
        val_metrics_cnn_lstm['accuracy']
    ],
    'Precision': [
        val_metrics_svm['precision'],
        val_metrics_rf['precision'],
        val_metrics_xgb['precision'],
        val_metrics_cnn_dl['precision'],
        val_metrics_lstm['precision'],
        val_metrics_cnn_lstm['precision']
    ],
    'Recall': [
        val_metrics_svm['recall'],
        val_metrics_rf['recall'],
        val_metrics_xgb['recall'],
        val_metrics_cnn_dl['recall'],
        val_metrics_lstm['recall'],
        val_metrics_cnn_lstm['recall']
    ],
    'F1': [
        val_metrics_svm['f1'],
        val_metrics_rf['f1'],
        val_metrics_xgb['f1'],
        val_metrics_cnn_dl['f1'],
        val_metrics_lstm['f1'],
        val_metrics_cnn_lstm['f1']
    ],
    'MCC': [
        val_metrics_svm['mcc'],
        val_metrics_rf['mcc'],
        val_metrics_xgb['mcc'],
        val_metrics_cnn_dl['mcc'],
        val_metrics_lstm['mcc'],
        val_metrics_cnn_lstm['mcc']
    ],
    'ROC-AUC': [
        val_metrics_svm['roc_auc'],
        val_metrics_rf['roc_auc'],
        val_metrics_xgb['roc_auc'],
        val_metrics_cnn_dl['roc_auc'],
        val_metrics_lstm['roc_auc'],
        val_metrics_cnn_lstm['roc_auc']
    ]
})

print("\n📊 Complete Model Comparison Table:")
display(all_models_comparison)

# Visualize comparison
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'MCC', 'ROC-AUC']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

for idx, (metric, ax) in enumerate(zip(metrics, axes.flatten())):
    values = all_models_comparison[metric].values
    models = all_models_comparison['Model'].values
    
    bars = ax.bar(range(len(models)), values, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{value:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f'{metric} Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=values.mean(), color='red', linestyle='--', linewidth=1, alpha=0.5, label='Mean')

plt.suptitle('Complete Model Performance Comparison (Validation Set)', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.show()

# Find best model by each metric
print("\n🏆 Best Models by Metric:")
for metric in ['Accuracy', 'Precision', 'Recall', 'F1', 'MCC', 'ROC-AUC']:
    best_idx = all_models_comparison[metric].idxmax()
    best_model = all_models_comparison.loc[best_idx, 'Model']
    best_value = all_models_comparison.loc[best_idx, metric]
    print(f"   {metric:<12}: {best_model:<15} ({best_value:.4f})")

# Overall best model (by MCC - best for imbalanced data)
best_overall_idx = all_models_comparison['MCC'].idxmax()
best_overall_model = all_models_comparison.loc[best_overall_idx, 'Model']
best_overall_mcc = all_models_comparison.loc[best_overall_idx, 'MCC']

print(f"\n🥇 OVERALL BEST MODEL (by MCC): {best_overall_model}")
print(f"   MCC: {best_overall_mcc:.4f}")

print("\n" + "="*70)


# ========== ROC Curve Comparison ==========

print("\n📈 Generating comprehensive ROC curve comparison...")

plt.figure(figsize=(12, 8))

# Traditional ML models
fpr_svm, tpr_svm, _ = roc_curve(y_val, y_val_proba_svm)
fpr_rf, tpr_rf, _ = roc_curve(y_val, y_val_proba_rf)
fpr_xgb, tpr_xgb, _ = roc_curve(y_val, y_val_proba_xgb)

# Deep Learning models
fpr_cnn, tpr_cnn, _ = roc_curve(y_val_true, y_val_proba_cnn_dl)
fpr_lstm, tpr_lstm, _ = roc_curve(y_val_true, y_val_proba_lstm)
fpr_cnn_lstm, tpr_cnn_lstm, _ = roc_curve(y_val_true, y_val_proba_cnn_lstm)

# Plot all models
models_roc = [
    ('SVM', fpr_svm, tpr_svm, val_metrics_svm['roc_auc'], '#1f77b4'),
    ('Random Forest', fpr_rf, tpr_rf, val_metrics_rf['roc_auc'], '#ff7f0e'),
    ('XGBoost', fpr_xgb, tpr_xgb, val_metrics_xgb['roc_auc'], '#2ca02c'),
    ('CNN', fpr_cnn, tpr_cnn, val_metrics_cnn_dl['roc_auc'], '#d62728'),
    ('LSTM', fpr_lstm, tpr_lstm, val_metrics_lstm['roc_auc'], '#9467bd'),
    ('CNN-LSTM', fpr_cnn_lstm, tpr_cnn_lstm, val_metrics_cnn_lstm['roc_auc'], '#8c564b')
]

for name, fpr, tpr, auc, color in models_roc:
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc:.4f})', color=color)

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('ROC Curve Comparison - All Models', fontsize=16, fontweight='bold')
plt.legend(loc="lower right", fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


print("\n✅ CELL 6 COMPLETE: All models trained and evaluated!")
print("\nKey Findings:")
print(f"  - Best Traditional ML: {comparison_df.loc[comparison_df['MCC'].idxmax(), 'Model']}")
print(f"  - Best Deep Learning: {all_models_comparison[all_models_comparison['Type']=='Deep Learning']['MCC'].idxmax()}")
print(f"  - Overall Best: {best_overall_model}")
print("\nNext: Test Set Evaluation and Final Report (Cell 7)")
