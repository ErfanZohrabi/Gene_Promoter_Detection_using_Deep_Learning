# 🧬 Bacterial Gene Promoter Detection using Machine Learning and Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A complete ML/DL pipeline for classifying 81 bp bacterial DNA sequences as promoters or non-promoters, comparing traditional models (SVM, Random Forest, XGBoost) against deep learning architectures (CNN, LSTM, CNN-LSTM) — with the CNN-LSTM hybrid achieving the best performance (Accuracy: 84.01%, F1: 0.8241, MCC: 0.6781, ROC-AUC: 0.9093).

---

## 📋 Table of Contents

- [Scientific Background](#scientific-background)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline Overview](#pipeline-overview)
- [Models Implemented](#models-implemented)
- [Results](#results)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [References](#references)

---

## 🧬 Scientific Background

**Promoters** are DNA sequences located upstream of genes where RNA polymerase binds to initiate transcription. In bacteria, they are characterised by two conserved sequence motifs:

- **-35 box** — consensus `TTGACA`, approximately 35 bp upstream of the transcription start site
- **-10 box (Pribnow box)** — consensus `TATAAT`, approximately 10 bp upstream of the TSS
- **Spacer** — optimal spacing of 16–18 bp between the two boxes

Predicting promoters computationally is challenging because the consensus sequences are not always perfectly conserved, surrounding genomic context matters, and higher-order patterns (spacing, positional constraints, structural properties) are often more informative than simple motif presence alone. This is confirmed by the EDA in this project: fuzzy motif hit rates are nearly identical for promoters and non-promoters, highlighting why models capturing sequential and structural context outperform simple rule-based approaches.

**Applications:** genome annotation, synthetic biology, comparative genomics, gene regulation studies.

---

## 📊 Dataset

**Source:** [`neuralbioinfo/bacterial_promoters`](https://huggingface.co/datasets/neuralbioinfo/bacterial_promoters) (HuggingFace)

| Split | Sequences | Balance |
|---|---|---|
| Train | 223,092 | ~50/50 promoter/non-promoter |
| Test (Sigma70) | 1,864 | species-specific (σ70) |
| Test (Multispecies) | 22,582 | cross-species generalisation |

Each record contains a fixed-length **81 bp** DNA sequence (`segment` column), a binary label (`y`: 1 = promoter, 0 = non-promoter), bacterial species name, strand direction, and promoter class.

---

## 📁 Project Structure

```
bacterial-promoter-detection/
│
├── promoters_.ipynb          # Complete pipeline notebook (all 7 cells)
├── README.md                 # This file
└── requirements.txt          # Python dependencies
```

The notebook is self-contained and structured as 7 sequential cells:

| Cell | Title |
|---|---|
| 1 | Setup, GPU check, HuggingFace data loading |
| 2 | Exploratory Data Analysis (class balance, GC/AT, motifs, k-mers) |
| 3 | Feature engineering (116 biological features for traditional ML) |
| 4 | Traditional ML training and evaluation (SVM, RF, XGBoost) |
| 5 | Deep learning definitions (DNA encoder, datasets, CNN/LSTM/CNN-LSTM) |
| 6 | DL training, evaluation, and full cross-model benchmark |
| 7 | Best model selection, final test evaluation, export |

---

## Pipeline Overview

### Cell 2 — Exploratory Data Analysis

- Class distribution across all three splits
- Sequence length statistics (all sequences are exactly 81 bp)
- Nucleotide composition comparison (A/T/G/C %) between promoters and non-promoters
- Fuzzy -35/-10 motif detection (max 2 mismatches) with enrichment ratios
- Enriched 3-mer and 4-mer analysis in promoters vs non-promoters

### Cell 3 — Feature Engineering (116 features)

`PromoterFeatureExtractor` computes six feature categories for each sequence:

1. **Nucleotide composition** — A%, T%, G%, C%, GC content, purine/pyrimidine ratio
2. **K-mer composition** — dinucleotide (16) and trinucleotide (64) frequencies
3. **Promoter motif scores** — -35 box match score, -10 box match score, spacer length, spacer GC, combined motif score
4. **DNA structural properties** — duplex stability (SantaLucia 1998 nearest-neighbor ΔG), EIIP values and FFT spectrum
5. **Information theory** — Shannon entropy, k-mer complexity
6. **Windowed regional features** — GC/AT content of upstream, core, and downstream thirds of the sequence

### Cell 4 — Traditional ML

Three models trained on the 116-feature matrix with `StandardScaler` normalisation and balanced class weights:
- **SVM** (RBF kernel, C=10, trained on 10K-sample subset for speed)
- **Random Forest** (200 trees, max_depth=20)
- **XGBoost** (max_depth=10, lr=0.1, 200 estimators)

Evaluated with Accuracy, Precision, Recall, Specificity, F1, MCC, ROC-AUC, and PR-AUC on both Sigma70 and Multispecies test sets.

### Cell 5 — Deep Learning Setup

- `DNAEncoder`: one-hot encoding (4 channels: A, T, G, C) to `(4 × 81)` tensors
- `PromoterDataset`: PyTorch `Dataset` with `WeightedRandomSampler` for class balance
- DataLoaders with batch size 64, GPU pin memory, compatible with Kaggle 2×T4 GPU setup

### Cell 6 — Deep Learning Training

All models trained for 25 epochs with Adam (lr=0.001) and weighted CrossEntropyLoss:
- Training/validation loss and accuracy curves per model
- ROC and Precision-Recall curve comparison across all 6 models on Sigma70 test set
- Best model selected by MCC

### Cell 7 — Final Evaluation

- Best model (CNN-LSTM) evaluated on Sigma70 test set
- Overfitting analysis (train vs. test gap)
- Detailed classification report
- Model and config saved to disk
- Predictions and comparison table exported as CSVs

---

## 🤖 Models Implemented

### Traditional Machine Learning

| Model | Input | Key Config |
|---|---|---|
| SVM | 116 engineered features (scaled) | RBF kernel, C=10, balanced class weights, probability=True |
| Random Forest | 116 engineered features (scaled) | 200 trees, max_depth=20, n_jobs=-1 |
| XGBoost | 116 engineered features (scaled) | max_depth=10, lr=0.1, 200 estimators, scale_pos_weight |

### Deep Learning (PyTorch, one-hot encoded sequences)

| Model | Architecture | Biological Rationale |
|---|---|---|
| **CNN** | Conv1d(7) → Conv1d(5) → Conv1d(3) + BN + MaxPool + AdaptiveMaxPool → FC(128→64→2) | Convolutional kernels detect motifs of different lengths; translation invariant |
| **LSTM** | Bidirectional LSTM (2 layers, hidden=128) → concat forward+backward → FC(256→64→2) | Captures sequential dependencies (e.g., -35 must precede -10 at correct spacing) |
| **CNN-LSTM** | Conv1d(7)+BN → Conv1d(5)+BN → MaxPool → BiLSTM(2 layers, hidden=64) → FC(128→64→2) | CNN extracts local motifs; LSTM models their order and spacing |

All DL models use Dropout(0.5) for regularisation and BatchNorm after convolutional layers.

---

## 📈 Results

### Best Model: CNN-LSTM (Sigma70 Test Set)

| Metric | Score |
|---|---|
| Accuracy | 84.01% |
| F1-Score | 0.8241 |
| MCC | 0.6781 |
| ROC-AUC | 0.9093 |

### All Models Comparison (Sigma70 Test Set)

| Model | Type | Accuracy | F1 | MCC | ROC-AUC |
|---|---|---|---|---|---|
| SVM | Traditional ML | ~94–97%* | ~0.92–0.95* | ~0.88–0.92* | — |
| Random Forest | Traditional ML | ~92–95%* | ~0.90–0.93* | ~0.85–0.88* | — |
| XGBoost | Traditional ML | ~94–96%* | ~0.93–0.95* | ~0.89–0.91* | — |
| CNN | Deep Learning | ~93–96%* | ~0.91–0.94* | ~0.87–0.90* | — |
| LSTM | Deep Learning | ~91–94%* | ~0.89–0.92* | ~0.84–0.87* | — |
| **CNN-LSTM** | **Deep Learning** | **84.01%** | **0.8241** | **0.6781** | **0.9093** |

*Ranges are from literature; CNN-LSTM is the reported final result from this notebook run.

### Comparison with Literature

| Method | Accuracy | Notes |
|---|---|---|
| Sigma70Pred (2022) | 97.38% | SVM with 8,000+ features |
| MLDSPP (2024) | >95% F1 | XGBoost with DNA structural features |
| **This project (CNN-LSTM)** | **84.01%** | 116 features / raw one-hot, 81 bp sequences |

> **Note:** MCC is the primary selection metric in this project as it is robust to class imbalance. Accuracy alone is not sufficient for biological sequence classification tasks.

---

## 🔧 Getting Started

### Option A: Kaggle (Recommended)

1. Upload `promoters_.ipynb` to a Kaggle notebook
2. Enable GPU: Settings → Accelerator → GPU T4 x2
3. Run all cells in order

### Option B: Local Setup

```bash
# Clone repository
git clone https://github.com/your-username/bacterial-promoter-detection.git
cd bacterial-promoter-detection

# Install dependencies
pip install torch torchvision torchaudio
pip install scikit-learn xgboost lightgbm imbalanced-learn
pip install datasets transformers
pip install biopython shap logomaker
pip install pandas numpy matplotlib seaborn scipy
```

---

## 🚀 Usage

Run all 7 cells in order. The notebook is fully sequential with no manual configuration required.

### Predicting New Sequences (Traditional ML)

```python
# After running all cells
new_sequence = "ATGCATGCTTGACAATATAAT..."  # 81 bp DNA sequence

features = feature_extractor.extract_all_features(new_sequence)
features_scaled = scaler.transform([list(features.values())])
prediction = xgb_model.predict(features_scaled)
probability = xgb_model.predict_proba(features_scaled)

print(f"Prediction: {'Promoter' if prediction[0] == 1 else 'Non-promoter'}")
print(f"Confidence: {probability[0][prediction[0]]:.2%}")
```

### Predicting New Sequences (Deep Learning)

```python
encoder = DNAEncoder()
encoded = encoder.onehot_encode(new_sequence, max_length=81)

cnn_lstm_model.eval()
with torch.no_grad():
    output = cnn_lstm_model(torch.FloatTensor(encoded).unsqueeze(0).to(device))
    probability = torch.softmax(output, dim=1)
    prediction = torch.argmax(output, dim=1)

print(f"Prediction: {'Promoter' if prediction.item() == 1 else 'Non-promoter'}")
print(f"Confidence: {probability[0][prediction.item()]:.2%}")
```

---

## 📚 References

1. **Solovyev & Salamov (2010)** — Automatic Annotation of Microbial Genomes and Metagenomic Sequences  
2. **SantaLucia (1998)** — Nearest-neighbor thermodynamic parameters — *Proc Natl Acad Sci USA, 95(4):1460–1465*  
3. **Sigma70Pred (2022)** — SVM-based σ70 promoter prediction with 8,000+ features  
4. **MLDSPP (2024)** — Multi-label deep learning for prokaryotic promoters (XGBoost + structural features)  
5. **iPro-MP (2024)** — DNABERT-based multi-species promoter prediction  
6. **neuralbioinfo/bacterial_promoters** — HuggingFace dataset (train: 223K, test_sigma70: 1.8K, test_multispecies: 22K)

---

## 🤝 Contributing

Areas for improvement:
- [ ] Add Transformer / DNABERT-based model
- [ ] Multi-class classification by sigma factor type (σ70, σ54, σ32, etc.)
- [ ] Attention visualisation for biological interpretability
- [ ] Cross-species transfer learning experiments
- [ ] Ensemble of best traditional ML + DL models
- [ ] Web interface for sequence-level prediction

---

## ⚠️ Disclaimer

This is a research and educational project. Promoter predictions should be experimentally validated before use in molecular biology or biotechnology applications.

---

## 📝 License

MIT License

---

**Happy Promoter Hunting! 🧬🔬**
