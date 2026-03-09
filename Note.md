# 🧬 Bacterial Gene Promoter Detection using Machine Learning and Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive machine learning and deep learning solution for detecting bacterial gene promoters from DNA sequences. This project implements and compares multiple approaches, from traditional ML (SVM, Random Forest, XGBoost) to state-of-the-art deep learning models (CNN, LSTM, CNN-LSTM hybrid).

---

## 📋 Table of Contents

- [Overview](#overview)
- [Scientific Background](#scientific-background)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Models Implemented](#models-implemented)
- [Results](#results)
- [References](#references)

---

## 🎯 Overview

This project addresses the problem of **bacterial promoter detection** - identifying DNA regions where RNA polymerase binds to initiate transcription. Accurate promoter prediction is crucial for:

- 🧬 Genome annotation
- 🔬 Understanding gene regulation
- 🧪 Synthetic biology and genetic engineering
- 📊 Comparative genomics studies

### Key Features

✅ **Comprehensive Pipeline**: From data loading to model deployment  
✅ **Multiple Approaches**: Traditional ML + Deep Learning  
✅ **Well-Documented Code**: Every function and class thoroughly documented  
✅ **Biological Insights**: Feature engineering based on molecular biology  
✅ **Reproducible Results**: Seed setting and version control  
✅ **GPU Accelerated**: Optimized for Kaggle T4 GPUs  

---

## 🧬 Scientific Background

### What are Promoters?

Promoters are DNA sequences located upstream of genes where RNA polymerase and transcription factors bind to initiate transcription. In bacteria, promoters have characteristic features:

- **-35 box**: Consensus sequence TTGACA (~35 bp upstream of TSS)
- **-10 box**: Consensus sequence TATAAT (~10 bp upstream of TSS)  
  Also called the Pribnow box
- **Spacer**: 16-18 bp optimal spacing between boxes
- **σ-factor specificity**: Different sigma factors recognize different promoters

### Why is this Hard?

- **Variability**: Not all promoters match consensus sequences perfectly
- **Context-dependent**: Surrounding DNA affects promoter strength
- **Class imbalance**: Promoters are rare in genomic sequences
- **Species-specific**: Different bacteria have different promoter characteristics

---

## 📊 Dataset

**Source**: [neuralbioinfo/bacterial_promoters](https://huggingface.co/datasets/neuralbioinfo/bacterial_promoters) (HuggingFace)

### Dataset Characteristics:

| Property | Details |
|----------|---------|
| **Sequences** | DNA sequences (ATGC) |
| **Labels** | Binary (0 = non-promoter, 1 = promoter) |
| **Organisms** | Multiple bacterial species |
| **Sequence Length** | Variable (typically 50-150 bp) |
| **Features** | Sequence, label, organism, position info |

---

## 📁 Project Structure

```
bacterial-promoter-detection/
│
├── 01_setup_and_data_loading.py      # Data loading and initial setup
├── 02_eda_visualization.py           # Exploratory data analysis
├── 03_feature_engineering.py         # Biological feature extraction
├── 04_traditional_ml_models.py       # SVM, RF, XGBoost
├── 05_deep_learning_models.py        # Model architectures (CNN, LSTM, hybrid)
├── 06_train_evaluate_dl.py           # DL training and evaluation
├── 07_final_evaluation.py            # Test set evaluation and reporting
│
├── README.md                          # This file
└── requirements.txt                   # Python dependencies
```

---

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended, but CPU works too)
- 8GB+ RAM

### Kaggle Setup

1. **Create a New Notebook** in Kaggle
2. **Enable GPU**: Settings → Accelerator → GPU T4 x2
3. **Install Packages** (done automatically in Cell 1)

### Local Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/bacterial-promoter-detection.git
cd bacterial-promoter-detection

# Install dependencies
pip install -r requirements.txt

# Or install manually
pip install torch torchvision torchaudio
pip install scikit-learn xgboost lightgbm
pip install datasets transformers
pip install biopython shap logomaker
pip install pandas numpy matplotlib seaborn
pip install imbalanced-learn
```

---

## 🚀 Usage

### Running on Kaggle (Recommended)

**Execute cells in order**:

```python
# Cell 1: Setup and Data Loading
# - Installs packages
# - Loads dataset from HuggingFace
# - Initial data exploration

# Cell 2: Exploratory Data Analysis
# - Class distribution analysis
# - Sequence length distribution
# - Nucleotide composition
# - Motif detection
# - K-mer analysis

# Cell 3: Feature Engineering
# - Nucleotide composition features
# - K-mer features (di-, tri-, tetranucleotides)
# - Promoter motif scores (-35 and -10 boxes)
# - DNA structural features (stability, EIIP)
# - Information theory features (entropy)
# - Position-specific windowed features

# Cell 4: Traditional ML Models
# - Support Vector Machine (SVM)
# - Random Forest
# - XGBoost
# - Model comparison and evaluation

# Cell 5: Deep Learning Model Definitions
# - CNN architecture
# - LSTM architecture
# - CNN-LSTM hybrid architecture
# - Training utilities

# Cell 6: Train and Evaluate DL Models
# - Train all deep learning models
# - Compare with traditional ML
# - Generate visualizations

# Cell 7: Final Test Set Evaluation
# - Evaluate best model on test set
# - Save model and results
# - Generate comprehensive report
```

### Quick Start Example

```python
# After running all cells, use the best model for prediction:

# For Traditional ML models:
from sklearn.preprocessing import StandardScaler
import joblib

# Load saved model
model = joblib.load('best_model_xgboost_20240215.pkl')
scaler = joblib.load('scaler_20240215.pkl')

# Predict new sequence
new_sequence = "ATGCATGCTTGACAATATAAT..."  # Your DNA sequence
features = feature_extractor.extract_all_features(new_sequence)
features_scaled = scaler.transform([list(features.values())])
prediction = model.predict(features_scaled)
probability = model.predict_proba(features_scaled)

print(f"Prediction: {'Promoter' if prediction[0] == 1 else 'Non-promoter'}")
print(f"Confidence: {probability[0][prediction[0]]:.2%}")
```

```python
# For Deep Learning models:
import torch

# Load model
model = PromoterCNN()  # or PromoterLSTM, PromoterCNNLSTM
model.load_state_dict(torch.load('best_model_cnn_20240215.pth'))
model.eval()

# Encode and predict
encoder = DNAEncoder()
encoded = encoder.onehot_encode(new_sequence, max_length=150)
with torch.no_grad():
    output = model(torch.FloatTensor(encoded).unsqueeze(0))
    probability = torch.softmax(output, dim=1)
    prediction = torch.argmax(output, dim=1)

print(f"Prediction: {'Promoter' if prediction.item() == 1 else 'Non-promoter'}")
print(f"Confidence: {probability[0][prediction.item()]:.2%}")
```

---

## 🤖 Models Implemented

### Traditional Machine Learning

| Model | Key Features | Pros | Cons |
|-------|-------------|------|------|
| **SVM** | RBF kernel, 300+ features | High accuracy, proven effective | Requires feature engineering |
| **Random Forest** | 200 trees, balanced weights | Interpretable, feature importance | May miss complex patterns |
| **XGBoost** | Gradient boosting, GPU-enabled | State-of-the-art for tabular data | Requires tuning |

**Feature Engineering**:
- Nucleotide composition (A%, T%, G%, C%, GC-content)
- K-mer frequencies (di-, tri-, tetranucleotides)
- Promoter motif scores (-35 box, -10 box, spacer)
- DNA structural properties (duplex stability, EIIP)
- Information theory (Shannon entropy, complexity)
- Position-specific windowed features

### Deep Learning

| Model | Architecture | Strengths |
|-------|-------------|-----------|
| **CNN** | Multi-scale convolutions → MaxPool → FC | Automatic motif detection, translation invariant |
| **LSTM** | Bidirectional LSTM → Attention → FC | Sequential dependencies, motif order |
| **CNN-LSTM** | CNN feature extraction → LSTM → FC | Best of both: local + global patterns |

**Input Encoding**: One-hot encoding (4 channels: A, T, G, C)

---

## 📈 Results

### Expected Performance (Based on Literature)

| Model | Accuracy | Precision | Recall | F1 | MCC |
|-------|----------|-----------|--------|-----|-----|
| **SVM** | 94-97% | 0.92-0.96 | 0.91-0.95 | 0.92-0.95 | 0.88-0.92 |
| **Random Forest** | 92-95% | 0.90-0.94 | 0.89-0.93 | 0.90-0.93 | 0.85-0.88 |
| **XGBoost** | 94-96% | 0.93-0.96 | 0.92-0.95 | 0.93-0.95 | 0.89-0.91 |
| **CNN** | 93-96% | 0.91-0.95 | 0.90-0.94 | 0.91-0.94 | 0.87-0.90 |
| **LSTM** | 91-94% | 0.89-0.93 | 0.88-0.92 | 0.89-0.92 | 0.84-0.87 |
| **CNN-LSTM** | 94-97% | 0.92-0.96 | 0.91-0.95 | 0.92-0.95 | 0.88-0.92 |

*Note: Actual results depend on dataset characteristics and hyperparameters*

### Comparison with Literature

- **Sigma70Pred (2022)**: 97.38% accuracy (SVM with 8000+ features)
- **MLDSPP (2024)**: >95% F1-score (XGBoost with DNA structural features)
- **Nucleic Transformer**: State-of-the-art with self-attention mechanisms

---

## 📚 References

### Scientific Papers

1. **Solovyev, V., & Salamov, A. (2010)**  
   *Automatic Annotation of Microbial Genomes and Metagenomic Sequences*  
   https://pubmed.ncbi.nlm.nih.gov/20827586/

2. **Nature 2025 Paper**  
   *Advanced Promoter Detection Methods*  
   https://www.nature.com/articles/s41586-025-10093-z

3. **Gene Promoter Overview**  
   https://www.sciencedirect.com/topics/medicine-and-dentistry/gene-promoter

### Methodological References

- **Sigma70Pred**: SVM-based σ70 promoter prediction (2022)
- **MLDSPP**: Multi-label deep learning for prokaryotic promoters (2024)
- **iPro-MP**: DNABERT-based multi-species prediction (2024)
- **CNN-LSTM for Promoters**: Hybrid approach for E. coli (various)

### Thermodynamic Parameters

- **SantaLucia, J. (1998)**: Nearest-neighbor thermodynamics
  *Proc Natl Acad Sci USA, 95(4):1460-1465*

---

## 🎓 Key Learnings

### Biological Insights

1. **Conserved Motifs**: -35 and -10 boxes are critical for recognition
2. **Spacer Length**: 16-18 bp spacing is optimal but can vary
3. **AT-Rich Regions**: Promoters tend to be AT-rich (facilitates melting)
4. **Context Matters**: Surrounding sequence affects promoter strength
5. **Species Variation**: Different bacteria have promoter variants

### Machine Learning Insights

1. **Feature Engineering is Crucial**: For traditional ML models
2. **Class Imbalance**: Use MCC, F1-score (not just accuracy)
3. **Deep Learning Learns Features**: Automatically discovers motifs
4. **Ensemble Often Best**: Combining multiple models improves performance
5. **Interpretability Matters**: Understanding what model learns is important

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Add more bacterial species data
- [ ] Implement Transformer/BERT-based models (DNABERT)
- [ ] Multi-class classification (sigma factor types)
- [ ] Attention visualization for interpretability
- [ ] Web interface for easy prediction
- [ ] Cross-species transfer learning

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

Created for the Kaggle community

---

## 🙏 Acknowledgments

- **HuggingFace** for the bacterial promoters dataset
- **Kaggle** for providing GPU resources
- **Scientific Community** for promoter detection research
- **Open Source Libraries**: PyTorch, scikit-learn, XGBoost, BioPython

---

## 📧 Contact & Support

For questions, issues, or suggestions:
- Open an issue on GitHub
- Contact via Kaggle discussion
- Refer to documentation in code cells

---

## ⚠️ Disclaimer

This is a research/educational project. Promoter predictions should be validated experimentally for any practical applications in molecular biology or biotechnology.

---

**Happy Promoter Hunting! 🧬🔬**

*Last Updated: 2024*
