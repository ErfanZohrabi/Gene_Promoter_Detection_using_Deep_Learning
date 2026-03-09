"""
==============================================================================
CELL 1: SETUP AND DATA LOADING
==============================================================================
Project: Bacterial Gene Promoter Detection using ML/DL
Dataset: neuralbioinfo/bacterial_promoters (HuggingFace)
Objective: Classify DNA sequences as promoters vs non-promoters
Hardware: Kaggle with 2x T4 GPUs

Scientific Background:
- Promoters are DNA regions where RNA polymerase binds to initiate transcription
- Bacterial promoters have conserved motifs: -35 box (TTGACA) and -10 box (TATAAT)
- Distance between these boxes is typically 16-18 bp
- Problem: Distinguish true promoters from random genomic sequences
==============================================================================
"""

# ========== Install Required Packages ==========
import sys
import subprocess

def install_packages():
    """
    Install all required packages for the project.
    
    Packages:
    - datasets: HuggingFace datasets library
    - transformers: Pre-trained models (DNABERT)
    - biopython: Biological sequence manipulation
    - shap: Model interpretability
    - logomaker: Sequence logo visualization
    """
    packages = [
        'datasets',
        'transformers',
        'biopython',
        'shap',
        'logomaker',
        'scikit-learn',
        'imbalanced-learn',
        'xgboost',
        'lightgbm'
    ]
    
    print("📦 Installing required packages...")
    for package in packages:
        print(f"  Installing {package}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])
    print("✅ All packages installed successfully!\n")

# Uncomment to install (run once)
# install_packages()


# ========== Import Libraries ==========
import os
import warnings
warnings.filterwarnings('ignore')

# Core Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json

# Machine Learning
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, matthews_corrcoef,
    confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)

# Deep Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Visualization
from IPython.display import display, HTML

# Check GPU availability
print("="*70)
print("🖥️  HARDWARE CONFIGURATION")
print("="*70)
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
print("="*70)
print()


# ========== Load Dataset from HuggingFace ==========
from datasets import load_dataset

def load_bacterial_promoter_data():
    """
    Load the bacterial promoters dataset from HuggingFace.
    
    Dataset Structure (based on neuralbioinfo/bacterial_promoters):
    - sequence: DNA sequence (ATGC characters)
    - label: Binary label (1 = promoter, 0 = non-promoter)
    - organism: Bacterial species (e.g., E. coli, B. subtilis)
    - position: Genomic position information
    - sigma_factor: Sigma factor type (σ70, σ54, etc.)
    
    Returns:
        dict: Dictionary containing train, validation, and test datasets
    """
    print("="*70)
    print("📊 LOADING DATASET FROM HUGGINGFACE")
    print("="*70)
    print("Dataset: neuralbioinfo/bacterial_promoters")
    print("Loading...")
    
    try:
        # Load dataset
        dataset = load_dataset("neuralbioinfo/bacterial_promoters")
        
        print("\n✅ Dataset loaded successfully!")
        print(f"\nDataset splits: {list(dataset.keys())}")
        
        # Display dataset information
        for split_name, split_data in dataset.items():
            print(f"\n{split_name.upper()} Split:")
            print(f"  Total samples: {len(split_data)}")
            print(f"  Features: {split_data.features}")
            
            # Check class distribution
            if 'label' in split_data.features:
                labels = split_data['label']
                label_counts = Counter(labels)
                print(f"  Class distribution:")
                print(f"    Non-promoters (0): {label_counts.get(0, 0):,} ({label_counts.get(0, 0)/len(labels)*100:.2f}%)")
                print(f"    Promoters (1): {label_counts.get(1, 0):,} ({label_counts.get(1, 0)/len(labels)*100:.2f}%)")
        
        print("="*70)
        return dataset
        
    except Exception as e:
        print(f"❌ Error loading dataset: {str(e)}")
        print("\n⚠️  Attempting to load with authentication...")
        print("If this fails, you may need to:")
        print("1. Login to HuggingFace: huggingface-cli login")
        print("2. Accept the dataset terms on HuggingFace")
        raise

# Load the dataset
dataset = load_bacterial_promoter_data()


# ========== Initial Data Exploration ==========
def explore_dataset(dataset, sample_size=5):
    """
    Perform initial exploration of the dataset.
    
    Args:
        dataset: HuggingFace dataset object
        sample_size: Number of samples to display
    """
    print("\n" + "="*70)
    print("🔍 DATASET EXPLORATION")
    print("="*70)
    
    # Get train split for exploration
    train_data = dataset['train']
    
    # Display sample sequences
    print(f"\n📝 Sample Sequences (first {sample_size}):")
    print("-"*70)
    
    for i in range(min(sample_size, len(train_data))):
        sample = train_data[i]
        sequence = sample['sequence']
        label = sample['label']
        
        print(f"\nSample {i+1}:")
        print(f"  Label: {label} ({'Promoter' if label == 1 else 'Non-promoter'})")
        print(f"  Length: {len(sequence)} bp")
        print(f"  Sequence: {sequence[:80]}...")
        
        # Calculate GC content
        gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence) * 100
        print(f"  GC Content: {gc_content:.2f}%")
    
    print("-"*70)
    
    # Sequence length statistics
    seq_lengths = [len(sample['sequence']) for sample in train_data]
    
    print("\n📏 Sequence Length Statistics:")
    print(f"  Min: {min(seq_lengths)} bp")
    print(f"  Max: {max(seq_lengths)} bp")
    print(f"  Mean: {np.mean(seq_lengths):.2f} bp")
    print(f"  Median: {np.median(seq_lengths):.2f} bp")
    print(f"  Std: {np.std(seq_lengths):.2f} bp")
    
    # Nucleotide composition
    all_sequences = ''.join([sample['sequence'] for sample in train_data])
    total_bases = len(all_sequences)
    
    print("\n🧬 Nucleotide Composition (Overall):")
    for base in ['A', 'T', 'G', 'C']:
        count = all_sequences.count(base)
        percentage = count / total_bases * 100
        print(f"  {base}: {count:,} ({percentage:.2f}%)")
    
    print("="*70)

# Explore the dataset
explore_dataset(dataset)


# ========== Convert to Pandas DataFrame ==========
def dataset_to_dataframe(dataset_split):
    """
    Convert HuggingFace dataset to Pandas DataFrame for easier manipulation.
    
    Args:
        dataset_split: Single split from HuggingFace dataset
        
    Returns:
        pd.DataFrame: Pandas DataFrame with all features
    """
    return pd.DataFrame(dataset_split)

# Convert all splits to DataFrames
print("\n" + "="*70)
print("🔄 CONVERTING TO PANDAS DATAFRAMES")
print("="*70)

df_train = dataset_to_dataframe(dataset['train'])
df_val = dataset_to_dataframe(dataset.get('validation', dataset['train']))  # Use train if no validation
df_test = dataset_to_dataframe(dataset.get('test', dataset['train']))  # Use train if no test

print(f"✅ Train DataFrame: {df_train.shape}")
print(f"✅ Validation DataFrame: {df_val.shape}")
print(f"✅ Test DataFrame: {df_test.shape}")
print("="*70)

# Display first few rows
print("\n📋 Train Data Preview:")
display(df_train.head())

print("\n✅ CELL 1 COMPLETE: Dataset loaded and ready for analysis!")
