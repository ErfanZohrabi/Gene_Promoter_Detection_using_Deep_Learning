"""
==============================================================================
CELL 5: DEEP LEARNING MODELS - CNN, LSTM, CNN-LSTM HYBRID
==============================================================================
Objective: Implement and train deep learning models for promoter detection

Models:
1. 1D Convolutional Neural Network (CNN) - Local motif detection
2. LSTM (Long Short-Term Memory) - Sequential dependencies
3. CNN-LSTM Hybrid - Combined local and sequential features

Key Advantages:
- Automatic feature learning (no manual feature engineering)
- Captures spatial patterns (motifs) and temporal dependencies
- State-of-the-art performance on sequence data

Scientific Reference:
- Nucleic Transformer: CNN + self-attention for E. coli promoters
- CNN-LSTM: Sn=0.89, Sp=0.98 for E. coli
==============================================================================
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, confusion_matrix
)
import time
import warnings
warnings.filterwarnings('ignore')


# ========== GPU Configuration ==========

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("="*70)
print("🖥️  DEVICE CONFIGURATION")
print("="*70)
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print("="*70)


# ========== DNA Sequence Encoder ==========

class DNAEncoder:
    """
    Encode DNA sequences for deep learning models.
    
    Methods:
    - One-hot encoding: Each nucleotide as a binary vector
    - K-mer embedding: Convert to k-mer tokens
    """
    
    def __init__(self, encoding_type='onehot'):
        """
        Initialize encoder.
        
        Args:
            encoding_type: 'onehot' or 'kmer'
        """
        self.encoding_type = encoding_type
        
        # Nucleotide to index mapping
        self.nucleotide_to_idx = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4}
        self.idx_to_nucleotide = {v: k for k, v in self.nucleotide_to_idx.items()}
    
    
    def onehot_encode(self, sequence, max_length=None):
        """
        One-hot encode a DNA sequence.
        
        Args:
            sequence: DNA sequence string
            max_length: Maximum sequence length (for padding/truncation)
            
        Returns:
            numpy array: One-hot encoded sequence (4 x length) or (4 x max_length)
        """
        if max_length is not None:
            # Pad or truncate
            if len(sequence) < max_length:
                sequence = sequence + 'N' * (max_length - len(sequence))
            elif len(sequence) > max_length:
                sequence = sequence[:max_length]
        
        # Initialize matrix
        encoding = np.zeros((4, len(sequence)), dtype=np.float32)
        
        # Encode each nucleotide
        for i, nucleotide in enumerate(sequence):
            idx = self.nucleotide_to_idx.get(nucleotide, 4)  # Unknown = N
            if idx < 4:  # Valid nucleotide
                encoding[idx, i] = 1.0
        
        return encoding
    
    
    def batch_encode(self, sequences, max_length=None):
        """
        Encode a batch of sequences.
        
        Args:
            sequences: List of DNA sequences
            max_length: Maximum sequence length
            
        Returns:
            numpy array: Batch of encoded sequences (batch_size, 4, max_length)
        """
        if max_length is None:
            max_length = max(len(seq) for seq in sequences)
        
        batch = np.array([self.onehot_encode(seq, max_length) for seq in sequences])
        return batch


# ========== PyTorch Dataset ==========

class PromoterDataset(Dataset):
    """
    PyTorch Dataset for bacterial promoter sequences.
    """
    
    def __init__(self, sequences, labels, encoder, max_length=None):
        """
        Initialize dataset.
        
        Args:
            sequences: List of DNA sequences
            labels: List of labels (0 or 1)
            encoder: DNAEncoder instance
            max_length: Maximum sequence length
        """
        self.sequences = sequences
        self.labels = labels
        self.encoder = encoder
        
        # Determine max length if not provided
        if max_length is None:
            self.max_length = max(len(seq) for seq in sequences)
        else:
            self.max_length = max_length
        
        print(f"📦 Dataset initialized:")
        print(f"   Samples: {len(self.sequences)}")
        print(f"   Max length: {self.max_length} bp")
    
    
    def __len__(self):
        return len(self.sequences)
    
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # Encode sequence
        encoded = self.encoder.onehot_encode(sequence, self.max_length)
        
        return torch.FloatTensor(encoded), torch.LongTensor([label])[0]


# ========== Create Datasets and DataLoaders ==========

print("\n" + "="*70)
print("🔧 PREPARING DATA FOR DEEP LEARNING")
print("="*70)

# Initialize encoder
encoder = DNAEncoder(encoding_type='onehot')

# Get sequences and labels
train_sequences = df_train['sequence'].tolist()
val_sequences = df_val['sequence'].tolist()
test_sequences = df_test['sequence'].tolist()

# Determine max sequence length
max_seq_length = max(
    max(len(s) for s in train_sequences),
    max(len(s) for s in val_sequences),
    max(len(s) for s in test_sequences)
)
print(f"\n📏 Maximum sequence length: {max_seq_length} bp")

# Create datasets
print("\n📦 Creating PyTorch datasets...")
train_dataset = PromoterDataset(train_sequences, y_train.tolist(), encoder, max_seq_length)
val_dataset = PromoterDataset(val_sequences, y_val.tolist(), encoder, max_seq_length)
test_dataset = PromoterDataset(test_sequences, y_test.tolist(), encoder, max_seq_length)

# Calculate class weights for weighted sampling
class_counts = np.bincount(y_train)
class_weights = 1. / class_counts
sample_weights = class_weights[y_train]

# Create weighted sampler
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

# Create DataLoaders
BATCH_SIZE = 64

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=2
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2
)

print(f"\n✅ DataLoaders created (batch size: {BATCH_SIZE})")
print(f"   Train batches: {len(train_loader)}")
print(f"   Val batches: {len(val_loader)}")
print(f"   Test batches: {len(test_loader)}")
print("="*70)


# ========== Model 1: 1D CNN ==========

class PromoterCNN(nn.Module):
    """
    1D Convolutional Neural Network for promoter detection.
    
    Architecture:
    - Multiple conv layers with different kernel sizes (capture motifs of different lengths)
    - MaxPooling for dimension reduction
    - Dropout for regularization
    - Fully connected layers for classification
    
    Biological Rationale:
    - Conv kernels detect motifs (like -35 and -10 boxes)
    - Multiple kernel sizes capture different motif lengths
    - MaxPooling identifies presence of motif anywhere in sequence
    """
    
    def __init__(self, input_channels=4, num_filters=128, dropout=0.5):
        """
        Initialize CNN.
        
        Args:
            input_channels: Number of input channels (4 for A,T,G,C)
            num_filters: Number of convolutional filters
            dropout: Dropout probability
        """
        super(PromoterCNN, self).__init__()
        
        # Convolutional layers with different kernel sizes
        # to capture motifs of different lengths
        self.conv1 = nn.Conv1d(input_channels, num_filters, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(num_filters, num_filters, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(num_filters, num_filters, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.bn2 = nn.BatchNorm1d(num_filters)
        self.bn3 = nn.BatchNorm1d(num_filters)
        
        # Pooling
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Global pooling
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(num_filters, 64)
        self.fc2 = nn.Linear(64, 2)  # Binary classification
    
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, 4, seq_length)
            
        Returns:
            Output logits (batch_size, 2)
        """
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        
        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.squeeze(-1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


# ========== Model 2: LSTM ==========

class PromoterLSTM(nn.Module):
    """
    LSTM model for promoter detection.
    
    Architecture:
    - Embedding layer (optional)
    - Bidirectional LSTM layers
    - Attention mechanism
    - Classification head
    
    Biological Rationale:
    - Captures sequential dependencies (e.g., -35 before -10)
    - Bidirectional: reads sequence in both directions
    - Attention: focuses on important regions
    """
    
    def __init__(self, input_size=4, hidden_size=128, num_layers=2, dropout=0.5):
        """
        Initialize LSTM.
        
        Args:
            input_size: Input dimension (4 for one-hot)
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super(PromoterLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size * 2, 64)  # *2 for bidirectional
        self.fc2 = nn.Linear(64, 2)
    
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, 4, seq_length)
            
        Returns:
            Output logits (batch_size, 2)
        """
        # Transpose for LSTM: (batch_size, seq_length, input_size)
        x = x.transpose(1, 2)
        
        # LSTM forward
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use final hidden state from both directions
        # hidden: (num_layers * 2, batch_size, hidden_size)
        forward_hidden = hidden[-2, :, :]
        backward_hidden = hidden[-1, :, :]
        hidden_concat = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        # Fully connected layers
        x = self.dropout(hidden_concat)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


# ========== Model 3: CNN-LSTM Hybrid ==========

class PromoterCNNLSTM(nn.Module):
    """
    CNN-LSTM hybrid model for promoter detection.
    
    Architecture:
    - CNN layers: Extract local features (motifs)
    - LSTM layers: Capture sequential dependencies
    - Classification head
    
    Best of Both Worlds:
    - CNN: Detects conserved motifs at different positions
    - LSTM: Understands motif order and spacing
    
    Scientific Basis:
    - Proven effective for E. coli promoters (Sn=0.89, Sp=0.98)
    """
    
    def __init__(self, input_channels=4, num_filters=64, 
                 lstm_hidden=64, dropout=0.5):
        """
        Initialize CNN-LSTM hybrid.
        
        Args:
            input_channels: Number of input channels (4 for A,T,G,C)
            num_filters: Number of CNN filters
            lstm_hidden: LSTM hidden size
            dropout: Dropout probability
        """
        super(PromoterCNNLSTM, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv1d(input_channels, num_filters, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(num_filters, num_filters, kernel_size=5, padding=2)
        
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.bn2 = nn.BatchNorm1d(num_filters)
        
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=num_filters,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layers
        self.fc1 = nn.Linear(lstm_hidden * 2, 64)
        self.fc2 = nn.Linear(64, 2)
    
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, 4, seq_length)
            
        Returns:
            Output logits (batch_size, 2)
        """
        # CNN feature extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        
        # Transpose for LSTM: (batch_size, seq_length, num_filters)
        x = x.transpose(1, 2)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use final hidden state
        forward_hidden = hidden[-2, :, :]
        backward_hidden = hidden[-1, :, :]
        hidden_concat = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        # Classification
        x = self.dropout(hidden_concat)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


# ========== Training Function ==========

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs=20, device='cuda'):
    """
    Train a PyTorch model.
    
    Args:
        model: PyTorch model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of training epochs
        device: Device to train on
        
    Returns:
        dict: Training history
    """
    model = model.to(device)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate metrics
        train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"  Best Val Acc: {best_val_acc:.4f}")
        print("-" * 70)
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return history, model


print("\n✅ CELL 5 COMPLETE: Deep learning models and training functions defined!")
print("\nNext: Model Training and Evaluation (Cell 6)")
