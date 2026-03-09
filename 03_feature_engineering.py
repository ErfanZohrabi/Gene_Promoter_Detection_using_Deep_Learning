"""
==============================================================================
CELL 3: FEATURE ENGINEERING
==============================================================================
Objective: Extract biological and statistical features from DNA sequences
           to use in traditional ML models (SVM, Random Forest, XGBoost)

Feature Categories:
1. Sequence-based: k-mer composition, nucleotide frequencies
2. Motif-based: -35 and -10 box scores, spacer length
3. Structural: DNA duplex stability, GC content profiles
4. Physicochemical: EIIP values, Z-curve parameters
5. Information theory: Shannon entropy, complexity
6. Position-specific: Windowed features

Scientific References:
- Sigma70Pred (2022): 8000+ features with SVM
- MLDSPP (2024): DNA structural properties with XGBoost
==============================================================================
"""

import numpy as np
import pandas as pd
from collections import Counter
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class PromoterFeatureExtractor:
    """
    Comprehensive feature extractor for bacterial promoter sequences.
    
    Extracts multiple feature categories based on biological significance
    and proven effectiveness in promoter detection literature.
    """
    
    def __init__(self):
        """Initialize feature extractor with consensus sequences and parameters."""
        
        # Consensus promoter motifs for E. coli σ70
        self.consensus_35 = "TTGACA"  # -35 box
        self.consensus_10 = "TATAAT"  # -10 box (Pribnow box)
        
        # EIIP (Electron-Ion Interaction Pseudopotential) values
        # Source: Cosic (1994), Nair & Sreenadhan (2006)
        self.eiip = {
            'A': 0.1260,
            'T': 0.1335,
            'G': 0.0806,
            'C': 0.1340
        }
        
        # Dinucleotide thermodynamic parameters (kcal/mol)
        # Source: SantaLucia (1998)
        self.duplex_stability = {
            'AA': -1.00, 'TT': -1.00,
            'AT': -0.88, 'TA': -0.58,
            'CA': -1.45, 'TG': -1.45,
            'GT': -1.44, 'AC': -1.44,
            'CT': -1.28, 'AG': -1.28,
            'GA': -1.30, 'TC': -1.30,
            'CG': -2.17, 'GC': -2.24,
            'GG': -1.84, 'CC': -1.84
        }
        
        print("✅ PromoterFeatureExtractor initialized")
        print(f"   - Consensus -35 box: {self.consensus_35}")
        print(f"   - Consensus -10 box: {self.consensus_10}")
    
    
    # ========== 1. Sequence Composition Features ==========
    
    def extract_nucleotide_composition(self, sequence):
        """
        Extract basic nucleotide composition features.
        
        Features:
        - Mononucleotide frequencies (A%, T%, G%, C%)
        - GC content
        - AT content
        - Purine/Pyrimidine ratio
        
        Returns:
            dict: Nucleotide composition features
        """
        total_len = len(sequence)
        if total_len == 0:
            return {f'{base}_freq': 0 for base in 'ATGC'}
        
        features = {}
        
        # Mononucleotide frequencies
        for base in 'ATGC':
            features[f'{base}_freq'] = sequence.count(base) / total_len
        
        # GC and AT content
        features['GC_content'] = features['G_freq'] + features['C_freq']
        features['AT_content'] = features['A_freq'] + features['T_freq']
        
        # Purine (A, G) vs Pyrimidine (T, C) ratio
        purine = features['A_freq'] + features['G_freq']
        pyrimidine = features['T_freq'] + features['C_freq']
        features['purine_pyrimidine_ratio'] = purine / pyrimidine if pyrimidine > 0 else 0
        
        return features
    
    
    def extract_kmer_composition(self, sequence, k_values=[2, 3, 4]):
        """
        Extract k-mer composition features.
        
        Args:
            sequence: DNA sequence
            k_values: List of k values for k-mers
            
        Features:
        - Dinucleotide (k=2): 16 features
        - Trinucleotide (k=3): 64 features
        - Tetranucleotide (k=4): 256 features
        
        Returns:
            dict: K-mer frequency features
        """
        features = {}
        
        for k in k_values:
            # Generate all possible k-mers
            bases = 'ATGC'
            
            # Count k-mers in sequence
            kmer_counts = Counter()
            for i in range(len(sequence) - k + 1):
                kmer = sequence[i:i+k]
                if all(base in bases for base in kmer):  # Valid k-mer
                    kmer_counts[kmer] += 1
            
            # Calculate frequencies
            total_kmers = sum(kmer_counts.values())
            
            # Generate all possible k-mers for this k
            from itertools import product
            all_kmers = [''.join(p) for p in product(bases, repeat=k)]
            
            for kmer in all_kmers:
                freq = kmer_counts.get(kmer, 0) / total_kmers if total_kmers > 0 else 0
                features[f'kmer_{k}_{kmer}'] = freq
        
        return features
    
    
    # ========== 2. Motif-Based Features ==========
    
    def score_motif_match(self, sequence, motif, max_mismatches=2):
        """
        Score the best match of a motif in the sequence.
        
        Args:
            sequence: DNA sequence
            motif: Consensus motif
            max_mismatches: Maximum allowed mismatches
            
        Returns:
            tuple: (best_score, best_position, mismatches)
                   Score is normalized (1.0 = perfect match)
        """
        motif_len = len(motif)
        best_score = 0
        best_position = -1
        best_mismatches = motif_len
        
        for i in range(len(sequence) - motif_len + 1):
            subseq = sequence[i:i+motif_len]
            mismatches = sum(1 for a, b in zip(subseq, motif) if a != b)
            
            if mismatches <= max_mismatches:
                score = (motif_len - mismatches) / motif_len
                if score > best_score:
                    best_score = score
                    best_position = i
                    best_mismatches = mismatches
        
        return best_score, best_position, best_mismatches
    
    
    def extract_promoter_motif_features(self, sequence):
        """
        Extract features related to promoter motifs (-35 and -10 boxes).
        
        Features:
        - -35 box match score
        - -10 box match score
        - Spacer length (distance between boxes)
        - Spacer GC content
        - Combined motif score
        
        Returns:
            dict: Promoter motif features
        """
        features = {}
        
        # Score -35 box
        score_35, pos_35, mismatch_35 = self.score_motif_match(
            sequence, self.consensus_35, max_mismatches=2
        )
        features['box_35_score'] = score_35
        features['box_35_position'] = pos_35 if pos_35 >= 0 else -1
        features['box_35_mismatches'] = mismatch_35
        
        # Score -10 box
        score_10, pos_10, mismatch_10 = self.score_motif_match(
            sequence, self.consensus_10, max_mismatches=2
        )
        features['box_10_score'] = score_10
        features['box_10_position'] = pos_10 if pos_10 >= 0 else -1
        features['box_10_mismatches'] = mismatch_10
        
        # Spacer analysis (if both boxes found)
        if pos_35 >= 0 and pos_10 >= 0 and pos_10 > pos_35:
            spacer_start = pos_35 + len(self.consensus_35)
            spacer_end = pos_10
            spacer_length = spacer_end - spacer_start
            spacer_seq = sequence[spacer_start:spacer_end]
            
            features['spacer_length'] = spacer_length
            features['spacer_gc_content'] = (spacer_seq.count('G') + spacer_seq.count('C')) / len(spacer_seq) if len(spacer_seq) > 0 else 0
            
            # Optimal spacer is 16-18 bp
            features['spacer_optimal'] = 1 if 16 <= spacer_length <= 18 else 0
        else:
            features['spacer_length'] = 0
            features['spacer_gc_content'] = 0
            features['spacer_optimal'] = 0
        
        # Combined score
        features['combined_motif_score'] = score_35 * score_10
        features['both_motifs_found'] = 1 if (score_35 > 0 and score_10 > 0) else 0
        
        return features
    
    
    # ========== 3. DNA Structural Features ==========
    
    def extract_stability_features(self, sequence):
        """
        Extract DNA duplex stability features based on thermodynamics.
        
        Uses nearest-neighbor parameters (SantaLucia, 1998).
        
        Features:
        - Mean duplex stability (ΔG)
        - Stability variance
        - Min/Max stability
        - Stability profile characteristics
        
        Returns:
            dict: DNA stability features
        """
        features = {}
        
        # Calculate stability for each dinucleotide
        stabilities = []
        for i in range(len(sequence) - 1):
            dinuc = sequence[i:i+2]
            if dinuc in self.duplex_stability:
                stabilities.append(self.duplex_stability[dinuc])
        
        if stabilities:
            features['stability_mean'] = np.mean(stabilities)
            features['stability_std'] = np.std(stabilities)
            features['stability_min'] = np.min(stabilities)
            features['stability_max'] = np.max(stabilities)
            features['stability_range'] = features['stability_max'] - features['stability_min']
        else:
            features['stability_mean'] = 0
            features['stability_std'] = 0
            features['stability_min'] = 0
            features['stability_max'] = 0
            features['stability_range'] = 0
        
        return features
    
    
    def extract_eiip_features(self, sequence):
        """
        Extract EIIP (Electron-Ion Interaction Pseudopotential) features.
        
        EIIP represents the energy of delocalized electrons and differs
        between coding and non-coding regions.
        
        Features:
        - Mean EIIP
        - EIIP variance
        - EIIP spectrum (via FFT)
        
        Returns:
            dict: EIIP features
        """
        features = {}
        
        # Convert sequence to EIIP values
        eiip_values = [self.eiip.get(base, 0) for base in sequence]
        
        if eiip_values:
            features['eiip_mean'] = np.mean(eiip_values)
            features['eiip_std'] = np.std(eiip_values)
            features['eiip_min'] = np.min(eiip_values)
            features['eiip_max'] = np.max(eiip_values)
            
            # FFT spectrum (dominant frequency)
            if len(eiip_values) > 3:
                fft = np.fft.fft(eiip_values)
                power_spectrum = np.abs(fft[:len(fft)//2])**2
                features['eiip_dominant_freq'] = np.argmax(power_spectrum[1:]) + 1  # Skip DC component
            else:
                features['eiip_dominant_freq'] = 0
        else:
            features['eiip_mean'] = 0
            features['eiip_std'] = 0
            features['eiip_min'] = 0
            features['eiip_max'] = 0
            features['eiip_dominant_freq'] = 0
        
        return features
    
    
    # ========== 4. Information Theory Features ==========
    
    def extract_entropy_features(self, sequence):
        """
        Extract information theory features.
        
        Features:
        - Shannon entropy
        - Relative entropy (vs genomic background)
        - Sequence complexity
        
        Returns:
            dict: Entropy and complexity features
        """
        features = {}
        
        # Shannon entropy
        if len(sequence) > 0:
            counts = Counter(sequence)
            total = sum(counts.values())
            probs = [count/total for count in counts.values()]
            entropy = -sum(p * np.log2(p) for p in probs if p > 0)
            features['shannon_entropy'] = entropy
            
            # Sequence complexity (unique k-mers / total k-mers)
            k = 3
            kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
            if kmers:
                features['kmer_complexity'] = len(set(kmers)) / len(kmers)
            else:
                features['kmer_complexity'] = 0
        else:
            features['shannon_entropy'] = 0
            features['kmer_complexity'] = 0
        
        return features
    
    
    # ========== 5. Position-Specific Features ==========
    
    def extract_windowed_features(self, sequence, window_size=10):
        """
        Extract features from sliding windows.
        
        Biological rationale:
        - Different regions of promoters have different characteristics
        - Position-specific patterns are important
        
        Args:
            sequence: DNA sequence
            window_size: Window size in bp
            
        Returns:
            dict: Windowed features
        """
        features = {}
        
        # Define windows (upstream, core promoter, downstream)
        seq_len = len(sequence)
        
        # Split into regions
        if seq_len >= 30:
            upstream = sequence[:seq_len//3]
            core = sequence[seq_len//3:2*seq_len//3]
            downstream = sequence[2*seq_len//3:]
            
            regions = [
                ('upstream', upstream),
                ('core', core),
                ('downstream', downstream)
            ]
            
            for region_name, region_seq in regions:
                if len(region_seq) > 0:
                    # GC content
                    gc = (region_seq.count('G') + region_seq.count('C')) / len(region_seq)
                    features[f'{region_name}_gc'] = gc
                    
                    # AT content
                    features[f'{region_name}_at'] = 1 - gc
        
        return features
    
    
    # ========== Main Feature Extraction ==========
    
    def extract_all_features(self, sequence, include_kmer=True):
        """
        Extract all features for a single sequence.
        
        Args:
            sequence: DNA sequence string
            include_kmer: Whether to include k-mer features (increases dimensionality)
            
        Returns:
            dict: Dictionary of all features
        """
        all_features = {}
        
        # 1. Nucleotide composition
        all_features.update(self.extract_nucleotide_composition(sequence))
        
        # 2. K-mer composition (optional, high-dimensional)
        if include_kmer:
            all_features.update(self.extract_kmer_composition(sequence, k_values=[2, 3]))
        
        # 3. Promoter motifs
        all_features.update(self.extract_promoter_motif_features(sequence))
        
        # 4. DNA structural features
        all_features.update(self.extract_stability_features(sequence))
        all_features.update(self.extract_eiip_features(sequence))
        
        # 5. Information theory
        all_features.update(self.extract_entropy_features(sequence))
        
        # 6. Windowed features
        all_features.update(self.extract_windowed_features(sequence))
        
        return all_features


# ========== Extract Features for All Datasets ==========

def extract_features_from_dataframe(df, feature_extractor, include_kmer=True):
    """
    Extract features for all sequences in a DataFrame.
    
    Args:
        df: Pandas DataFrame with 'sequence' column
        feature_extractor: PromoterFeatureExtractor instance
        include_kmer: Whether to include k-mer features
        
    Returns:
        pd.DataFrame: Feature matrix
    """
    print(f"\n🔧 Extracting features from {len(df)} sequences...")
    print(f"   K-mer features: {'Enabled' if include_kmer else 'Disabled'}")
    
    features_list = []
    
    for idx, row in df.iterrows():
        sequence = row['sequence']
        features = feature_extractor.extract_all_features(sequence, include_kmer=include_kmer)
        features_list.append(features)
        
        # Progress indicator
        if (idx + 1) % 1000 == 0:
            print(f"   Processed: {idx + 1}/{len(df)}", end='\r')
    
    print(f"   Processed: {len(df)}/{len(df)}")
    
    # Convert to DataFrame
    features_df = pd.DataFrame(features_list)
    
    print(f"✅ Feature extraction complete!")
    print(f"   Total features: {len(features_df.columns)}")
    print(f"   Feature matrix shape: {features_df.shape}")
    
    return features_df


# Initialize feature extractor
print("="*70)
print("🔬 INITIALIZING FEATURE EXTRACTOR")
print("="*70)
feature_extractor = PromoterFeatureExtractor()


# Extract features for train/val/test sets
print("\n" + "="*70)
print("🔧 EXTRACTING FEATURES FOR ALL DATASETS")
print("="*70)

# For initial testing, we'll use k-mer features up to k=3 (balance between performance and speed)
# Set include_kmer=False for faster processing with fewer features
INCLUDE_KMER = True  # Set to False for faster processing

X_train_features = extract_features_from_dataframe(df_train, feature_extractor, include_kmer=INCLUDE_KMER)
X_val_features = extract_features_from_dataframe(df_val, feature_extractor, include_kmer=INCLUDE_KMER)
X_test_features = extract_features_from_dataframe(df_test, feature_extractor, include_kmer=INCLUDE_KMER)

# Get labels
y_train = df_train['label'].values
y_val = df_val['label'].values
y_test = df_test['label'].values

print("\n" + "="*70)
print("📊 FEATURE EXTRACTION SUMMARY")
print("="*70)
print(f"Train set: {X_train_features.shape[0]} samples × {X_train_features.shape[1]} features")
print(f"Val set:   {X_val_features.shape[0]} samples × {X_val_features.shape[1]} features")
print(f"Test set:  {X_test_features.shape[0]} samples × {X_test_features.shape[1]} features")
print("="*70)

# Display sample features
print("\n📋 Sample Features (first 10):")
display(X_train_features.head())

# Feature statistics
print("\n📈 Feature Statistics:")
print(X_train_features.describe())

print("\n✅ CELL 3 COMPLETE: Feature extraction finished!")
print(f"   Total features extracted: {X_train_features.shape[1]}")
print("\nNext: Traditional ML Models (Cell 4)")
