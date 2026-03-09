"""
==============================================================================
CELL 2: EXPLORATORY DATA ANALYSIS (EDA) AND VISUALIZATION
==============================================================================
Objective: Understand the data distribution, sequence characteristics,
           and biological patterns in promoters vs non-promoters
           
Key Analyses:
1. Class distribution and imbalance
2. Sequence length distribution
3. Nucleotide composition (overall and positional)
4. K-mer analysis
5. Motif detection (-35 and -10 boxes)
6. GC content analysis
==============================================================================
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from collections import Counter
import re

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ========== Class Distribution Analysis ==========
def plot_class_distribution(df_train, df_val, df_test):
    """
    Visualize the distribution of promoter vs non-promoter sequences.
    
    Args:
        df_train: Training DataFrame
        df_val: Validation DataFrame
        df_test: Test DataFrame
        
    Purpose: Check for class imbalance which affects:
    - Choice of evaluation metrics (prefer MCC, F1, AUROC over accuracy)
    - Need for class balancing techniques (SMOTE, class weights)
    - Sampling strategies
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    splits = [
        (df_train, 'Train', axes[0]),
        (df_val, 'Validation', axes[1]),
        (df_test, 'Test', axes[2])
    ]
    
    for df, split_name, ax in splits:
        counts = df['label'].value_counts().sort_index()
        
        # Bar plot
        bars = ax.bar(['Non-promoter (0)', 'Promoter (1)'], counts.values, 
                      color=['#ff7f0e', '#2ca02c'], alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(count):,}\n({count/len(df)*100:.1f}%)',
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'{split_name} Set\nTotal: {len(df):,} sequences', 
                     fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Class Distribution Across Splits', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
    
    # Calculate imbalance ratio
    train_ratio = df_train['label'].value_counts().values
    imbalance_ratio = train_ratio[0] / train_ratio[1] if len(train_ratio) > 1 else 1
    
    print(f"\n📊 Class Imbalance Analysis:")
    print(f"  Imbalance Ratio (Non-promoter/Promoter): {imbalance_ratio:.2f}:1")
    
    if imbalance_ratio > 1.5:
        print(f"  ⚠️  Dataset is imbalanced!")
        print(f"  Recommendations:")
        print(f"    - Use stratified sampling")
        print(f"    - Apply class weights in loss function")
        print(f"    - Use metrics: MCC, F1, AUROC (not just accuracy)")
        print(f"    - Consider SMOTE or other resampling techniques")
    else:
        print(f"  ✅ Dataset is relatively balanced")

plot_class_distribution(df_train, df_val, df_test)


# ========== Sequence Length Distribution ==========
def plot_sequence_length_distribution(df_train):
    """
    Analyze and visualize sequence length distribution.
    
    Biological Significance:
    - Bacterial promoters are typically 50-100 bp upstream of TSS
    - Consistent length suggests standardized sequence windows
    - Variable length requires padding/truncation strategies
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Calculate lengths
    df_train['seq_length'] = df_train['sequence'].apply(len)
    
    # Histogram
    for label, color, name in [(0, '#ff7f0e', 'Non-promoter'), (1, '#2ca02c', 'Promoter')]:
        lengths = df_train[df_train['label'] == label]['seq_length']
        axes[0].hist(lengths, bins=50, alpha=0.6, color=color, label=name, edgecolor='black')
    
    axes[0].set_xlabel('Sequence Length (bp)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Sequence Length Distribution', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Box plot
    data_to_plot = [
        df_train[df_train['label'] == 0]['seq_length'],
        df_train[df_train['label'] == 1]['seq_length']
    ]
    bp = axes[1].boxplot(data_to_plot, labels=['Non-promoter', 'Promoter'],
                          patch_artist=True, widths=0.6)
    
    for patch, color in zip(bp['boxes'], ['#ff7f0e', '#2ca02c']):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    axes[1].set_ylabel('Sequence Length (bp)', fontsize=12)
    axes[1].set_title('Sequence Length by Class', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Statistics
    print("\n📏 Sequence Length Statistics:")
    for label, name in [(0, 'Non-promoter'), (1, 'Promoter')]:
        lengths = df_train[df_train['label'] == label]['seq_length']
        print(f"\n  {name}:")
        print(f"    Min: {lengths.min()} bp")
        print(f"    Max: {lengths.max()} bp")
        print(f"    Mean: {lengths.mean():.2f} bp")
        print(f"    Median: {lengths.median():.2f} bp")
        print(f"    Std: {lengths.std():.2f} bp")

plot_sequence_length_distribution(df_train)


# ========== Nucleotide Composition Analysis ==========
def plot_nucleotide_composition(df_train):
    """
    Compare nucleotide composition between promoters and non-promoters.
    
    Biological Rationale:
    - Promoters are often AT-rich (especially -10 box: TATAAT)
    - -35 box (TTGACA) has specific AT/GC balance
    - Different composition helps distinguish promoters
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Calculate nucleotide percentages
    def get_nucleotide_composition(sequences):
        composition = {'A': 0, 'T': 0, 'G': 0, 'C': 0}
        total = 0
        
        for seq in sequences:
            for base in seq:
                if base in composition:
                    composition[base] += 1
                    total += 1
        
        return {base: count/total*100 for base, count in composition.items()}
    
    # Get compositions
    promoter_seqs = df_train[df_train['label'] == 1]['sequence']
    non_promoter_seqs = df_train[df_train['label'] == 0]['sequence']
    
    promoter_comp = get_nucleotide_composition(promoter_seqs)
    non_promoter_comp = get_nucleotide_composition(non_promoter_seqs)
    
    # Grouped bar chart
    x = np.arange(4)
    width = 0.35
    bases = ['A', 'T', 'G', 'C']
    
    promoter_vals = [promoter_comp[b] for b in bases]
    non_promoter_vals = [non_promoter_comp[b] for b in bases]
    
    bars1 = axes[0].bar(x - width/2, non_promoter_vals, width, label='Non-promoter',
                        color='#ff7f0e', alpha=0.7, edgecolor='black')
    bars2 = axes[0].bar(x + width/2, promoter_vals, width, label='Promoter',
                        color='#2ca02c', alpha=0.7, edgecolor='black')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    axes[0].set_ylabel('Percentage (%)', fontsize=12)
    axes[0].set_xlabel('Nucleotide', fontsize=12)
    axes[0].set_title('Nucleotide Composition Comparison', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(bases)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # GC content comparison
    promoter_gc = promoter_comp['G'] + promoter_comp['C']
    non_promoter_gc = non_promoter_comp['G'] + non_promoter_comp['C']
    
    bars = axes[1].bar(['Non-promoter', 'Promoter'], 
                       [non_promoter_gc, promoter_gc],
                       color=['#ff7f0e', '#2ca02c'], alpha=0.7, edgecolor='black')
    
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    axes[1].set_ylabel('GC Content (%)', fontsize=12)
    axes[1].set_title('GC Content Comparison', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_ylim([0, max(non_promoter_gc, promoter_gc) * 1.2])
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print("\n🧬 Nucleotide Composition Analysis:")
    print(f"\n  Non-promoters:")
    for base in bases:
        print(f"    {base}: {non_promoter_comp[base]:.2f}%")
    print(f"    GC Content: {non_promoter_gc:.2f}%")
    
    print(f"\n  Promoters:")
    for base in bases:
        print(f"    {base}: {promoter_comp[base]:.2f}%")
    print(f"    GC Content: {promoter_gc:.2f}%")
    
    print(f"\n  Difference (Promoter - Non-promoter):")
    for base in bases:
        diff = promoter_comp[base] - non_promoter_comp[base]
        print(f"    {base}: {diff:+.2f}%")
    print(f"    GC Content: {promoter_gc - non_promoter_gc:+.2f}%")

plot_nucleotide_composition(df_train)


# ========== Motif Detection: -35 and -10 Boxes ==========
def detect_promoter_motifs(df_train, sample_size=1000):
    """
    Search for canonical bacterial promoter motifs.
    
    Key Motifs:
    - -35 box: TTGACA (consensus sequence)
    - -10 box: TATAAT (Pribnow box)
    - Spacer: 16-18 bp between motifs (optimal)
    
    Uses fuzzy matching to allow for variations.
    """
    print("\n" + "="*70)
    print("🔍 SEARCHING FOR PROMOTER MOTIFS")
    print("="*70)
    
    # Consensus sequences
    consensus_35 = "TTGACA"
    consensus_10 = "TATAAT"
    
    # Function to find motif with mismatches
    def find_motif_fuzzy(sequence, motif, max_mismatches=2):
        """Find motif allowing up to max_mismatches differences"""
        positions = []
        motif_len = len(motif)
        
        for i in range(len(sequence) - motif_len + 1):
            subseq = sequence[i:i+motif_len]
            mismatches = sum(1 for a, b in zip(subseq, motif) if a != b)
            if mismatches <= max_mismatches:
                positions.append((i, mismatches))
        
        return positions
    
    # Sample sequences
    promoter_sample = df_train[df_train['label'] == 1].sample(min(sample_size, len(df_train[df_train['label'] == 1])))
    non_promoter_sample = df_train[df_train['label'] == 0].sample(min(sample_size, len(df_train[df_train['label'] == 0])))
    
    # Count motif occurrences
    stats = {
        'Promoter': {'box_35': 0, 'box_10': 0, 'both': 0, 'total': len(promoter_sample)},
        'Non-promoter': {'box_35': 0, 'box_10': 0, 'both': 0, 'total': len(non_promoter_sample)}
    }
    
    for label, sample, name in [(1, promoter_sample, 'Promoter'), 
                                 (0, non_promoter_sample, 'Non-promoter')]:
        for seq in sample['sequence']:
            found_35 = len(find_motif_fuzzy(seq, consensus_35, max_mismatches=2)) > 0
            found_10 = len(find_motif_fuzzy(seq, consensus_10, max_mismatches=2)) > 0
            
            if found_35:
                stats[name]['box_35'] += 1
            if found_10:
                stats[name]['box_10'] += 1
            if found_35 and found_10:
                stats[name]['both'] += 1
    
    # Print results
    print(f"\n📊 Motif Detection Results (max 2 mismatches, n={sample_size}):")
    print("\n  Promoters:")
    print(f"    -35 box (TTGACA): {stats['Promoter']['box_35']:,} ({stats['Promoter']['box_35']/stats['Promoter']['total']*100:.1f}%)")
    print(f"    -10 box (TATAAT): {stats['Promoter']['box_10']:,} ({stats['Promoter']['box_10']/stats['Promoter']['total']*100:.1f}%)")
    print(f"    Both motifs: {stats['Promoter']['both']:,} ({stats['Promoter']['both']/stats['Promoter']['total']*100:.1f}%)")
    
    print("\n  Non-promoters:")
    print(f"    -35 box (TTGACA): {stats['Non-promoter']['box_35']:,} ({stats['Non-promoter']['box_35']/stats['Non-promoter']['total']*100:.1f}%)")
    print(f"    -10 box (TATAAT): {stats['Non-promoter']['box_10']:,} ({stats['Non-promoter']['box_10']/stats['Non-promoter']['total']*100:.1f}%)")
    print(f"    Both motifs: {stats['Non-promoter']['both']:,} ({stats['Non-promoter']['both']/stats['Non-promoter']['total']*100:.1f}%)")
    
    # Enrichment
    enrichment_35 = (stats['Promoter']['box_35']/stats['Promoter']['total']) / (stats['Non-promoter']['box_35']/stats['Non-promoter']['total']) if stats['Non-promoter']['box_35'] > 0 else float('inf')
    enrichment_10 = (stats['Promoter']['box_10']/stats['Promoter']['total']) / (stats['Non-promoter']['box_10']/stats['Non-promoter']['total']) if stats['Non-promoter']['box_10'] > 0 else float('inf')
    
    print(f"\n  Enrichment in Promoters:")
    print(f"    -35 box: {enrichment_35:.2f}x")
    print(f"    -10 box: {enrichment_10:.2f}x")
    
    print("="*70)

detect_promoter_motifs(df_train)


# ========== K-mer Frequency Analysis ==========
def plot_kmer_analysis(df_train, k=3, top_n=15):
    """
    Analyze and compare k-mer frequencies between promoters and non-promoters.
    
    Args:
        k: K-mer length (3 for trinucleotides, 4 for tetranucleotides, etc.)
        top_n: Number of top k-mers to display
        
    Biological Significance:
    - Different k-mer patterns distinguish promoters
    - AT-rich k-mers often enriched in promoters
    - Used as features for ML models
    """
    print(f"\n🔬 Analyzing {k}-mer Frequencies...")
    
    # Function to extract k-mers
    def get_kmers(sequence, k):
        return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    
    # Count k-mers for each class
    promoter_kmers = Counter()
    non_promoter_kmers = Counter()
    
    for _, row in df_train.iterrows():
        kmers = get_kmers(row['sequence'], k)
        if row['label'] == 1:
            promoter_kmers.update(kmers)
        else:
            non_promoter_kmers.update(kmers)
    
    # Normalize by total counts
    promoter_total = sum(promoter_kmers.values())
    non_promoter_total = sum(non_promoter_kmers.values())
    
    promoter_kmers_norm = {kmer: count/promoter_total for kmer, count in promoter_kmers.items()}
    non_promoter_kmers_norm = {kmer: count/non_promoter_total for kmer, count in non_promoter_kmers.items()}
    
    # Find most enriched k-mers in promoters
    enrichment = {}
    for kmer in promoter_kmers_norm:
        if kmer in non_promoter_kmers_norm and non_promoter_kmers_norm[kmer] > 0:
            enrichment[kmer] = promoter_kmers_norm[kmer] / non_promoter_kmers_norm[kmer]
    
    # Get top enriched k-mers
    top_enriched = sorted(enrichment.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    kmers = [item[0] for item in top_enriched]
    ratios = [item[1] for item in top_enriched]
    
    bars = ax.barh(kmers, ratios, color='#2ca02c', alpha=0.7, edgecolor='black')
    ax.axvline(x=1, color='red', linestyle='--', linewidth=2, label='Equal frequency')
    
    ax.set_xlabel('Enrichment Ratio (Promoter / Non-promoter)', fontsize=12)
    ax.set_ylabel(f'{k}-mer', fontsize=12)
    ax.set_title(f'Top {top_n} Most Enriched {k}-mers in Promoters', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n  Top {top_n} enriched {k}-mers in promoters:")
    for i, (kmer, ratio) in enumerate(top_enriched, 1):
        print(f"    {i}. {kmer}: {ratio:.2f}x enriched")

plot_kmer_analysis(df_train, k=3, top_n=15)
plot_kmer_analysis(df_train, k=4, top_n=15)


print("\n✅ CELL 2 COMPLETE: Exploratory Data Analysis finished!")
print("Key Insights:")
print("  1. Class distribution and imbalance assessed")
print("  2. Sequence length characteristics understood")
print("  3. Nucleotide composition differences identified")
print("  4. Promoter motifs (-35/-10 boxes) detected")
print("  5. K-mer enrichment patterns analyzed")
print("\nNext: Feature Engineering (Cell 3)")
