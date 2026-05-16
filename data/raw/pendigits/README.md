# Pen-Based Recognition of Handwritten Digits (Signals / Handwriting)

**Source**: UCI ML Repository (id=81)
**Entity column**: `row index (one row per drawn digit)`
**Use case for this pipeline**: Cluster handwritten-digit pen-stroke signals; expect ~10 natural clusters (one per digit 0-9)

## Description
10,992 handwritten digits, each represented by 16 features: 8 (x, y) coordinate pairs sampled along the pen trajectory. Resampled to a common length and normalised. Plus the digit label (0-9).
