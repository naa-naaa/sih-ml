"""
DATASET DIAGNOSTIC TOOL
=======================
Run this BEFORE training to check if your dataset is adequate
"""

import pandas as pd
import numpy as np
from collections import Counter

def diagnose_dataset(csv_path):
    """
    Comprehensive dataset analysis to identify issues
    """
    print("=" * 70)
    print("DATASET DIAGNOSTIC REPORT")
    print("=" * 70)
    
    # Load training dataset
    df = pd.read_csv(csv_path)
    
    print(f"\n📊 BASIC STATISTICS:")
    print(f"  Total districts: {len(df)}")
    
    # Check if this is raw time-series data or processed features
    if 'status' not in df.columns:
        print(f"\n❌ Column 'status' not found in the dataset.")
        print(f"   Available columns: {list(df.columns)}")
        
        if 'date' in df.columns and 'gwl' in df.columns:
            print(f"\nℹ️  This appears to be RAW TIME-SERIES data, not processed features!")
            print(f"   - Total records: {len(df):,}")
            print(f"   - Unique districts: {df['district'].nunique() if 'district' in df.columns else 'N/A'}")
            print(f"   - Date range: {df['date'].min()} to {df['date'].max()}" if 'date' in df.columns else "")
            print(f"\n💡 You need to run the TRAINING SCRIPT first to generate:")
            print(f"   - extracted_features.csv (features per district)")
            print(f"   - training_dataset.csv (features + labels)")
            print(f"\n   Then run this diagnostic on 'training_dataset.csv'")
            return None
        
        print(f"\nℹ️  Creating synthetic 'status' column from 'gwl' (tertiles: Low/Medium/High)")
        df['status'] = pd.qcut(df['gwl'], q=3, labels=['Low', 'Medium', 'High'])
    
    if 'gwl_avg' in df.columns:
        print(f"  ✅ Detected PROCESSED FEATURES dataset")
        print(f"  Features: gwl_avg, trend_slope, rainfall_avg, recharge_avg")
    else:
        print(f"  Features: {', '.join([c for c in df.columns if c not in ['date', 'state', 'district', 'status']])[:100]}")
    
    # Check class distribution
    print(f"\n📈 CLASS DISTRIBUTION:")
    class_counts = df['status'].value_counts()
    print(class_counts)
    
    total = len(df)
    print(f"\n  Percentages:")
    for status, count in class_counts.items():
        pct = count / total * 100
        print(f"    {status:20s}: {count:3d} ({pct:5.1f}%)")
    
    # Check for severe imbalance
    print(f"\n⚠️  IMBALANCE CHECK:")
    max_class = class_counts.max()
    min_class = class_counts.min()
    imbalance_ratio = max_class / min_class
    
    if imbalance_ratio > 5:
        print(f"  ❌ SEVERE IMBALANCE (ratio: {imbalance_ratio:.1f}:1)")
        print(f"     Largest class: {max_class} samples")
        print(f"     Smallest class: {min_class} samples")
        print(f"     Impact: Model will be biased toward majority class")
    elif imbalance_ratio > 2:
        print(f"  ⚠️  MODERATE IMBALANCE (ratio: {imbalance_ratio:.1f}:1)")
        print(f"     Consider using class weights or resampling")
    else:
        print(f"  ✅ BALANCED (ratio: {imbalance_ratio:.1f}:1)")
    
    # Check dataset size adequacy
    print(f"\n📏 DATASET SIZE ASSESSMENT:")
    n_classes = len(class_counts)
    min_recommended = n_classes * 20  # At least 20 samples per class
    optimal_recommended = n_classes * 50  # Optimal: 50+ per class
    
    if total < min_recommended:
        print(f"  ❌ CRITICALLY SMALL ({total} samples)")
        print(f"     Minimum recommended: {min_recommended} samples")
        print(f"     Current: {total} samples")
        print(f"     RECOMMENDATION: Collect more data or use cross-validation")
    elif total < optimal_recommended:
        print(f"  ⚠️  SMALL ({total} samples)")
        print(f"     Optimal: {optimal_recommended}+ samples")
        print(f"     RECOMMENDATION: Collect more data if possible")
    else:
        print(f"  ✅ ADEQUATE ({total} samples)")
    
    # Simulate train/test split
    print(f"\n🔄 SIMULATED 80/20 SPLIT:")
    train_size = int(total * 0.8)
    test_size = total - train_size
    print(f"  Training: {train_size} samples")
    print(f"  Testing: {test_size} samples")
    
    if test_size < n_classes * 2:
        print(f"  ❌ WARNING: Test set too small!")
        print(f"     Need at least 2 samples per class")
        print(f"     You'll have ~{test_size} samples for {n_classes} classes")
    else:
        print(f"  ✅ Test set size adequate")
    
    # Check per-class test set size
    print(f"\n  Expected test samples per class:")
    for status, count in class_counts.items():
        expected_test = int(count * 0.2)
        if expected_test == 0:
            print(f"    {status:20s}: {expected_test} samples ❌ (class will be missing!)")
        elif expected_test < 2:
            print(f"    {status:20s}: {expected_test} samples ⚠️  (too few!)")
        else:
            print(f"    {status:20s}: {expected_test} samples ✅")
    
    # Feature quality check
    print(f"\n🔍 FEATURE QUALITY:")
    feature_cols = ['gwl_avg', 'trend_slope', 'rainfall_avg', 'recharge_avg']
    
    features_found = False
    for col in feature_cols:
        if col in df.columns:
            features_found = True
            missing = df[col].isna().sum()
            zeros = (df[col] == 0).sum()
            std = df[col].std()
            print(f"  {col:20s}: Missing={missing:3d}, Zeros={zeros:3d}, StdDev={std:.2f}")
            
            if missing > 0:
                print(f"    ⚠️  Has {missing} missing values")
            if zeros > total * 0.5:
                print(f"    ⚠️  {zeros} zero values (>{50}% of data)")
    
    if not features_found:
        print(f"  ⚠️  Standard features (gwl_avg, trend_slope, etc.) not found")
        print(f"     This may be raw time-series data instead of processed features")
    
    # Overall recommendation
    print(f"\n" + "=" * 70)
    print("🎯 OVERALL RECOMMENDATION:")
    print("=" * 70)
    
    issues = []
    
    if total < min_recommended:
        issues.append("Dataset too small - collect more data")
    
    if imbalance_ratio > 5:
        issues.append("Severe class imbalance - use class weights or SMOTE")
    
    if test_size < n_classes * 2:
        issues.append("Test set will be too small - use cross-validation")
    
    if any(int(count * 0.2) < 1 for count in class_counts.values()):
        issues.append("Some classes will be missing from test set")
    
    if not features_found and 'date' in df.columns:
        issues.append("Using raw time-series data - need to extract features first")
    
    if issues:
        print("❌ ISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        
        print("\n💡 SUGGESTED ACTIONS:")
        if not features_found and 'date' in df.columns:
            print("  1. RUN THE TRAINING SCRIPT to extract features from time-series data")
            print("  2. This will create 'training_dataset.csv' with processed features")
        
        if total < 50:
            print("  3. USE CROSS-VALIDATION instead of train/test split")
            print("  4. Collect more district data (target: 100+ districts)")
        
        if imbalance_ratio > 5:
            print("  5. Apply SMOTE or class weights during training")
        
        if n_classes > 4:
            print("  6. Consider combining similar classes (e.g., Critical + Over-Exploited)")
    else:
        print("✅ Dataset appears adequate for training!")
        print("   Proceed with standard 80/20 train/test split")
    
    return df


def recommend_approach(df):
    """Recommend best training approach based on dataset size"""
    if df is None:
        return
    
    n_samples = len(df)
    n_classes = df['status'].nunique()
    
    print("\n" + "=" * 70)
    print("📋 RECOMMENDED TRAINING APPROACH:")
    print("=" * 70)
    
    if n_samples < 30:
        print("🔄 USE LEAVE-ONE-OUT CROSS-VALIDATION")
        print("   - Dataset too small for train/test split")
        print("   - Each district tested individually")
        
    elif n_samples < 50:
        print("🔄 USE 5-FOLD CROSS-VALIDATION")
        print("   - Better than single split for small datasets")
        print("   - Each fold gets tested exactly once")
        
    elif n_samples < 100:
        print("🔄 USE 10-FOLD CROSS-VALIDATION OR 80/20 SPLIT")
        print("   - Cross-validation gives more reliable metrics")
        print("   - Single split acceptable but less robust")
        
    else:
        print("✅ USE STANDARD 80/20 TRAIN/TEST SPLIT")
        print("   - Dataset large enough for reliable evaluation")


if __name__ == "__main__":
    import sys
    
    # Check if file path provided
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        # Try multiple possible files
        possible_files = [
            'i2final_lstm.csv',
            'C:\\codes\\PyTorchTest2\\i2final_lstm.csv'
        ]
        
        csv_path = None
        for path in possible_files:
            try:
                import os
                if os.path.exists(path):
                    csv_path = path
                    print(f"Found dataset: {path}\n")
                    break
            except:
                continue
        
        if csv_path is None:
            print("❌ No dataset file found!")
            print("\nUsage: python diagnostic.py <path_to_csv>")
            print("\nLooking for one of:")
            for f in possible_files:
                print(f"  - {f}")
            sys.exit(1)
    
    try:
        df = diagnose_dataset(csv_path)
        recommend_approach(df)
    except FileNotFoundError:
        print(f"❌ File not found: {csv_path}")
        print("   Make sure the file path is correct")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()