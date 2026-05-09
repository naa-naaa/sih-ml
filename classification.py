"""
K-FOLD CROSS-VALIDATION FOR SMALL DATASETS
==========================================
Enhanced training with stratified K-fold for reliable accuracy on limited data
"""
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import pandas as pd
import numpy as np
from scipy.stats import theilslopes
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

# ============================================
# DATA LOADING & FEATURE EXTRACTION
# (Keep your existing functions)
# ============================================

def load_and_prepare_data(csv_path):
    """Load dataset"""
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"✅ Loaded {len(df):,} records")
    print(f"✅ States: {df['state'].nunique()}, Districts: {df['district'].nunique()}")
    
    return df


def calculate_trend_slope(gwl_series, dates):
    """Calculate Sen's Slope"""
    mask = ~np.isnan(gwl_series)
    gwl_clean = gwl_series[mask]
    dates_clean = dates[mask]
    
    if len(gwl_clean) < 24:
        return np.nan
    
    df_temp = pd.DataFrame({'date': dates_clean, 'gwl': gwl_clean})
    monthly_gwl = df_temp.set_index('date').resample('M')['gwl'].mean()
    monthly_gwl = monthly_gwl.dropna()
    
    if len(monthly_gwl) < 24:
        return np.nan
    
    time_index = np.arange(len(monthly_gwl))
    slope, intercept, _, _ = theilslopes(monthly_gwl.values, time_index)
    annual_slope = slope * 12
    
    return annual_slope


def extract_features_per_district(df, min_years=3):
    """Extract features for each district"""
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING")
    print("=" * 70)
    
    features_list = []
    grouped = df.groupby(['state', 'district'])
    
    for (state, district), group_df in grouped:
        if group_df['year'].nunique() < min_years:
            continue
        
        trend_slope = calculate_trend_slope(
            group_df['gwl'].values,
            group_df['date'].values
        )
        
        if np.isnan(trend_slope):
            continue
        
        features_list.append({
            'state': state,
            'district': district,
            'gwl_avg': group_df['gwl'].mean(),
            'rainfall_avg': group_df.groupby('year')['rainfall'].sum().mean(),
            'recharge_avg': group_df.groupby('year')['effective_recharge'].sum().mean(),
            'trend_slope': trend_slope
        })
    
    features_df = pd.DataFrame(features_list)
    print(f"✅ Extracted features for {len(features_df)} districts")
    
    return features_df


def apply_rule_based_labels(features_df):
    """Classify districts"""
    def classify(row):
        gwl = row['gwl_avg']
        trend = row['trend_slope']
        recharge = row['recharge_avg']
        rainfall = row['rainfall_avg']
        
        if gwl > 30 and trend < -0.5:
            return 'Over-Exploited'
        if gwl > 25 and trend < -0.3:
            return 'Critical'
        if gwl > 20 and trend < -0.2 and recharge < 300:
            return 'Critical'
        if gwl > 15 or trend < -0.15 or rainfall < 800:
            return 'Semi-Critical'
        return 'Safe'
    
    features_df['status'] = features_df.apply(classify, axis=1)
    return features_df


# ============================================
# K-FOLD CROSS-VALIDATION TRAINING
# ============================================

def create_model(n_features, n_classes):
    """Build neural network"""
    model = Sequential([
        Dense(16, activation='relu', input_shape=(n_features,)),
        Dropout(0.3),
        Dense(8, activation='relu'),
        Dropout(0.2),
        Dense(n_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_with_kfold(features_df, n_splits=5, use_recharge=True):
    """
    Train model using Stratified K-Fold Cross-Validation
    
    Parameters:
    -----------
    features_df : DataFrame
        Features with labels
    n_splits : int
        Number of folds (recommended: 5 for 25 samples)
    use_recharge : bool
        Include recharge feature
    
    Returns:
    --------
    Best model, metrics, and all fold results
    """
    print("\n" + "=" * 70)
    print(f"K-FOLD CROSS-VALIDATION TRAINING ({n_splits} folds)")
    print("=" * 70)
    
    # Define features
    if use_recharge:
        FEATURE_COLS = ['gwl_avg', 'trend_slope', 'rainfall_avg', 'recharge_avg']
    else:
        FEATURE_COLS = ['gwl_avg', 'trend_slope', 'rainfall_avg']
    
    print(f"Features: {FEATURE_COLS}")
    
    # Prepare data
    X = features_df[FEATURE_COLS].values
    y = features_df['status'].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"\nDataset: {len(features_df)} districts")
    print(f"Classes: {len(label_encoder.classes_)}")
    print(f"\nClass distribution:")
    for i, class_name in enumerate(label_encoder.classes_):
        count = np.sum(y_encoded == i)
        print(f"  {class_name:20s}: {count} samples ({count/len(y)*100:.1f}%)")
    
    # Initialize K-Fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Storage for results
    fold_results = []
    all_y_true = []
    all_y_pred = []
    fold_histories = []
    best_model = None
    best_accuracy = 0
    
    print(f"\n🔄 Starting {n_splits}-Fold Cross-Validation...")
    print("=" * 70)
    
    # Train on each fold
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded), 1):
        print(f"\n{'='*70}")
        print(f"FOLD {fold}/{n_splits}")
        print(f"{'='*70}")
        
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]
        
        print(f"Training samples: {len(X_train)} | Validation samples: {len(X_val)}")
        
        # Normalize features (fit on train, transform both)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Create fresh model for this fold
        model = create_model(len(FEATURE_COLS), len(label_encoder.classes_))
        
        # Train
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=0
        )
        
        history = model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=100,
            batch_size=8,  # Smaller batch for small dataset
            callbacks=[early_stop],
            verbose=0
        )
        
        fold_histories.append(history)
        
        # Evaluate on validation set
        y_pred_proba = model.predict(X_val_scaled, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate metrics for this fold
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
        
        fold_results.append({
            'fold': fold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'train_samples': len(X_train),
            'val_samples': len(X_val)
        })
        
        print(f"✅ Fold {fold} Results:")
        print(f"   Accuracy:  {accuracy*100:.2f}%")
        print(f"   Precision: {precision*100:.2f}%")
        print(f"   Recall:    {recall*100:.2f}%")
        print(f"   F1-Score:  {f1*100:.2f}%")
        
        # Track best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_scaler = scaler
        
        # Store predictions for overall confusion matrix
        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)
    
    # ============================================
    # AGGREGATE RESULTS ACROSS ALL FOLDS
    # ============================================
    
    print(f"\n{'='*70}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*70}")
    
    results_df = pd.DataFrame(fold_results)
    
    print(f"\n📊 PERFORMANCE ACROSS {n_splits} FOLDS:")
    print(results_df.to_string(index=False))
    
    print(f"\n📈 AVERAGE METRICS (Mean ± Std):")
    print(f"  Accuracy:  {results_df['accuracy'].mean()*100:.2f}% ± {results_df['accuracy'].std()*100:.2f}%")
    print(f"  Precision: {results_df['precision'].mean()*100:.2f}% ± {results_df['precision'].std()*100:.2f}%")
    print(f"  Recall:    {results_df['recall'].mean()*100:.2f}% ± {results_df['recall'].std()*100:.2f}%")
    print(f"  F1-Score:  {results_df['f1_score'].mean()*100:.2f}% ± {results_df['f1_score'].std()*100:.2f}%")
    
    # Overall confusion matrix
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    
    cm = confusion_matrix(all_y_true, all_y_pred)
    
    print(f"\n📋 OVERALL CLASSIFICATION REPORT:")
    y_true_labels = label_encoder.inverse_transform(all_y_true)
    y_pred_labels = label_encoder.inverse_transform(all_y_pred)
    print(classification_report(y_true_labels, y_pred_labels, zero_division=0))
    
    print(f"\n📈 OVERALL CONFUSION MATRIX:")
    cm_df = pd.DataFrame(
        cm, 
        index=label_encoder.classes_, 
        columns=label_encoder.classes_
    )
    print(cm_df)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title(f'{n_splits}-Fold Cross-Validation - Overall Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('kfold_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: kfold_confusion_matrix.png")
    plt.close()
    
    # Plot fold performance
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    x = range(1, n_splits + 1)
    plt.plot(x, results_df['accuracy']*100, 'o-', label='Accuracy', linewidth=2, markersize=8)
    plt.plot(x, results_df['f1_score']*100, 's-', label='F1-Score', linewidth=2, markersize=8)
    plt.axhline(results_df['accuracy'].mean()*100, color='blue', linestyle='--', alpha=0.5)
    plt.axhline(results_df['f1_score'].mean()*100, color='orange', linestyle='--', alpha=0.5)
    plt.xlabel('Fold')
    plt.ylabel('Score (%)')
    plt.title('Performance Across Folds')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(x)
    
    plt.subplot(1, 2, 2)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    means = [
        results_df['accuracy'].mean()*100,
        results_df['precision'].mean()*100,
        results_df['recall'].mean()*100,
        results_df['f1_score'].mean()*100
    ]
    stds = [
        results_df['accuracy'].std()*100,
        results_df['precision'].std()*100,
        results_df['recall'].std()*100,
        results_df['f1_score'].std()*100
    ]
    plt.bar(metrics, means, yerr=stds, capsize=5, alpha=0.7, color=['blue', 'green', 'orange', 'red'])
    plt.ylabel('Score (%)')
    plt.title('Average Metrics with Std Dev')
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('kfold_performance.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: kfold_performance.png")
    plt.close()
    
    print(f"\n🏆 Best Model: Fold with {best_accuracy*100:.2f}% accuracy")
    
    # Save results
    results_df.to_csv('kfold_results.csv', index=False)
    print(f"✅ Saved: kfold_results.csv")
    
    return best_model, best_scaler, label_encoder, FEATURE_COLS, results_df


# ============================================
# FINAL MODEL TRAINING ON ALL DATA
# ============================================

def train_final_model(features_df, scaler, label_encoder, feature_cols):
    """
    After K-fold validation, train final model on ALL data
    """
    print(f"\n{'='*70}")
    print("TRAINING FINAL MODEL ON ALL DATA")
    print(f"{'='*70}")
    
    X = features_df[feature_cols].values
    y = features_df['status'].values
    y_encoded = label_encoder.transform(y)
    
    X_scaled = scaler.fit_transform(X)
    
    model = create_model(len(feature_cols), len(label_encoder.classes_))
    
    early_stop = EarlyStopping(
        monitor='loss',
        patience=20,
        restore_best_weights=True,
        verbose=0
    )
    
    print(f"Training on all {len(X)} samples...")
    
    history = model.fit(
        X_scaled, y_encoded,
        epochs=150,
        batch_size=8,
        callbacks=[early_stop],
        verbose=1
    )
    
    print(f"✅ Final model trained")
    
    return model, scaler


# ============================================
# SAVE MODEL
# ============================================

def save_model_artifacts(model, scaler, label_encoder, feature_cols, metrics_df):
    """Save trained model and artifacts"""
    print(f"\n{'='*70}")
    print("SAVING MODEL ARTIFACTS")
    print(f"{'='*70}")
    
    model.save('groundwater_model.keras')
    print("✅ Saved: groundwater_model.keras")
    
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("✅ Saved: scaler.pkl")
    
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    print("✅ Saved: label_encoder.pkl")
    
    with open('feature_columns.pkl', 'wb') as f:
        pickle.dump(feature_cols, f)
    print("✅ Saved: feature_columns.pkl")


# ============================================
# MAIN PIPELINE
# ============================================

def main(csv_path, gec_labels_path=None, n_folds=5, use_recharge=True):
    """
    Complete K-Fold training pipeline
    
    Parameters:
    -----------
    csv_path : str
        Path to your data CSV
    gec_labels_path : str, optional
        Path to GEC labels (if available)
    n_folds : int
        Number of folds (5-10 recommended for 25 samples)
    use_recharge : bool
        Include recharge feature
    """
    print("\n" + "=" * 70)
    print("GROUNDWATER CLASSIFICATION - K-FOLD TRAINING")
    print("=" * 70)
    
    # Load and process data
    df = load_and_prepare_data(csv_path)
    features_df = extract_features_per_district(df, min_years=3)
    features_df = apply_rule_based_labels(features_df)
    
    # K-Fold Cross-Validation
    best_model, scaler, encoder, features, results = train_with_kfold(
        features_df, 
        n_splits=n_folds,
        use_recharge=use_recharge
    )
    
    # Train final model on all data
    final_model, final_scaler = train_final_model(
        features_df, scaler, encoder, features
    )
    
    # Save artifacts
    save_model_artifacts(final_model, final_scaler, encoder, features, results)
    
    print(f"\n{'='*70}")
    print("🎉 K-FOLD TRAINING COMPLETE!")
    print(f"{'='*70}")
    print("\n📦 Generated Files:")
    print("  1. groundwater_model.keras - Final trained model")
    print("  2. scaler.pkl, label_encoder.pkl, feature_columns.pkl")
    print("  3. kfold_results.csv - Performance metrics per fold")
    print("  4. kfold_confusion_matrix.png - Overall confusion matrix")
    print("  5. kfold_performance.png - Fold performance comparison")
    
    return final_model, final_scaler, encoder, features, results


# ============================================
# USAGE
# ============================================

if __name__ == "__main__":
    
    # For 25 districts, use 5-fold CV (5 samples per fold)
    model, scaler, encoder, features, results = main(
        csv_path='C:\\codes\\PyTorchTest2\\i2final_lstm.csv',
        n_folds=5,  # Recommended for 25 samples
        use_recharge=True
    )
    
    print("\n✅ Model ready for deployment!")
    print(f"Average accuracy: {results['accuracy'].mean()*100:.2f}% ± {results['accuracy'].std()*100:.2f}%")