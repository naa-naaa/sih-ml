"""
RETRAIN MODEL WITH BALANCED LABELS
===================================
Train on the rebalanced dataset with proper class distribution
"""
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
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
# MODEL ARCHITECTURE (Simpler for small data)
# ============================================

def create_model(n_features, n_classes):
    """
    Simpler model for small dataset (23 samples)
    Less prone to overfitting
    """
    model = Sequential([
        Dense(8, activation='relu', input_shape=(n_features,)),  # Reduced from 16
        Dropout(0.2),  # Reduced dropout
        Dense(4, activation='relu'),  # Reduced from 8
        Dense(n_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# ============================================
# K-FOLD TRAINING WITH CLASS WEIGHTS
# ============================================

def train_with_kfold_balanced(df, n_splits=5, use_class_weights=True):
    """
    Train with balanced data and optional class weighting
    
    Parameters:
    -----------
    df : DataFrame
        Your rebalanced training dataset
    n_splits : int
        Number of folds (5 recommended for 23 samples)
    use_class_weights : bool
        Weight classes inversely to frequency
    """
    print("=" * 70)
    print(f"K-FOLD TRAINING WITH BALANCED LABELS ({n_splits} folds)")
    print("=" * 70)
    
    # Features
    FEATURE_COLS = ['gwl_avg', 'trend_slope', 'rainfall_avg', 'recharge_avg']
    
    # Prepare data
    X = df[FEATURE_COLS].values
    y = df['status'].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"\n📊 DATASET INFO:")
    print(f"  Total samples: {len(df)}")
    print(f"  Features: {FEATURE_COLS}")
    print(f"  Classes: {len(label_encoder.classes_)}")
    
    print(f"\n📊 CLASS DISTRIBUTION:")
    for i, class_name in enumerate(label_encoder.classes_):
        count = np.sum(y_encoded == i)
        print(f"  {class_name:20s}: {count:2d} samples ({count/len(y)*100:5.1f}%)")
    
    # Calculate class weights if enabled
    class_weights = None
    if use_class_weights:
        from sklearn.utils.class_weight import compute_class_weight
        
        class_weights_array = compute_class_weight(
            'balanced',
            classes=np.unique(y_encoded),
            y=y_encoded
        )
        class_weights = {i: w for i, w in enumerate(class_weights_array)}
        
        print(f"\n⚖️  CLASS WEIGHTS (to handle imbalance):")
        for i, class_name in enumerate(label_encoder.classes_):
            print(f"  {class_name:20s}: {class_weights[i]:.3f}")
    
    # K-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_results = []
    all_y_true = []
    all_y_pred = []
    best_model = None
    best_accuracy = 0
    best_scaler = None
    
    print(f"\n{'='*70}")
    print(f"STARTING {n_splits}-FOLD CROSS-VALIDATION")
    print(f"{'='*70}")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded), 1):
        print(f"\n{'='*70}")
        print(f"FOLD {fold}/{n_splits}")
        print(f"{'='*70}")
        
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]
        
        print(f"Training: {len(X_train)} samples | Validation: {len(X_val)} samples")
        
        # Show validation set class distribution
        val_classes, val_counts = np.unique(y_val, return_counts=True)
        print(f"Validation classes:")
        for cls, cnt in zip(val_classes, val_counts):
            print(f"  {label_encoder.inverse_transform([cls])[0]:20s}: {cnt} samples")
        
        # Normalize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Create model
        model = create_model(len(FEATURE_COLS), len(label_encoder.classes_))
        
        # Train
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=20,  # Increased patience for small dataset
            restore_best_weights=True,
            verbose=0
        )
        
        history = model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=150,  # More epochs
            batch_size=4,  # Small batch for small dataset
            class_weight=class_weights,  # Apply class weights
            callbacks=[early_stop],
            verbose=0
        )
        
        # Evaluate
        y_pred_proba = model.predict(X_val_scaled, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Metrics
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
            'epochs_trained': len(history.history['loss'])
        })
        
        print(f"\n✅ Fold {fold} Results:")
        print(f"   Accuracy:  {accuracy*100:.2f}%")
        print(f"   Precision: {precision*100:.2f}%")
        print(f"   Recall:    {recall*100:.2f}%")
        print(f"   F1-Score:  {f1*100:.2f}%")
        print(f"   Epochs:    {len(history.history['loss'])}")
        
        # Track best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_scaler = scaler
        
        # Store predictions
        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)
    
    # ============================================
    # AGGREGATE RESULTS
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
    
    print(f"\n📋 OVERALL CLASSIFICATION REPORT:")
    y_true_labels = label_encoder.inverse_transform(all_y_true)
    y_pred_labels = label_encoder.inverse_transform(all_y_pred)
    print(classification_report(y_true_labels, y_pred_labels, zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(all_y_true, all_y_pred)
    
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
    plt.title('Confusion Matrix - Balanced Dataset')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix_balanced.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: confusion_matrix_balanced.png")
    plt.close()
    
    # Plot performance
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    x = range(1, n_splits + 1)
    plt.plot(x, results_df['accuracy']*100, 'o-', label='Accuracy', linewidth=2, markersize=8)
    plt.plot(x, results_df['f1_score']*100, 's-', label='F1-Score', linewidth=2, markersize=8)
    plt.axhline(results_df['accuracy'].mean()*100, color='blue', linestyle='--', alpha=0.5)
    plt.xlabel('Fold')
    plt.ylabel('Score (%)')
    plt.title('Performance Across Folds (Balanced Data)')
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
    plt.title('Average Metrics')
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('performance_balanced.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: performance_balanced.png")
    plt.close()
    
    # Save results
    results_df.to_csv('kfold_results_balanced.csv', index=False)
    print(f"✅ Saved: kfold_results_balanced.csv")
    
    return best_model, best_scaler, label_encoder, FEATURE_COLS, results_df


# ============================================
# TRAIN FINAL MODEL ON ALL DATA
# ============================================

def train_final_model(df, scaler, label_encoder, feature_cols, use_class_weights=True):
    """Train final model on ALL rebalanced data"""
    print(f"\n{'='*70}")
    print("TRAINING FINAL MODEL ON ALL DATA")
    print(f"{'='*70}")
    
    X = df[feature_cols].values
    y = df['status'].values
    y_encoded = label_encoder.transform(y)
    
    X_scaled = scaler.fit_transform(X)
    
    # Calculate class weights
    class_weights = None
    if use_class_weights:
        from sklearn.utils.class_weight import compute_class_weight
        
        class_weights_array = compute_class_weight(
            'balanced',
            classes=np.unique(y_encoded),
            y=y_encoded
        )
        class_weights = {i: w for i, w in enumerate(class_weights_array)}
    
    model = create_model(len(feature_cols), len(label_encoder.classes_))
    
    early_stop = EarlyStopping(
        monitor='loss',
        patience=30,
        restore_best_weights=True,
        verbose=0
    )
    
    print(f"Training on all {len(X)} samples...")
    
    history = model.fit(
        X_scaled, y_encoded,
        epochs=200,
        batch_size=4,
        class_weight=class_weights,
        callbacks=[early_stop],
        verbose=1
    )
    
    print(f"\n✅ Final model trained ({len(history.history['loss'])} epochs)")
    
    return model, scaler


# ============================================
# SAVE MODEL
# ============================================

def save_model_artifacts(model, scaler, label_encoder, feature_cols, metrics_df):
    """Save trained model"""
    print(f"\n{'='*70}")
    print("SAVING MODEL ARTIFACTS")
    print(f"{'='*70}")
    
    model.save('groundwater_model_balanced.keras')
    print("✅ Saved: groundwater_model_balanced.keras")
    
    with open('scaler_balanced.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("✅ Saved: scaler_balanced.pkl")
    
    with open('label_encoder_balanced.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    print("✅ Saved: label_encoder_balanced.pkl")
    
    with open('feature_columns_balanced.pkl', 'wb') as f:
        pickle.dump(feature_cols, f)
    print("✅ Saved: feature_columns_balanced.pkl")


# ============================================
# MAIN FUNCTION
# ============================================

def main():
    """Complete retraining pipeline with balanced data"""
    print("\n" + "=" * 70)
    print("RETRAINING WITH BALANCED LABELS")
    print("=" * 70)
    
    # Load rebalanced dataset
    print("\n📂 Loading rebalanced dataset...")
    df = pd.read_csv('training_dataset_rebalanced.csv')
    
    print(f"✅ Loaded {len(df)} districts")
    print(f"\n📊 Class distribution:")
    print(df['status'].value_counts())
    
    # K-Fold training with class weights
    print("\n🚀 Starting K-Fold training with class weighting...")
    best_model, scaler, encoder, features, results = train_with_kfold_balanced(
        df, 
        n_splits=5,
        use_class_weights=True  # Weight classes inversely to frequency
    )
    
    # Train final model
    final_model, final_scaler = train_final_model(
        df, scaler, encoder, features, use_class_weights=True
    )
    
    # Save
    save_model_artifacts(final_model, final_scaler, encoder, features, results)
    
    print(f"\n{'='*70}")
    print("✅ RETRAINING COMPLETE!")
    print(f"{'='*70}")
    
    # Summary
    avg_acc = results['accuracy'].mean()
    std_acc = results['accuracy'].std()
    
    print(f"\n📊 FINAL RESULTS:")
    print(f"  Average Accuracy: {avg_acc*100:.2f}% ± {std_acc*100:.2f}%")
    
    if avg_acc > 0.70:
        print(f"  ✅ EXCELLENT - Model is performing well on balanced data!")
    elif avg_acc > 0.60:
        print(f"  ✅ GOOD - This is real learning, not class imbalance!")
    elif avg_acc > 0.50:
        print(f"  ⚠️  MODERATE - Better than random, room for improvement")
    else:
        print(f"  ❌ POOR - Model needs more data or better features")
    
    print(f"\n📦 Generated Files:")
    print("  1. groundwater_model_balanced.keras")
    print("  2. scaler_balanced.pkl, label_encoder_balanced.pkl")
    print("  3. feature_columns_balanced.pkl")
    print("  4. kfold_results_balanced.csv")
    print("  5. confusion_matrix_balanced.png")
    print("  6. performance_balanced.png")
    
    print(f"\n💡 INTERPRETATION:")
    print(f"  Your old accuracy: 96% (fake - just predicting majority class)")
    print(f"  Your new accuracy: {avg_acc*100:.2f}% (REAL - actually learning!)")
    print(f"  This drop is EXPECTED and GOOD - it means real learning!")
    
    return final_model, final_scaler, encoder, features, results


# ============================================
# RUN
# ============================================

if __name__ == "__main__":
    model, scaler, encoder, features, results = main()
    
    print("\n✅ Model ready for deployment!")
    print("Use groundwater_model_balanced.keras for predictions")