"""
SIMPLE ML MODELS FOR SMALL DATASETS (23 samples)
=================================================
Classical ML algorithms that work well with limited data
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# ============================================
# COMPARE MULTIPLE ALGORITHMS
# ============================================

def compare_algorithms(df):
    """
    Test multiple ML algorithms and find the best one
    """
    print("=" * 70)
    print("COMPARING ML ALGORITHMS FOR SMALL DATASET")
    print("=" * 70)
    
    # Prepare data
    FEATURE_COLS = ['gwl_avg', 'trend_slope', 'rainfall_avg', 'recharge_avg']
    X = df[FEATURE_COLS].values
    y = df['status'].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"\n📊 Dataset: {len(df)} samples, {len(FEATURE_COLS)} features, {len(label_encoder.classes_)} classes")
    
    # Define algorithms to test
    algorithms = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=3, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42),
        'SVM (Linear)': SVC(kernel='linear', random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=3),
        'Naive Bayes': GaussianNB(),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, max_depth=2, random_state=42)
    }
    
    # 5-Fold Cross-Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = []
    
    print(f"\n🔄 Running 5-Fold Cross-Validation on {len(algorithms)} algorithms...")
    print("=" * 70)
    
    for name, model in algorithms.items():
        # Cross-validate
        scores = cross_val_score(model, X_scaled, y_encoded, cv=cv, scoring='accuracy')
        
        results.append({
            'Algorithm': name,
            'Mean Accuracy': scores.mean(),
            'Std Dev': scores.std(),
            'Min': scores.min(),
            'Max': scores.max()
        })
        
        print(f"\n{name:25s}: {scores.mean()*100:5.2f}% ± {scores.std()*100:5.2f}%")
        print(f"{'':25s}  Range: [{scores.min()*100:.2f}%, {scores.max()*100:.2f}%]")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results).sort_values('Mean Accuracy', ascending=False)
    
    print(f"\n{'='*70}")
    print("RANKING OF ALGORITHMS")
    print(f"{'='*70}")
    print(results_df.to_string(index=False))
    
    # Find best algorithm
    best_algo_name = results_df.iloc[0]['Algorithm']
    best_accuracy = results_df.iloc[0]['Mean Accuracy']
    
    print(f"\n🏆 BEST ALGORITHM: {best_algo_name}")
    print(f"   Accuracy: {best_accuracy*100:.2f}%")
    
    # Visualize comparison
    plt.figure(figsize=(12, 6))
    
    # Sort by accuracy
    results_df_sorted = results_df.sort_values('Mean Accuracy')
    
    colors = ['green' if acc > 0.6 else 'orange' if acc > 0.5 else 'red' 
              for acc in results_df_sorted['Mean Accuracy']]
    
    plt.barh(results_df_sorted['Algorithm'], results_df_sorted['Mean Accuracy']*100, 
             xerr=results_df_sorted['Std Dev']*100, capsize=5, alpha=0.7, color=colors)
    plt.xlabel('Accuracy (%)')
    plt.title('Algorithm Comparison (5-Fold CV)')
    plt.axvline(50, color='red', linestyle='--', alpha=0.5, label='Random Baseline (50%)')
    plt.axvline(60, color='orange', linestyle='--', alpha=0.5, label='Good (60%)')
    plt.axvline(70, color='green', linestyle='--', alpha=0.5, label='Excellent (70%)')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('algorithm_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: algorithm_comparison.png")
    plt.close()
    
    return results_df, algorithms, X_scaled, y_encoded, label_encoder, scaler, FEATURE_COLS


# ============================================
# TRAIN BEST MODEL WITH DETAILED EVALUATION
# ============================================

def train_best_model(df, best_algo_name, algorithms, X_scaled, y_encoded, label_encoder):
    """
    Train the best algorithm and show detailed results
    """
    print(f"\n{'='*70}")
    print(f"DETAILED EVALUATION OF BEST MODEL: {best_algo_name}")
    print(f"{'='*70}")
    
    # Get best model
    best_model = algorithms[best_algo_name]
    
    # 5-Fold CV with detailed metrics
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    all_y_true = []
    all_y_pred = []
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_scaled, y_encoded), 1):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]
        
        # Clone and train model
        from sklearn.base import clone
        model = clone(best_model)
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_val)
        
        # Store
        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)
        
        accuracy = accuracy_score(y_val, y_pred)
        fold_results.append({
            'Fold': fold,
            'Accuracy': accuracy,
            'Val_Samples': len(y_val)
        })
        
        print(f"Fold {fold}: {accuracy*100:.2f}% ({len(y_val)} samples)")
    
    # Overall metrics
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    
    print(f"\n{'='*70}")
    print("OVERALL PERFORMANCE")
    print(f"{'='*70}")
    
    # Classification report
    y_true_labels = label_encoder.inverse_transform(all_y_true)
    y_pred_labels = label_encoder.inverse_transform(all_y_pred)
    
    print("\n📋 CLASSIFICATION REPORT:")
    print(classification_report(y_true_labels, y_pred_labels, zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(all_y_true, all_y_pred)
    cm_df = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)
    
    print("\n📈 CONFUSION MATRIX:")
    print(cm_df)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title(f'Confusion Matrix - {best_algo_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('best_model_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: best_model_confusion_matrix.png")
    plt.close()
    
    # Per-class accuracy
    print(f"\n📊 PER-CLASS ACCURACY:")
    for i, class_name in enumerate(label_encoder.classes_):
        class_mask = (all_y_true == i)
        if class_mask.sum() > 0:
            class_acc = (all_y_pred[class_mask] == i).sum() / class_mask.sum()
            print(f"  {class_name:20s}: {class_acc*100:.2f}%")
    
    return best_model, fold_results


# ============================================
# TRAIN FINAL MODEL ON ALL DATA
# ============================================

def train_final_model(best_model, X_scaled, y_encoded):
    """
    Train final model on all data for deployment
    """
    print(f"\n{'='*70}")
    print("TRAINING FINAL MODEL ON ALL DATA")
    print(f"{'='*70}")
    
    from sklearn.base import clone
    final_model = clone(best_model)
    final_model.fit(X_scaled, y_encoded)
    
    print(f"✅ Final model trained on {len(X_scaled)} samples")
    
    return final_model


# ============================================
# SAVE MODEL
# ============================================

def save_best_model(model, scaler, label_encoder, feature_cols, algo_name, n_samples=None):
    """Save the best classical ML model"""
    import pickle
    
    print(f"\n{'='*70}")
    print("SAVING BEST MODEL")
    print(f"{'='*70}")
    
    # Save model
    with open('best_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("✅ Saved: best_model.pkl")
    
    with open('scaler_best.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("✅ Saved: scaler_best.pkl")
    
    with open('label_encoder_best.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    print("✅ Saved: label_encoder_best.pkl")
    
    with open('feature_columns_best.pkl', 'wb') as f:
        pickle.dump(feature_cols, f)
    print("✅ Saved: feature_columns_best.pkl")
    
    # Save metadata
    # If n_samples not provided, try to infer it from scaler if possible
    if n_samples is None:
        try:
            # scaler may have been fit on an array; try to infer from mean_ attribute length
            n_samples = getattr(scaler, 'n_samples_seen_', None)
        except Exception:
            n_samples = None

    metadata = {
        'algorithm': algo_name,
        'features': feature_cols,
        'n_samples': n_samples,
        'n_classes': len(label_encoder.classes_),
        'classes': list(label_encoder.classes_)
    }
    
    with open('model_metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    print("✅ Saved: model_metadata.pkl")


# ============================================
# MAIN FUNCTION
# ============================================

def main():
    """
    Complete pipeline: Compare algorithms, train best one
    """
    print("\n" + "=" * 70)
    print("CLASSICAL ML FOR SMALL DATASETS")
    print("Better than Neural Networks for <100 samples")
    print("=" * 70)
    
    # Load data
    df = pd.read_csv('training_dataset_rebalanced.csv')
    
    print(f"\n📂 Loaded {len(df)} districts")
    print(f"Class distribution:")
    print(df['status'].value_counts())
    
    # Compare all algorithms
    results_df, algorithms, X_scaled, y_encoded, label_encoder, scaler, features = compare_algorithms(df)
    
    # Get best algorithm
    best_algo_name = results_df.iloc[0]['Algorithm']
    best_accuracy = results_df.iloc[0]['Mean Accuracy']
    
    # Detailed evaluation of best model
    best_model, fold_results = train_best_model(
        df, best_algo_name, algorithms, X_scaled, y_encoded, label_encoder
    )
    
    # Train final model
    final_model = train_final_model(best_model, X_scaled, y_encoded)
    
    # Save
    # Pass number of samples so metadata is accurate
    save_best_model(final_model, scaler, label_encoder, features, best_algo_name, n_samples=len(X_scaled))
    
    print(f"\n{'='*70}")
    print("✅ TRAINING COMPLETE!")
    print(f"{'='*70}")
    
    print(f"\n📊 SUMMARY:")
    print(f"  Best Algorithm: {best_algo_name}")
    print(f"  Accuracy: {best_accuracy*100:.2f}%")
    
    if best_accuracy > 0.70:
        print(f"  ✅ EXCELLENT - Great performance for 23 samples!")
    elif best_accuracy > 0.60:
        print(f"  ✅ GOOD - Reasonable for small dataset")
    elif best_accuracy > 0.50:
        print(f"  ⚠️  MODERATE - Better than random (50%)")
    else:
        print(f"  ❌ POOR - Need more data or better features")
    
    print(f"\n💡 COMPARISON:")
    print(f"  Neural Network: 47% ± 37.68%")
    print(f"  Best Classical ML: {best_accuracy*100:.2f}% ± {results_df.iloc[0]['Std Dev']*100:.2f}%")
    
    improvement = best_accuracy - 0.47
    if improvement > 0.1:
        print(f"  ✅ {improvement*100:.2f}% improvement over neural network!")
    
    print(f"\n📦 Generated Files:")
    print("  1. best_model.pkl - Best classical ML model")
    print("  2. scaler_best.pkl, label_encoder_best.pkl")
    print("  3. model_metadata.pkl - Model info")
    print("  4. algorithm_comparison.png - All algorithms compared")
    print("  5. best_model_confusion_matrix.png - Best model performance")
    
    return final_model, scaler, label_encoder, features, results_df


if __name__ == "__main__":
    model, scaler, encoder, features, results = main()
    
    print("\n✅ Model ready for predictions!")
    print("Use best_model.pkl instead of the neural network")