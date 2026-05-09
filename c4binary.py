"""
BINARY CLASSIFICATION: SAFE vs AT-RISK
=======================================
Simplify to 2 classes for better accuracy with small data
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

def create_binary_labels(df):
    """
    Merge 3 classes into 2:
    - Safe → Safe
    - Semi-Critical + Critical → At-Risk
    """
    print("=" * 70)
    print("CREATING BINARY CLASSIFICATION")
    print("=" * 70)
    
    print("\nOriginal classes:")
    print(df['status'].value_counts())
    
    # Create binary labels
    def to_binary(status):
        if status == 'Safe':
            return 'Safe'
        else:  # Semi-Critical or Critical
            return 'At-Risk'
    
    df['status_binary'] = df['status'].apply(to_binary)
    
    print("\nNew binary classes:")
    print(df['status_binary'].value_counts())
    
    # Check balance
    counts = df['status_binary'].value_counts()
    ratio = counts.max() / counts.min()
    
    print(f"\nBalance ratio: {ratio:.2f}:1")
    if ratio < 2:
        print("✅ Well balanced!")
    elif ratio < 3:
        print("⚠️  Moderate imbalance")
    else:
        print("❌ Still imbalanced")
    
    return df


def train_binary_classifier(df):
    """Train binary classifier"""
    print("\n" + "=" * 70)
    print("TRAINING BINARY CLASSIFIER")
    print("=" * 70)
    
    # Prepare data
    FEATURE_COLS = ['gwl_avg', 'trend_slope', 'rainfall_avg', 'recharge_avg']
    X = df[FEATURE_COLS].values
    y = df['status_binary'].values
    
    # Encode
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"\nDataset: {len(df)} samples, 2 classes")
    print(f"Classes: {label_encoder.classes_}")
    
    # Try multiple algorithms
    algorithms = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {}
    
    for name, model in algorithms.items():
        scores = cross_val_score(model, X_scaled, y_encoded, cv=cv, scoring='accuracy')
        results[name] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores
        }
        print(f"\n{name}:")
        print(f"  Accuracy: {scores.mean()*100:.2f}% ± {scores.std()*100:.2f}%")
        print(f"  Folds: {[f'{s*100:.1f}%' for s in scores]}")
    
    # Choose best
    best_name = max(results, key=lambda k: results[k]['mean'])
    best_accuracy = results[best_name]['mean']
    
    print(f"\n🏆 Best: {best_name} ({best_accuracy*100:.2f}%)")
    
    # Train final model
    best_model = algorithms[best_name]
    best_model.fit(X_scaled, y_encoded)
    
    # Detailed evaluation
    all_y_true = []
    all_y_pred = []
    
    for train_idx, val_idx in cv.split(X_scaled, y_encoded):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]
        
        from sklearn.base import clone
        model = clone(best_model)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)
    
    # Report
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)
    
    y_true_labels = label_encoder.inverse_transform(all_y_true)
    y_pred_labels = label_encoder.inverse_transform(all_y_pred)
    
    print(classification_report(y_true_labels, y_pred_labels, zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(all_y_true, all_y_pred)
    cm_df = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)
    
    print("\nConfusion Matrix:")
    print(cm_df)
    
    # Save
    with open('binary_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    with open('binary_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('binary_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print("\n✅ Saved: binary_model.pkl, binary_scaler.pkl, binary_encoder.pkl")
    
    return best_model, scaler, label_encoder, best_accuracy


def main():
    """Run binary classification"""
    df = pd.read_csv('training_dataset_rebalanced.csv')
    
    df = create_binary_labels(df)
    df.to_csv('training_dataset_binary.csv', index=False)
    
    model, scaler, encoder, accuracy = train_binary_classifier(df)
    
    print(f"\n{'='*70}")
    print("BINARY CLASSIFICATION COMPLETE")
    print(f"{'='*70}")
    print(f"\n📊 Results:")
    print(f"  3-Class Neural Network: 47% ± 37.68%")
    print(f"  2-Class ML Model: {accuracy*100:.2f}%")
    
    if accuracy > 0.75:
        print(f"  ✅ EXCELLENT improvement!")
    elif accuracy > 0.65:
        print(f"  ✅ GOOD improvement!")
    
    return model, scaler, encoder


if __name__ == "__main__":
    main()