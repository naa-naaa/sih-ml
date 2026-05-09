"""
VALIDATE YOUR 96% ACCURACY - SANITY CHECKS
==========================================
Run these checks to ensure your high accuracy is real and not a red flag
"""
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================
# CHECK 1: Examine Individual Fold Performance
# ============================================

def check_fold_consistency():
    """Are all folds performing similarly?"""
    print("=" * 70)
    print("CHECK 1: FOLD CONSISTENCY")
    print("=" * 70)
    
    results = pd.read_csv('kfold_results.csv')
    
    print("\nPer-Fold Performance:")
    print(results.to_string(index=False))
    
    # Check for outliers
    mean_acc = results['accuracy'].mean()
    std_acc = results['accuracy'].std()
    
    print(f"\nStatistics:")
    print(f"  Mean: {mean_acc*100:.2f}%")
    print(f"  Std:  ±{std_acc*100:.2f}%")
    print(f"  Min:  {results['accuracy'].min()*100:.2f}%")
    print(f"  Max:  {results['accuracy'].max()*100:.2f}%")
    
    # Flag if any fold is way off
    for idx, row in results.iterrows():
        if abs(row['accuracy'] - mean_acc) > 2 * std_acc:
            print(f"\n⚠️  WARNING: Fold {row['fold']} is an outlier!")
            print(f"   Accuracy: {row['accuracy']*100:.2f}% (expected {mean_acc*100:.2f}% ± {2*std_acc*100:.2f}%)")
    
    return results


# ============================================
# CHECK 2: Class Distribution Analysis
# ============================================

def check_class_balance():
    """Is your dataset balanced or is one class dominating?"""
    print("\n" + "=" * 70)
    print("CHECK 2: CLASS DISTRIBUTION")
    print("=" * 70)
    
    # Load training data
    df = pd.read_csv('training_dataset.csv')
    
    class_counts = df['status'].value_counts()
    class_pcts = df['status'].value_counts(normalize=True) * 100
    
    print("\nClass Distribution:")
    for cls in class_counts.index:
        print(f"  {cls:20s}: {class_counts[cls]:3d} samples ({class_pcts[cls]:5.1f}%)")
    
    # Check for severe imbalance
    max_pct = class_pcts.max()
    if max_pct > 80:
        print(f"\n⚠️  WARNING: Severe class imbalance detected!")
        print(f"   Majority class: {class_pcts.idxmax()} ({max_pct:.1f}%)")
        print(f"   A dummy classifier predicting only '{class_pcts.idxmax()}' would get {max_pct:.1f}% accuracy!")
        print(f"   Your 96% accuracy might just be predicting the majority class.")
    elif max_pct > 60:
        print(f"\n⚠️  Moderate imbalance: Majority class is {max_pct:.1f}%")
    else:
        print(f"\n✅ Classes are reasonably balanced")
    
    # Visualize
    plt.figure(figsize=(10, 6))
    class_counts.plot(kind='bar', color='steelblue', alpha=0.7)
    plt.title('Class Distribution in Training Data')
    plt.xlabel('Groundwater Status')
    plt.ylabel('Number of Districts')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
    print("\n✅ Saved: class_distribution.png")
    plt.close()
    
    return class_counts


# ============================================
# CHECK 3: Confusion Matrix Deep Dive
# ============================================

def analyze_confusion_matrix():
    """Where is the model making mistakes?"""
    print("\n" + "=" * 70)
    print("CHECK 3: CONFUSION MATRIX ANALYSIS")
    print("=" * 70)
    
    # You'll need to look at the printed confusion matrix from training
    # But we can provide interpretation guidelines
    
    print("\nWhat to look for in your confusion matrix:")
    print("\n✅ GOOD SIGNS:")
    print("  - High numbers on the diagonal (correct predictions)")
    print("  - Small numbers off-diagonal (few errors)")
    print("  - All classes have predictions (not just majority)")
    
    print("\n⚠️  RED FLAGS:")
    print("  - One row dominates (model predicts one class for everything)")
    print("  - Empty rows/columns (model never predicts certain classes)")
    print("  - Systematic confusion between two classes")
    
    print("\n📸 Check your generated file: kfold_confusion_matrix.png")


# ============================================
# CHECK 4: Feature Importance via Permutation
# ============================================

def check_feature_importance():
    """Are all features contributing or is model using shortcuts?"""
    print("\n" + "=" * 70)
    print("CHECK 4: FEATURE IMPORTANCE")
    print("=" * 70)
    
    # Load model and data
    model = load_model('groundwater_model.keras')
    
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    with open('feature_columns.pkl', 'rb') as f:
        features = pickle.load(f)
    
    # Load data
    df = pd.read_csv('training_dataset.csv')
    X = df[features].values
    y = df['status'].values
    y_encoded = encoder.transform(y)
    X_scaled = scaler.transform(X)
    
    # Baseline accuracy
    baseline_pred = model.predict(X_scaled, verbose=0)
    baseline_acc = (np.argmax(baseline_pred, axis=1) == y_encoded).mean()
    
    print(f"Baseline accuracy: {baseline_acc*100:.2f}%")
    print("\nFeature importance (drop in accuracy when feature is shuffled):")
    
    importance = {}
    for i, feature in enumerate(features):
        # Shuffle this feature
        X_shuffled = X_scaled.copy()
        np.random.shuffle(X_shuffled[:, i])
        
        # Re-evaluate
        pred = model.predict(X_shuffled, verbose=0)
        acc = (np.argmax(pred, axis=1) == y_encoded).mean()
        
        importance[feature] = baseline_acc - acc
        print(f"  {feature:20s}: {importance[feature]*100:+6.2f}% (accuracy drop)")
    
    # Visualize
    plt.figure(figsize=(10, 6))
    sorted_features = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
    features_list = [x[0] for x in sorted_features]
    values_list = [x[1]*100 for x in sorted_features]
    
    colors = ['red' if v < 0 else 'green' for v in values_list]
    plt.barh(features_list, values_list, color=colors, alpha=0.7)
    plt.xlabel('Impact on Accuracy (%)')
    plt.title('Feature Importance (Permutation Test)')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("\n✅ Saved: feature_importance.png")
    plt.close()
    
    # Interpretation
    print("\n💡 INTERPRETATION:")
    critical_features = [k for k, v in importance.items() if abs(v) > 0.1]
    if len(critical_features) == 0:
        print("⚠️  WARNING: No feature seems important!")
        print("   Model might be overfitting or using memorization")
    elif len(critical_features) == 1:
        print(f"⚠️  Model relies heavily on just '{critical_features[0]}'")
        print("   This might be a shortcut - verify this feature is predictive")
    else:
        print(f"✅ Model uses multiple features: {critical_features}")
    
    return importance


# ============================================
# CHECK 5: Test on "Unseen" Districts
# ============================================

def simulate_new_predictions():
    """Can the model generalize to similar data?"""
    print("\n" + "=" * 70)
    print("CHECK 5: GENERALIZATION TEST")
    print("=" * 70)
    
    # Load model
    model = load_model('groundwater_model.keras')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    with open('feature_columns.pkl', 'rb') as f:
        features = pickle.load(f)
    
    # Load training data
    df = pd.read_csv('training_dataset.csv')
    
    print("\nOriginal training data statistics:")
    print(df[features].describe())
    
    # Create synthetic "new" districts with realistic values
    print("\n🧪 Testing on synthetic districts...")
    
    test_cases = [
        {
            'name': 'Critical GWL, bad trend',
            'gwl_avg': 28,
            'trend_slope': -0.4,
            'rainfall_avg': 750,
            'recharge_avg': 250
        },
        {
            'name': 'Safe district',
            'gwl_avg': 12,
            'trend_slope': -0.05,
            'rainfall_avg': 1200,
            'recharge_avg': 500
        },
        {
            'name': 'Borderline case',
            'gwl_avg': 18,
            'trend_slope': -0.2,
            'rainfall_avg': 900,
            'recharge_avg': 350
        }
    ]
    
    for case in test_cases:
        name = case.pop('name')
        
        # Prepare input
        if 'recharge_avg' not in features:
            case.pop('recharge_avg', None)
        
        X_test = np.array([[case[f] for f in features]])
        X_test_scaled = scaler.transform(X_test)
        
        # Predict
        pred_proba = model.predict(X_test_scaled, verbose=0)[0]
        pred_class = encoder.inverse_transform([np.argmax(pred_proba)])[0]
        confidence = pred_proba.max()
        
        print(f"\n{name}:")
        print(f"  Input: {case}")
        print(f"  Prediction: {pred_class} (confidence: {confidence*100:.1f}%)")
        print(f"  All probabilities: {dict(zip(encoder.classes_, pred_proba))}")


# ============================================
# CHECK 6: Compare to Dummy Baselines
# ============================================

def compare_to_baselines():
    """Is your model better than naive approaches?"""
    print("\n" + "=" * 70)
    print("CHECK 6: BASELINE COMPARISON")
    print("=" * 70)
    
    df = pd.read_csv('training_dataset.csv')
    
    # Dummy classifier: Always predict majority class
    majority_class = df['status'].mode()[0]
    majority_acc = (df['status'] == majority_class).mean()
    
    print(f"\nDummy Baseline Accuracies:")
    print(f"  Always predict '{majority_class}': {majority_acc*100:.2f}%")
    
    # Random classifier
    classes = df['status'].unique()
    random_acc = (1 / len(classes))
    print(f"  Random guessing: {random_acc*100:.2f}%")
    
    # Your model
    results = pd.read_csv('kfold_results.csv')
    your_acc = results['accuracy'].mean()
    
    print(f"\nYour K-Fold Model: {your_acc*100:.2f}%")
    
    # Comparison
    print(f"\n📊 COMPARISON:")
    improvement_over_dummy = your_acc - majority_acc
    improvement_over_random = your_acc - random_acc
    
    if improvement_over_dummy < 0.05:
        print(f"⚠️  WARNING: Only {improvement_over_dummy*100:.1f}% better than dummy classifier!")
        print("   Your model might just be learning class frequencies")
    else:
        print(f"✅ Model is {improvement_over_dummy*100:.1f}% better than dummy classifier")
        print(f"✅ Model is {improvement_over_random*100:.1f}% better than random guessing")


# ============================================
# RUN ALL CHECKS
# ============================================

def run_all_checks():
    """Execute all validation checks"""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE ACCURACY VALIDATION")
    print("Running 6 sanity checks on your 96% accuracy...")
    print("=" * 70)
    
    try:
        check_fold_consistency()
    except Exception as e:
        print(f"❌ Check 1 failed: {e}")
    
    try:
        check_class_balance()
    except Exception as e:
        print(f"❌ Check 2 failed: {e}")
    
    try:
        analyze_confusion_matrix()
    except Exception as e:
        print(f"❌ Check 3 failed: {e}")
    
    try:
        check_feature_importance()
    except Exception as e:
        print(f"❌ Check 4 failed: {e}")
    
    try:
        simulate_new_predictions()
    except Exception as e:
        print(f"❌ Check 5 failed: {e}")
    
    try:
        compare_to_baselines()
    except Exception as e:
        print(f"❌ Check 6 failed: {e}")
    
    print("\n" + "=" * 70)
    print("✅ VALIDATION COMPLETE")
    print("=" * 70)
    print("\n📝 SUMMARY:")
    print("Review the checks above to determine if your 96% accuracy is:")
    print("  ✅ Real and trustworthy")
    print("  ⚠️  Inflated due to class imbalance")
    print("  ❌ Result of overfitting/data leakage")


if __name__ == "__main__":
    run_all_checks()