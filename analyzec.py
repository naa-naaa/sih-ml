"""
FIXING YOUR GROUNDWATER CLASSIFICATION
======================================
Adjust rules to match your actual data distribution
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================
# STEP 1: Analyze Your Data Distribution
# ============================================

def analyze_data_distribution(csv_path):
    """Understand what values you actually have"""
    print("=" * 70)
    print("DATA DISTRIBUTION ANALYSIS")
    print("=" * 70)
    
    df = pd.read_csv(csv_path)
    
    print("\n📊 FEATURE STATISTICS:")
    print(df[['gwl_avg', 'trend_slope', 'rainfall_avg', 'recharge_avg']].describe())
    
    # Visualize distributions
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    features = ['gwl_avg', 'trend_slope', 'rainfall_avg', 'recharge_avg']
    
    for idx, feature in enumerate(features):
        ax = axes[idx // 2, idx % 2]
        
        # Histogram with KDE
        ax.hist(df[feature], bins=15, alpha=0.6, color='steelblue', edgecolor='black')
        ax.set_xlabel(feature)
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {feature}')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add percentile lines
        percentiles = [25, 50, 75]
        for p in percentiles:
            val = df[feature].quantile(p/100)
            ax.axvline(val, color='red', linestyle='--', alpha=0.5, linewidth=1)
            ax.text(val, ax.get_ylim()[1] * 0.95, f'P{p}', 
                   rotation=90, verticalalignment='top', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
    print("\n✅ Saved: feature_distributions.png")
    plt.close()
    
    return df


# ============================================
# STEP 2: Data-Driven Classification Rules
# ============================================

def create_data_driven_rules(df):
    """Create classification rules based on YOUR data quartiles"""
    print("\n" + "=" * 70)
    print("CREATING DATA-DRIVEN CLASSIFICATION RULES")
    print("=" * 70)
    
    # Calculate percentiles for each feature
    gwl_q25 = df['gwl_avg'].quantile(0.25)
    gwl_q50 = df['gwl_avg'].quantile(0.50)
    gwl_q75 = df['gwl_avg'].quantile(0.75)
    
    trend_q25 = df['trend_slope'].quantile(0.25)
    trend_q50 = df['trend_slope'].quantile(0.50)
    trend_q75 = df['trend_slope'].quantile(0.75)
    
    rain_q25 = df['rainfall_avg'].quantile(0.25)
    rain_q50 = df['rainfall_avg'].quantile(0.50)
    
    rech_q25 = df['recharge_avg'].quantile(0.25)
    rech_q50 = df['recharge_avg'].quantile(0.50)
    
    print("\n📏 CALCULATED THRESHOLDS:")
    print(f"\nGWL (depth to water):")
    print(f"  25th percentile: {gwl_q25:.2f}")
    print(f"  50th percentile: {gwl_q50:.2f}")
    print(f"  75th percentile: {gwl_q75:.2f}")
    print(f"  Note: Higher = deeper (worse)")
    
    print(f"\nTrend Slope (annual change):")
    print(f"  25th percentile: {trend_q25:.2f}")
    print(f"  50th percentile: {trend_q50:.2f}")
    print(f"  75th percentile: {trend_q75:.2f}")
    print(f"  Note: More negative = declining faster (worse)")
    
    print(f"\nRainfall:")
    print(f"  25th percentile: {rain_q25:.2f}")
    print(f"  50th percentile: {rain_q50:.2f}")
    
    print(f"\nRecharge:")
    print(f"  25th percentile: {rech_q25:.2f}")
    print(f"  50th percentile: {rech_q50:.2f}")
    
    # Define classification function
    def classify_data_driven(row):
        """
        Multi-criteria classification based on percentiles
        
        Over-Exploited: Worst 10-15% (multiple bad indicators)
        Critical: Bad 15-25% (2+ bad indicators)
        Semi-Critical: Moderate 25-40% (1-2 warning signs)
        Safe: Best 40%+ (mostly good indicators)
        """
        gwl = row['gwl_avg']
        trend = row['trend_slope']
        rain = row['rainfall_avg']
        rech = row['recharge_avg']
        
        # Count risk factors
        risk_score = 0
        
        # GWL risk (deeper = worse, but your data has negative values!)
        if gwl > gwl_q75:
            risk_score += 2  # High risk
        elif gwl > gwl_q50:
            risk_score += 1  # Medium risk
        
        # Trend risk (more negative = worse)
        if trend < trend_q25:
            risk_score += 2  # Rapidly declining
        elif trend < trend_q50:
            risk_score += 1  # Moderately declining
        
        # Rainfall risk (lower = worse)
        if rain < rain_q25:
            risk_score += 1
        
        # Recharge risk (lower = worse)
        if rech < rech_q25:
            risk_score += 1
        
        # Classify based on total risk score
        if risk_score >= 5:
            return 'Over-Exploited'
        elif risk_score >= 3:
            return 'Critical'
        elif risk_score >= 2:
            return 'Semi-Critical'
        else:
            return 'Safe'
    
    return classify_data_driven


# ============================================
# STEP 3: Apply and Evaluate New Rules
# ============================================

def apply_new_classification(df):
    """Apply data-driven rules and check distribution"""
    print("\n" + "=" * 70)
    print("APPLYING NEW CLASSIFICATION RULES")
    print("=" * 70)
    
    classify_func = create_data_driven_rules(df)
    
    # Apply classification
    df['status_old'] = df['status']  # Save old labels
    df['status'] = df.apply(classify_func, axis=1)
    
    print("\n📊 NEW CLASS DISTRIBUTION:")
    new_counts = df['status'].value_counts()
    new_pcts = df['status'].value_counts(normalize=True) * 100
    
    for cls in ['Over-Exploited', 'Critical', 'Semi-Critical', 'Safe']:
        if cls in new_counts.index:
            print(f"  {cls:20s}: {new_counts[cls]:3d} samples ({new_pcts[cls]:5.1f}%)")
    
    # Compare old vs new
    print("\n📊 COMPARISON (Old → New):")
    comparison = pd.crosstab(df['status_old'], df['status'])
    print(comparison)
    
    # Check balance
    max_pct = new_pcts.max()
    min_pct = new_pcts.min()
    
    print(f"\n📈 BALANCE CHECK:")
    print(f"  Largest class: {max_pct:.1f}%")
    print(f"  Smallest class: {min_pct:.1f}%")
    print(f"  Ratio: {max_pct/min_pct:.2f}:1")
    
    if max_pct > 70:
        print(f"  ⚠️  Still imbalanced! Consider adjusting thresholds.")
    elif max_pct < 40:
        print(f"  ✅ Good balance!")
    else:
        print(f"  ⚠️  Moderate balance - acceptable for ML")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Old distribution
    df['status_old'].value_counts().plot(kind='bar', ax=axes[0], color='lightcoral', alpha=0.7)
    axes[0].set_title('Old Classification (Rule-Based)')
    axes[0].set_xlabel('Status')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # New distribution
    df['status'].value_counts().plot(kind='bar', ax=axes[1], color='steelblue', alpha=0.7)
    axes[1].set_title('New Classification (Data-Driven)')
    axes[1].set_xlabel('Status')
    axes[1].set_ylabel('Count')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('classification_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✅ Saved: classification_comparison.png")
    plt.close()
    
    return df


# ============================================
# STEP 4: Alternative - Use Clustering
# ============================================

def use_clustering_for_labels(df):
    """Use K-Means to discover natural groupings"""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    print("\n" + "=" * 70)
    print("ALTERNATIVE: CLUSTERING-BASED CLASSIFICATION")
    print("=" * 70)
    
    # Prepare features
    features = ['gwl_avg', 'trend_slope', 'rainfall_avg', 'recharge_avg']
    X = df[features].values
    
    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Try different numbers of clusters
    print("\n🔍 Testing different cluster counts...")
    
    inertias = []
    for k in range(2, 6):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        print(f"  K={k}: Inertia={kmeans.inertia_:.2f}")
    
    # Use 4 clusters (matching our 4 classes)
    print(f"\n✅ Using K=4 clusters")
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Analyze clusters
    print(f"\n📊 CLUSTER STATISTICS:")
    for cluster_id in range(4):
        cluster_data = df[df['cluster'] == cluster_id]
        print(f"\nCluster {cluster_id} ({len(cluster_data)} districts):")
        print(f"  Avg GWL:      {cluster_data['gwl_avg'].mean():8.2f}")
        print(f"  Avg Trend:    {cluster_data['trend_slope'].mean():8.2f}")
        print(f"  Avg Rainfall: {cluster_data['rainfall_avg'].mean():8.2f}")
        print(f"  Avg Recharge: {cluster_data['recharge_avg'].mean():8.2f}")
    
    # Map clusters to meaningful names based on characteristics
    print(f"\n🏷️  SUGGESTED CLUSTER MAPPING:")
    print("(Review cluster stats and manually assign labels)")
    print("Example:")
    print("  Cluster 0 → Safe (best indicators)")
    print("  Cluster 1 → Semi-Critical")
    print("  Cluster 2 → Critical")
    print("  Cluster 3 → Over-Exploited (worst indicators)")
    
    # Visualize clusters
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    scatter_pairs = [
        ('gwl_avg', 'trend_slope'),
        ('gwl_avg', 'rainfall_avg'),
        ('trend_slope', 'rainfall_avg'),
        ('rainfall_avg', 'recharge_avg')
    ]
    
    for idx, (feat1, feat2) in enumerate(scatter_pairs):
        ax = axes[idx // 2, idx % 2]
        
        for cluster_id in range(4):
            cluster_data = df[df['cluster'] == cluster_id]
            ax.scatter(
                cluster_data[feat1], 
                cluster_data[feat2],
                label=f'Cluster {cluster_id}',
                alpha=0.6,
                s=100
            )
        
        ax.set_xlabel(feat1)
        ax.set_ylabel(feat2)
        ax.set_title(f'{feat1} vs {feat2}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('clustering_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✅ Saved: clustering_analysis.png")
    plt.close()
    
    return df


# ============================================
# MAIN WORKFLOW
# ============================================

def main():
    """Complete workflow to fix classification"""
    print("\n" + "=" * 70)
    print("FIXING GROUNDWATER CLASSIFICATION")
    print("=" * 70)
    
    # Load your training dataset
    df = pd.read_csv('training_dataset.csv')
    
    # Step 1: Analyze distributions
    df = analyze_data_distribution('training_dataset.csv')
    
    # Step 2: Apply data-driven rules
    df = apply_new_classification(df)
    
    # Save new labels
    df.to_csv('training_dataset_rebalanced.csv', index=False)
    print(f"\n💾 Saved: training_dataset_rebalanced.csv")
    
    # Step 3: Optional - try clustering
    print("\n" + "=" * 70)
    df_clustered = use_clustering_for_labels(df)
    df_clustered.to_csv('training_dataset_clustered.csv', index=False)
    print(f"💾 Saved: training_dataset_clustered.csv")
    
    print("\n" + "=" * 70)
    print("✅ CLASSIFICATION FIXED!")
    print("=" * 70)
    print("\n📋 NEXT STEPS:")
    print("1. Review feature_distributions.png - understand your data")
    print("2. Check classification_comparison.png - see old vs new")
    print("3. Review clustering_analysis.png - see natural groupings")
    print("4. Choose best approach:")
    print("   Option A: Use training_dataset_rebalanced.csv (data-driven rules)")
    print("   Option B: Use training_dataset_clustered.csv (clustering)")
    print("5. Retrain model with new balanced labels")
    print("\n⚠️  IMPORTANT: After rebalancing, your accuracy will likely DROP to 60-80%")
    print("   But this will be REAL accuracy, not inflated by class imbalance!")


if __name__ == "__main__":
    main()