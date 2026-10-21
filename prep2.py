# Preprocesses and engineers lag and meteorological features from raw groundwater observation tables.
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

print("="*70)
print("  GWL + RAINFALL LSTM DATA PREPARATION")
print("  Simplified Two-Feature Model")
print("="*70)

# ============================================
# STEP 1: LOAD AND STANDARDIZE
# ============================================

def load_dataset(file_path, date_col, value_col, state_col, district_col, dataset_name):
    """Load CSV and standardize columns"""
    print(f"\n📂 Loading {dataset_name.upper()} data...")
    print("-"*70)
    
    try:
        import os
        if not os.path.exists(file_path):
            print(f"   ❌ File not found: {file_path}")
            return None
        
        df = pd.read_csv(file_path)
        print(f"   ✅ Loaded {len(df):,} rows")
        
        missing_cols = []
        if date_col not in df.columns:
            missing_cols.append(f"Date: '{date_col}'")
        if value_col not in df.columns:
            missing_cols.append(f"Value: '{value_col}'")
        if state_col not in df.columns:
            missing_cols.append(f"State: '{state_col}'")
        if district_col not in df.columns:
            missing_cols.append(f"District: '{district_col}'")
        
        if missing_cols:
            print(f"   ❌ Missing columns: {', '.join(missing_cols)}")
            return None
        
        df = df.rename(columns={
            date_col: 'datetime',
            value_col: 'value',
            state_col: 'state',
            district_col: 'district'
        })
        
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df = df.dropna(subset=['datetime'])
        
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df.dropna(subset=['value'])
        
        df['date'] = df['datetime'].dt.date
        df['date'] = pd.to_datetime(df['date'])
        
        df = df[['state', 'district', 'date', 'value']]
        
        print(f"   📊 Cleaned: {len(df):,} rows")
        print(f"   📍 States: {df['state'].nunique()} | Districts: {df['district'].nunique()}")
        print(f"   📅 Range: {df['date'].min()} to {df['date'].max()}")
        
        return df
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return None


# ============================================
# STEP 2: AGGREGATE TO DAILY
# ============================================

def aggregate_to_daily(df, dataset_name):
    """Aggregate to one value per (state, district, date)"""
    print(f"\n🔄 Aggregating {dataset_name.upper()} to daily...")
    print("-"*70)
    
    dup_check = df.groupby(['state', 'district', 'date']).size()
    max_readings = dup_check.max()
    print(f"   📊 Max readings per day: {max_readings}")
    
    if dataset_name == 'rainfall':
        df_daily = df.groupby(['state', 'district', 'date']).agg({
            'value': 'sum'
        }).reset_index()
        print(f"   ✅ Aggregation: SUM (total daily rainfall)")
    else:
        df_daily = df.groupby(['state', 'district', 'date']).agg({
            'value': 'mean'
        }).reset_index()
        print(f"   ✅ Aggregation: MEAN (average daily value)")
    
    df_daily = df_daily.rename(columns={'value': dataset_name})
    print(f"   📦 Result: {len(df_daily):,} rows")
    
    return df_daily


# ============================================
# STEP 3: MERGE WITH NEAREST DATE MATCHING
# ============================================

def merge_gwl_rainfall(gwl_df, rainfall_df, max_days_diff=7):
    """Merge GWL and Rainfall using nearest date matching"""
    print(f"\n🔗 Merging GWL with RAINFALL...")
    print("-"*70)
    
    merged_rows = []
    districts = gwl_df['district'].unique()
    
    for i, district in enumerate(districts, 1):
        if i % 10 == 0 or i == 1:
            print(f"   Processing district {i}/{len(districts)}...")
        
        gwl_dist = gwl_df[gwl_df['district'] == district].copy()
        rain_dist = rainfall_df[rainfall_df['district'] == district].copy()
        
        if len(rain_dist) == 0:
            gwl_dist['rainfall'] = np.nan
            merged_rows.append(gwl_dist)
            continue
        
        # Convert to numpy datetime64
        gwl_dates_pd = pd.to_datetime(gwl_dist['date'])
        rain_dates_pd = pd.to_datetime(rain_dist['date'])
        
        gwl_dates = gwl_dates_pd.values.astype('datetime64[D]')
        rain_dates = rain_dates_pd.values.astype('datetime64[D]')
        rain_values = rain_dist['rainfall'].values
        
        matched_values = np.full(len(gwl_dates), np.nan)
        
        for idx in range(len(gwl_dates)):
            gwl_date = gwl_dates[idx]
            time_diffs = np.abs((rain_dates - gwl_date) / np.timedelta64(1, 'D')).astype(int)
            
            min_idx = np.argmin(time_diffs)
            min_diff = time_diffs[min_idx]
            
            if min_diff <= max_days_diff:
                matched_values[idx] = rain_values[min_idx]
        
        gwl_dist['rainfall'] = matched_values
        merged_rows.append(gwl_dist)
    
    merged_df = pd.concat(merged_rows, ignore_index=True)
    merged_df = merged_df.sort_values(['state', 'district', 'date']).reset_index(drop=True)
    
    total = len(merged_df)
    matched = merged_df['rainfall'].notna().sum()
    rate = (matched / total * 100) if total > 0 else 0
    
    print(f"   ✅ Complete!")
    print(f"      Total rows: {total:,}")
    print(f"      Matched: {matched:,} ({rate:.1f}%)")
    print(f"      Missing: {total - matched:,} ({100-rate:.1f}%)")
    
    return merged_df


# ============================================
# STEP 4: FILL MISSING VALUES
# ============================================

def fill_missing_values(df, method='interpolate', max_gap=7):
    """Fill missing rainfall values per district"""
    print(f"\n🔧 Filling missing rainfall ({method}, max gap: {max_gap} days)...")
    print("-"*70)
    
    filled_df = []
    
    for district in df['district'].unique():
        dist_df = df[df['district'] == district].copy()
        dist_df = dist_df.set_index('date').sort_index()
        
        if method == 'interpolate':
            dist_df['rainfall'] = dist_df['rainfall'].interpolate(
                method='linear', limit=max_gap, limit_direction='both'
            )
        elif method == 'forward_fill':
            dist_df['rainfall'] = dist_df['rainfall'].fillna(method='ffill', limit=max_gap)
        elif method == 'seasonal_mean':
            dist_df['dow'] = dist_df.index.dayofweek
            for dow in range(7):
                mask = (dist_df['dow'] == dow) & dist_df['rainfall'].isna()
                mean_val = dist_df[dist_df['dow'] == dow]['rainfall'].mean()
                if not np.isnan(mean_val):
                    dist_df.loc[mask, 'rainfall'] = mean_val
            dist_df = dist_df.drop('dow', axis=1)
        
        dist_df = dist_df.reset_index()
        filled_df.append(dist_df)
    
    result = pd.concat(filled_df, ignore_index=True)
    
    missing_gwl = result['gwl'].isna().sum()
    missing_rain = result['rainfall'].isna().sum()
    total = len(result)
    
    print(f"   ✅ Complete!")
    print(f"      GWL missing: {missing_gwl} ({missing_gwl/total*100:.1f}%)")
    print(f"      Rainfall missing: {missing_rain} ({missing_rain/total*100:.1f}%)")
    
    return result


# ============================================
# STEP 5: CREATE FEATURES
# ============================================

def create_features(df):
    """Add derived features per district"""
    print(f"\n🛠️  Creating derived features...")
    print("-"*70)
    
    feature_df = []
    
    for district in df['district'].unique():
        dist_df = df[df['district'] == district].copy()
        dist_df = dist_df.sort_values('date')
        
        # Rainfall features
        dist_df['rainfall_3day_sum'] = dist_df['rainfall'].rolling(window=3, min_periods=1).sum()
        dist_df['rainfall_7day_ma'] = dist_df['rainfall'].rolling(window=7, min_periods=1).mean()
        dist_df['rainfall_30day_sum'] = dist_df['rainfall'].rolling(window=30, min_periods=1).sum()
        
        # GWL features
        dist_df['gwl_change'] = dist_df['gwl'].diff()
        dist_df['gwl_7day_ma'] = dist_df['gwl'].rolling(window=7, min_periods=1).mean()
        dist_df['gwl_30day_ma'] = dist_df['gwl'].rolling(window=30, min_periods=1).mean()
        
        # Temporal features (seasonal patterns)
        dist_df['day_of_year'] = pd.to_datetime(dist_df['date']).dt.dayofyear
        dist_df['month'] = pd.to_datetime(dist_df['date']).dt.month
        dist_df['year'] = pd.to_datetime(dist_df['date']).dt.year
        dist_df['is_monsoon'] = dist_df['month'].isin([6, 7, 8, 9]).astype(int)  # June-Sept
        
        feature_df.append(dist_df)
    
    result = pd.concat(feature_df, ignore_index=True)
    result = result.dropna()
    
    print(f"   ✅ Complete!")
    print(f"   📊 Final features: {list(result.columns)}")
    print(f"   📦 Final rows: {len(result):,}")
    
    return result


# ============================================
# STEP 6: SAVE
# ============================================

def save_processed_data(df, output_file):
    """Save processed data"""
    print(f"\n💾 Saving: {output_file}")
    print("-"*70)
    
    df.to_csv(output_file, index=False)
    print(f"   ✅ Saved! {len(df):,} rows × {len(df.columns)} columns")
    
    print(f"\n   📊 Per-District Summary:")
    summary = df.groupby('district').agg({
        'date': ['min', 'max', 'count']
    })
    summary.columns = ['start_date', 'end_date', 'num_days']
    summary = summary.sort_values('num_days', ascending=False)
    
    print(summary.head(15).to_string())
    if len(summary) > 15:
        print(f"   ... ({len(summary)-15} more districts)")
    
    print(f"\n   📊 Overall Statistics:")
    print(df[['gwl', 'rainfall']].describe().to_string())
    
    return output_file


# ============================================
# MAIN
# ============================================

def main():
    print("\n" + "🌊"*35)
    print("   GWL + RAINFALL ONLY - DATA PREP")
    print("🌊"*35 + "\n")
    
    # Get inputs
    print("📁 File paths:")
    gwl_file = input("1. GWL CSV: ").strip()
    rainfall_file = input("2. Rainfall CSV: ").strip()
    
    print("\n🔧 GWL Column mappings:")
    gwl_date = input("  Date column: ").strip()
    gwl_value = input("  Value column: ").strip()
    gwl_state = input("  State column: ").strip()
    gwl_district = input("  District column: ").strip()
    
    print("\n🔧 Rainfall Column mappings:")
    rain_date = input("  Date column: ").strip()
    rain_value = input("  Value column: ").strip()
    rain_state = input("  State column: ").strip()
    rain_district = input("  District column: ").strip()
    
    # Load
    print("\n" + "="*70)
    print("LOADING...")
    print("="*70)
    
    gwl_df = load_dataset(gwl_file, gwl_date, gwl_value, gwl_state, gwl_district, 'gwl')
    rain_df = load_dataset(rainfall_file, rain_date, rain_value, rain_state, rain_district, 'rainfall')
    
    if gwl_df is None or rain_df is None:
        print("\n❌ Failed to load datasets")
        return
    
    print("\n✅ Both datasets loaded!")
    
    # Aggregate
    gwl_daily = aggregate_to_daily(gwl_df, 'gwl')
    rain_daily = aggregate_to_daily(rain_df, 'rainfall')
    
    # Merge
    max_days = int(input("\nMax days for date matching (7 recommended, 14 for more coverage): ").strip() or "7")
    merged_df = merge_gwl_rainfall(gwl_daily, rain_daily, max_days)
    
    if len(merged_df) == 0:
        print("\n❌ No data after merge")
        return
    
    # Fill
    print("\n⚙️  Fill missing rainfall values:")
    fill = input("Method? (interpolate/forward_fill/seasonal_mean/no): ").strip().lower()
    if fill != 'no':
        gap = int(input("Max gap days (7 recommended): ").strip() or "7")
        merged_df = fill_missing_values(merged_df, fill, gap)
    
    # Features
    final_df = create_features(merged_df)
    
    # Save
    output = input("\n💾 Output filename: ").strip()
    save_processed_data(final_df, output)
    
    print("\n" + "="*70)
    print("✅ PREPARATION COMPLETE!")
    print("="*70)
    print(f"Total rows: {len(final_df):,}")
    print(f"Districts: {final_df['district'].nunique()}")
    print(f"States: {final_df['state'].nunique()}")
    print(f"Date range: {final_df['date'].min()} to {final_df['date'].max()}")
    print("\n📌 This CSV is ready for LSTM training!")
    print("   Simplified 2-feature model: GWL + Rainfall")
    print("="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()