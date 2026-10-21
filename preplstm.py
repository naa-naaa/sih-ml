# Prepares and sequences normalized sliding-window time-series data for LSTM training.
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

print("="*70)
print("  MULTI-DISTRICT LSTM DATA PREPARATION")
print("  All Districts Combined into One CSV")
print("="*70)

# ============================================
# STEP 1: LOAD AND STANDARDIZE ALL DATASETS
# ============================================

def load_dataset(file_path, date_col, value_col, state_col, district_col, dataset_name):
    """
    Load CSV and standardize columns - KEEP ALL DISTRICTS
    """
    print(f"\n📂 Loading {dataset_name.upper()} data...")
    print("-"*70)
    
    try:
        # Check if file exists
        import os
        if not os.path.exists(file_path):
            print(f"   ❌ File not found: {file_path}")
            return None
        
        df = pd.read_csv(file_path)
        print(f"   ✅ Loaded {len(df):,} rows")
        print(f"   📋 Available columns: {list(df.columns)}")
        
        # Check if required columns exist
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
            print(f"   💡 Check spelling - column names are case-sensitive!")
            return None
        
        # Standardize column names
        df = df.rename(columns={
            date_col: 'datetime',
            value_col: 'value',
            state_col: 'state',
            district_col: 'district'
        })
        
        # Parse dates
        print("   🔧 Parsing dates...")
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        invalid_dates = df['datetime'].isna().sum()
        if invalid_dates > 0:
            print(f"   ⚠️  {invalid_dates} rows with invalid dates (will be removed)")
        df = df.dropna(subset=['datetime'])
        
        if len(df) == 0:
            print(f"   ❌ No valid dates found after parsing")
            return None
        
        # Convert values to numeric
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        invalid_values = df['value'].isna().sum()
        if invalid_values > 0:
            print(f"   ⚠️  {invalid_values} rows with invalid values (will be removed)")
        df = df.dropna(subset=['value'])
        
        if len(df) == 0:
            print(f"   ❌ No valid numeric values found")
            return None
        
        # Extract date only (ignore time)
        df['date'] = df['datetime'].dt.date
        df['date'] = pd.to_datetime(df['date'])
        
        # Keep only needed columns
        df = df[['state', 'district', 'date', 'value']]
        
        print(f"   📊 Cleaned: {len(df):,} rows")
        print(f"   📍 States: {df['state'].nunique()} | Districts: {df['district'].nunique()}")
        print(f"   📅 Date range: {df['date'].min()} to {df['date'].max()}")
        
        return df
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# ============================================
# STEP 2: AGGREGATE MULTIPLE READINGS PER DAY
# ============================================

def aggregate_to_daily(df, dataset_name):
    """
    Aggregate to one value per (state, district, date)
    """
    print(f"\n🔄 Aggregating {dataset_name.upper()} to daily...")
    print("-"*70)
    
    # Check for duplicates
    dup_check = df.groupby(['state', 'district', 'date']).size()
    max_readings = dup_check.max()
    
    print(f"   📊 Max readings per day: {max_readings}")
    
    # Aggregate based on dataset type
    if dataset_name == 'rainfall':
        # Rainfall: SUM
        df_daily = df.groupby(['state', 'district', 'date']).agg({
            'value': 'sum'
        }).reset_index()
        print(f"   ✅ Aggregation: SUM (total daily rainfall)")
    else:
        # GWL, ET, Soil Moisture: MEAN
        df_daily = df.groupby(['state', 'district', 'date']).agg({
            'value': 'mean'
        }).reset_index()
        print(f"   ✅ Aggregation: MEAN (average daily value)")
    
    # Rename value column
    df_daily = df_daily.rename(columns={'value': dataset_name})
    
    print(f"   📦 Result: {len(df_daily):,} rows")
    print(f"   📍 Districts: {df_daily['district'].nunique()}")
    
    return df_daily


# ============================================
# STEP 3: SMART MERGE WITH NEAREST DATE MATCHING
# ============================================
def merge_with_nearest_dates(gwl_df, other_df, other_name, max_days_diff=7):
    """
    Merge datasets even if dates don't match exactly
    For each GWL record, find closest date in other dataset within max_days_diff
    
    This solves the "0 overlapping dates" problem!
    """
    print(f"\n🔗 Merging GWL with {other_name.upper()} (nearest date matching)...")
    print("-"*70)
    
    merged_rows = []
    
    # Group by district for faster processing
    districts = gwl_df['district'].unique()
    
    for i, district in enumerate(districts, 1):
        if i % 10 == 0 or i == 1:
            print(f"   Processing district {i}/{len(districts)}...")
        
        # Get data for this district
        gwl_dist = gwl_df[gwl_df['district'] == district].copy()
        other_dist = other_df[other_df['district'] == district].copy()
        
        if len(other_dist) == 0:
            # No data for this district in other dataset - fill with NaN
            gwl_dist[other_name] = np.nan
            merged_rows.append(gwl_dist)
            continue
        
        # ===== CRITICAL FIX: Convert ALL dates to numpy datetime64 format =====
        # This ensures vectorized operations work correctly
        gwl_dates = gwl_dist['date'].values.astype('datetime64[D]')
        other_dates = other_dist['date'].values.astype('datetime64[D]')
        other_values = other_dist[other_name].values
        
        # Pre-allocate result array
        matched_values = np.full(len(gwl_dates), np.nan)
        
        # For each GWL date, find nearest date in other dataset
        for idx, gwl_date in enumerate(gwl_dates):
            # Calculate absolute day differences (vectorized operation)
            time_diffs = np.abs((other_dates - gwl_date).astype('timedelta64[D]').astype(int))
            
            # Find nearest date
            min_diff_idx = np.argmin(time_diffs)
            min_diff = time_diffs[min_diff_idx]
            
            if min_diff <= max_days_diff:
                # Use nearest date if within threshold
                matched_values[idx] = other_values[min_diff_idx]
        
        # Add matched values to dataframe
        gwl_dist[other_name] = matched_values
        merged_rows.append(gwl_dist)
    
    # Concatenate all districts
    merged_df = pd.concat(merged_rows, ignore_index=True)
    
    # Calculate statistics
    total_rows = len(merged_df)
    matched_rows = merged_df[other_name].notna().sum()
    match_rate = (matched_rows / total_rows * 100) if total_rows > 0 else 0
    
    print(f"   ✅ Merge complete!")
    print(f"      Total rows: {total_rows:,}")
    print(f"      Matched: {matched_rows:,} ({match_rate:.1f}%)")
    print(f"      Missing: {total_rows - matched_rows:,} ({100-match_rate:.1f}%)")
    
    return merged_df
    """
    Merge datasets even if dates don't match exactly
    For each GWL record, find closest date in other dataset within max_days_diff
    
    This solves the "0 overlapping dates" problem!
    """
    print(f"\n🔗 Merging GWL with {other_name.upper()} (nearest date matching)...")
    print("-"*70)
    
    merged_rows = []
    
    # Group by district for faster processing
    districts = gwl_df['district'].unique()
    
    for i, district in enumerate(districts, 1):
        if i % 10 == 0 or i == 1:
            print(f"   Processing district {i}/{len(districts)}...")
        
        # Get data for this district
        gwl_dist = gwl_df[gwl_df['district'] == district].copy()
        other_dist = other_df[other_df['district'] == district].copy()
        
        if len(other_dist) == 0:
            # No data for this district in other dataset - skip or fill with NaN
            gwl_dist[other_name] = np.nan
            merged_rows.append(gwl_dist)
            continue
        
        # Convert dates to numpy datetime64 for fast vectorized operations
        gwl_dates = gwl_dist['date'].values.astype('datetime64[D]')
        other_dates = other_dist['date'].values.astype('datetime64[D]')
        other_values = other_dist[other_name].values
        
        # Create array to store matched values
        matched_values = np.full(len(gwl_dates), np.nan)
        
        # For each GWL date, find nearest date in other dataset
        for idx, gwl_date in enumerate(gwl_dates):
            # Calculate absolute day differences
            time_diffs = np.abs((other_dates - gwl_date).astype('timedelta64[D]').astype(int))
            
            # Find nearest date
            min_diff_idx = np.argmin(time_diffs)
            min_diff = time_diffs[min_diff_idx]
            
            if min_diff <= max_days_diff:
                # Use nearest date if within threshold
                matched_values[idx] = other_values[min_diff_idx]
        
        # Add matched values to dataframe
        gwl_dist[other_name] = matched_values
        merged_rows.append(gwl_dist)
    
    merged_df = pd.concat(merged_rows, ignore_index=True)
    
    # Calculate statistics
    total_rows = len(merged_df)
    matched_rows = merged_df[other_name].notna().sum()
    match_rate = (matched_rows / total_rows * 100) if total_rows > 0 else 0
    
    print(f"   ✅ Merge complete!")
    print(f"      Total rows: {total_rows:,}")
    print(f"      Matched: {matched_rows:,} ({match_rate:.1f}%)")
    print(f"      Missing: {total_rows - matched_rows:,} ({100-match_rate:.1f}%)")
    
    return merged_df


# ============================================
# STEP 4: MERGE ALL DATASETS
# ============================================

def merge_all_datasets(gwl_daily, rainfall_daily, et_daily, soilmoisture_daily=None, 
                       max_days_diff=7):
    """
    Merge all datasets using nearest date matching
    """
    print(f"\n🔗 Merging All Datasets...")
    print("-"*70)
    print(f"   Strategy: Nearest date matching (within {max_days_diff} days)")
    
    # Start with GWL (this is our primary dataset)
    merged = gwl_daily.copy()
    print(f"   📦 Starting with GWL: {len(merged):,} rows, {merged['district'].nunique()} districts")
    
    # Merge rainfall
    merged = merge_with_nearest_dates(merged, rainfall_daily, 'rainfall', max_days_diff)
    
    # Merge ET
    merged = merge_with_nearest_dates(merged, et_daily, 'et', max_days_diff)
    
    # Merge Soil Moisture if provided
    if soilmoisture_daily is not None:
        merged = merge_with_nearest_dates(merged, soilmoisture_daily, 'soilmoisture', max_days_diff)
    
    # Sort by state, district, date
    merged = merged.sort_values(['state', 'district', 'date']).reset_index(drop=True)
    
    # Summary
    print(f"\n   📊 Final Merged Dataset:")
    print(f"      Total rows: {len(merged):,}")
    print(f"      States: {merged['state'].nunique()}")
    print(f"      Districts: {merged['district'].nunique()}")
    print(f"      Date range: {merged['date'].min()} to {merged['date'].max()}")
    
    # Check missing values per column
    print(f"\n   📊 Missing Values:")
    for col in ['gwl', 'rainfall', 'et', 'soilmoisture']:
        if col in merged.columns:
            missing = merged[col].isna().sum()
            missing_pct = (missing / len(merged) * 100) if len(merged) > 0 else 0
            print(f"      {col}: {missing:,} ({missing_pct:.1f}%)")
    
    return merged


# ============================================
# STEP 5: FILL MISSING VALUES (OPTIONAL)
# ============================================

def fill_missing_values(df, method='interpolate', max_gap=7):
    """
    Fill missing values within each district's time series
    """
    print(f"\n🔧 Filling missing values...")
    print("-"*70)
    print(f"   Method: {method}")
    print(f"   Max gap: {max_gap} days")
    
    filled_df = []
    
    # Process each district separately
    districts = df['district'].unique()
    
    for district in districts:
        dist_df = df[df['district'] == district].copy()
        dist_df = dist_df.set_index('date').sort_index()
        
        # Fill numeric columns only
        numeric_cols = ['gwl', 'rainfall', 'et', 'soilmoisture']
        numeric_cols = [col for col in numeric_cols if col in dist_df.columns]
        
        if method == 'interpolate':
            # Linear interpolation (limited to max_gap days)
            for col in numeric_cols:
                dist_df[col] = dist_df[col].interpolate(
                    method='linear',
                    limit=max_gap,
                    limit_direction='both'
                )
        elif method == 'forward_fill':
            # Forward fill
            for col in numeric_cols:
                dist_df[col] = dist_df[col].fillna(method='ffill', limit=max_gap)
        elif method == 'seasonal_mean':
            # Fill with day-of-week mean
            dist_df['dow'] = dist_df.index.dayofweek
            for col in numeric_cols:
                for dow in range(7):
                    mask = (dist_df['dow'] == dow) & dist_df[col].isna()
                    mean_val = dist_df[dist_df['dow'] == dow][col].mean()
                    if not np.isnan(mean_val):
                        dist_df.loc[mask, col] = mean_val
            dist_df = dist_df.drop('dow', axis=1)
        
        dist_df = dist_df.reset_index()
        filled_df.append(dist_df)
    
    result = pd.concat(filled_df, ignore_index=True)
    
    # Check remaining missing values
    print(f"\n   📊 Missing Values After Filling:")
    for col in ['gwl', 'rainfall', 'et', 'soilmoisture']:
        if col in result.columns:
            missing = result[col].isna().sum()
            missing_pct = (missing / len(result) * 100) if len(result) > 0 else 0
            print(f"      {col}: {missing:,} ({missing_pct:.1f}%)")
    
    return result


# ============================================
# STEP 6: ADD DERIVED FEATURES
# ============================================

def create_features(df):
    """
    Add derived features per district
    """
    print(f"\n🛠️  Creating derived features...")
    print("-"*70)
    
    feature_df = []
    
    for district in df['district'].unique():
        dist_df = df[df['district'] == district].copy()
        dist_df = dist_df.sort_values('date')
        
        # Effective recharge
        if 'rainfall' in dist_df.columns and 'et' in dist_df.columns:
            dist_df['effective_recharge'] = dist_df['rainfall'] - dist_df['et']
        
        # Rolling averages
        if 'rainfall' in dist_df.columns:
            dist_df['rainfall_7day_ma'] = dist_df['rainfall'].rolling(window=7, min_periods=1).mean()
        
        if 'gwl' in dist_df.columns:
            dist_df['gwl_change'] = dist_df['gwl'].diff()
            dist_df['gwl_7day_ma'] = dist_df['gwl'].rolling(window=7, min_periods=1).mean()
        
        # Temporal features
        dist_df['day_of_year'] = pd.to_datetime(dist_df['date']).dt.dayofyear
        dist_df['month'] = pd.to_datetime(dist_df['date']).dt.month
        dist_df['year'] = pd.to_datetime(dist_df['date']).dt.year
        
        feature_df.append(dist_df)
    
    result = pd.concat(feature_df, ignore_index=True)
    
    # Drop NaN from differencing (first row per district)
    result = result.dropna()
    
    print(f"   ✅ Feature engineering complete")
    print(f"   📊 Final features: {list(result.columns)}")
    print(f"   📦 Final rows: {len(result):,}")
    
    return result


# ============================================
# STEP 7: SAVE PROCESSED DATA
# ============================================

def save_processed_data(df, output_file='multivariate_lstm_all_districts.csv'):
    """
    Save processed data
    """
    print(f"\n💾 Saving processed data...")
    print("-"*70)
    
    df.to_csv(output_file, index=False)
    print(f"   ✅ Saved: {output_file}")
    print(f"   📦 Rows: {len(df):,} | Columns: {len(df.columns)}")
    
    # Per-district summary
    print(f"\n   📊 Data per District:")
    district_summary = df.groupby('district').agg({
        'date': ['min', 'max', 'count']
    })
    print(district_summary.head(10).to_string())
    
    if len(df['district'].unique()) > 10:
        print(f"   ... ({len(df['district'].unique())-10} more districts)")
    
    # Summary statistics
    print(f"\n   📊 Overall Statistics:")
    numeric_cols = ['gwl', 'rainfall', 'et', 'soilmoisture']
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    print(df[numeric_cols].describe().to_string())
    
    return output_file


# ============================================
# MAIN EXECUTION
# ============================================

def main():
    """
    Complete pipeline for multi-district data preparation
    """
    print("\n" + "🌊"*35)
    print("   MULTI-DISTRICT LSTM DATA PREPARATION")
    print("🌊"*35 + "\n")
    
    # Get file paths
    print("📁 Enter file paths:")
    print("-"*70)
    
    gwl_file = input("1. GWL CSV: ").strip()
    rainfall_file = input("2. Rainfall CSV: ").strip()
    et_file = input("3. ET CSV: ").strip()
    
    use_sm = input("4. Use Soil Moisture? (yes/no): ").strip().lower()
    sm_file = None
    if use_sm == 'yes':
        sm_file = input("   Soil Moisture CSV: ").strip()
    
    # Get column mappings
    print("\n🔧 Column Mappings:")
    print("-"*70)
    
    print("\nGWL CSV:")
    gwl_date = input("  Date column: ").strip()
    gwl_value = input("  Value column: ").strip()
    gwl_state = input("  State column: ").strip()
    gwl_district = input("  District column: ").strip()
    
    print("\nRainfall CSV:")
    rain_date = input("  Date column: ").strip()
    rain_value = input("  Value column: ").strip()
    rain_state = input("  State column: ").strip()
    rain_district = input("  District column: ").strip()
    
    print("\nET CSV:")
    et_date = input("  Date column: ").strip()
    et_value = input("  Value column: ").strip()
    et_state = input("  State column: ").strip()
    et_district = input("  District column: ").strip()
    
    sm_date, sm_value, sm_state, sm_district = None, None, None, None
    if sm_file:
        print("\nSoil Moisture CSV:")
        sm_date = input("  Date column: ").strip()
        sm_value = input("  Value column: ").strip()
        sm_state = input("  State column: ").strip()
        sm_district = input("  District column: ").strip()
    
    # Load datasets (ALL districts)
    print("\n" + "="*70)
    print("LOADING DATASETS...")
    print("="*70)
    
    gwl_df = load_dataset(gwl_file, gwl_date, gwl_value, gwl_state, gwl_district, 'gwl')
    rain_df = load_dataset(rainfall_file, rain_date, rain_value, rain_state, rain_district, 'rainfall')
    et_df = load_dataset(et_file, et_date, et_value, et_state, et_district, 'et')
    
    sm_df = None
    if sm_file:
        sm_df = load_dataset(sm_file, sm_date, sm_value, sm_state, sm_district, 'soilmoisture')
    
    # Check which datasets failed
    failed = []
    if gwl_df is None:
        failed.append("GWL")
    if rain_df is None:
        failed.append("Rainfall")
    if et_df is None:
        failed.append("ET")
    if sm_file and sm_df is None:
        failed.append("Soil Moisture")
    
    if failed:
        print(f"\n❌ Failed to load: {', '.join(failed)}")
        print("\n💡 Common issues:")
        print("   1. Wrong file path (check spelling)")
        print("   2. Wrong column names (they are case-sensitive)")
        print("   3. CSV is empty or corrupted")
        print("\n🔍 Please check the error messages above and try again.")
        return
    
    print("\n✅ All datasets loaded successfully!")
    
    # Aggregate to daily
    gwl_daily = aggregate_to_daily(gwl_df, 'gwl')
    rain_daily = aggregate_to_daily(rain_df, 'rainfall')
    et_daily = aggregate_to_daily(et_df, 'et')
    
    sm_daily = None
    if sm_df is not None:
        sm_daily = aggregate_to_daily(sm_df, 'soilmoisture')
    
    # Merge with nearest date matching
    print("\n⚙️  Merge Configuration:")
    max_days = int(input("Max days difference for date matching (recommend 7): ").strip() or "7")
    
    merged_df = merge_all_datasets(gwl_daily, rain_daily, et_daily, sm_daily, max_days)
    
    if len(merged_df) == 0:
        print("\n❌ No data after merging")
        return
    
    # Fill missing values
    fill_method = input("\nFill missing values? (interpolate/forward_fill/seasonal_mean/no): ").strip().lower()
    
    if fill_method != 'no':
        max_gap = int(input("Max gap to fill (days, recommend 7): ").strip() or "7")
        merged_df = fill_missing_values(merged_df, fill_method, max_gap)
    
    # Feature engineering
    final_df = create_features(merged_df)
    
    # Save
    output_file = input("\n💾 Output filename: ").strip()
    save_processed_data(final_df, output_file)
    
    # Final summary
    print("\n" + "="*70)
    print("✅ PREPARATION COMPLETE!")
    print("="*70)
    print(f"Total rows: {len(final_df):,}")
    print(f"Districts: {final_df['district'].nunique()}")
    print(f"States: {final_df['state'].nunique()}")
    print(f"Date range: {final_df['date'].min()} to {final_df['date'].max()}")
    print("\n📌 This CSV is ready for LSTM training!")
    print("   Each district will be trained separately during LSTM phase.")
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