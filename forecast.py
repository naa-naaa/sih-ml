import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

print("="*70)
print("     GROUNDWATER LEVEL FORECASTING - ARIMA/SARIMA MODEL")
print("="*70)

# ============================================
# 1️⃣ FILE UPLOAD & COLUMN MAPPING
# ============================================
def load_and_map_columns():
    """Interactive file loading with column mapping"""
    print("\n📁 STEP 1: Load CSV File")
    print("-"*70)
    
    file_path = input("Enter path to CSV file: ").strip()
    
    try:
        df = pd.read_csv(file_path)
        print(f"✅ File loaded successfully!")
        print(f"   Rows: {len(df):,} | Columns: {len(df.columns)}")
        
        print("\n📋 Available columns:")
        for i, col in enumerate(df.columns, 1):
            print(f"   {i}. {col}")
        
        print("\n🔧 Column Mapping")
        print("-"*70)
        
        # Map columns
        datetime_col = input("Enter column name for DateTime (e.g., 'dateTime'): ").strip()
        value_col = input("Enter column name for GWL Value (e.g., 'dataValue'): ").strip()
        
        # Location columns (optional)
        has_state = input("Does your data have a 'State' column? (yes/no): ").strip().lower()
        if has_state == 'yes':
            state_col = input("Enter column name for State: ").strip()
        else:
            state_col = None
        
        has_district = input("Does your data have a 'District' column? (yes/no): ").strip().lower()
        if has_district == 'yes':
            district_col = input("Enter column name for District: ").strip()
        else:
            district_col = None
        
        # Validate columns exist
        required_cols = [datetime_col, value_col]
        missing = [col for col in required_cols if col not in df.columns]
        
        if missing:
            print(f"❌ Error: Columns not found: {missing}")
            return None, None
        
        return df, {
            'datetime': datetime_col,
            'value': value_col,
            'state': state_col,
            'district': district_col
        }
        
    except Exception as e:
        print(f"❌ Error loading file: {str(e)}")
        return None, None

# ============================================
# 2️⃣ LOCATION SELECTION
# ============================================
def select_location(df, col_map):
    """Select state and district if available"""
    
    # If no location columns, use all data
    if col_map['state'] is None and col_map['district'] is None:
        print("\n📍 No location columns found. Using all data.")
        return df
    
    print("\n📍 STEP 2: Select Location")
    print("-"*70)
    
    # State selection
    if col_map['state']:
        states = df[col_map['state']].unique()
        print(f"\n🌐 Available States ({len(states)}):")
        for i, state in enumerate(sorted(states), 1):
            print(f"   {i}. {state}")
        
        state = input("\nEnter State name (exact match): ").strip()
        df = df[df[col_map['state']] == state]
        
        if len(df) == 0:
            print(f"❌ No data found for state: {state}")
            return None
        print(f"   ✅ Filtered to {len(df):,} records for {state}")
    
    # District selection
    if col_map['district']:
        districts = df[col_map['district']].unique()
        print(f"\n🏘️ Available Districts ({len(districts)}):")
        for i, district in enumerate(sorted(districts), 1):
            print(f"   {i}. {district}")
        
        district = input("\nEnter District name (exact match): ").strip()
        df = df[df[col_map['district']] == district]
        
        if len(df) == 0:
            print(f"❌ No data found for district: {district}")
            return None
        print(f"   ✅ Filtered to {len(df):,} records for {district}")
    
    return df

# ============================================
# 3️⃣ DATA PREPROCESSING WITH AUGMENTATION
# ============================================
def preprocess_timeseries(df, col_map):
    """
    Advanced preprocessing with data augmentation
    Handles: duplicates, irregular timestamps, missing values, outliers
    """
    print("\n🔧 STEP 3: Data Preprocessing & Augmentation")
    print("-"*70)
    
    df = df.copy()
    datetime_col = col_map['datetime']
    value_col = col_map['value']
    
    # Convert datetime
    print("\n1️⃣ Converting datetime...")
    try:
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        print(f"   ✅ Datetime range: {df[datetime_col].min()} to {df[datetime_col].max()}")
    except Exception as e:
        print(f"   ❌ Error parsing dates: {str(e)}")
        return None
    
    # Convert values to numeric
    print("\n2️⃣ Converting values to numeric...")
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
    
    # Remove NaN values
    original_count = len(df)
    df = df.dropna(subset=[value_col])
    removed = original_count - len(df)
    if removed > 0:
        print(f"   ⚠️ Removed {removed} rows with invalid values")
    
    # Remove outliers (using IQR method)
    print("\n3️⃣ Removing outliers...")
    Q1 = df[value_col].quantile(0.25)
    Q3 = df[value_col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR  # 3x IQR for more tolerance
    upper_bound = Q3 + 3 * IQR
    
    outliers_mask = (df[value_col] < lower_bound) | (df[value_col] > upper_bound)
    outliers_count = outliers_mask.sum()
    
    if outliers_count > 0:
        print(f"   ⚠️ Found {outliers_count} outliers")
        remove_outliers = input("   Remove outliers? (yes/no): ").strip().lower()
        if remove_outliers == 'yes':
            df = df[~outliers_mask]
            print(f"   ✅ Removed {outliers_count} outliers")
    
    # Extract date only (remove time component)
    print("\n4️⃣ Extracting date (removing time component)...")
    df['date'] = df[datetime_col].dt.date
    df['date'] = pd.to_datetime(df['date'])
    # Handle multiple readings per day (CRITICAL FIX for duplicates)
# Handle multiple readings per day (CRITICAL FIX for duplicates)
print("\n5️⃣ Handling multiple readings per day...")
readings_per_day = df.groupby('date').size()
max_readings = readings_per_day.max()

if max_readings > 1:
    print(f"   ⚠️ Found up to {max_readings} readings per day")
    print("   Aggregating multiple readings per day (taking mean)...")
    
    # Aggregate: mean for value, count for tracking
    df_daily = df.groupby('date').agg({
        value_col: 'mean',  # Take mean of multiple readings
        datetime_col: 'count'  # Count number of readings
    }).reset_index()
    df_daily.columns = ['date', 'value', 'num_readings']
else:
    df_daily = df[['date', value_col]].copy()
    df_daily.columns = ['date', 'value']
    df_daily['num_readings'] = 1

# Explicitly remove duplicate dates if any
duplicate_dates = df_daily[df_daily['date'].duplicated()]
if not duplicate_dates.empty:
    print("⚠️ Warning: Duplicate dates found after aggregation. Removing duplicates...")
    print(duplicate_dates)
    df_daily = df_daily.drop_duplicates(subset='date')

# Remove duplicate dates if any (to avoid reindex error)
duplicate_dates = df_daily[df_daily['date'].duplicated()]
if not duplicate_dates.empty:
    print("⚠️ Warning: Duplicate dates found after aggregation. Removing duplicates...")
    print(duplicate_dates)
    df_daily = df_daily.drop_duplicates(subset='date')

# Set date as index
df_daily = df_daily.set_index('date').sort_index()

# Create complete daily range (DATA AUGMENTATION)
print("\n6️⃣ Creating continuous daily time series (augmentation)...")
full_date_range = pd.date_range(
    start=df_daily.index.min(),
    end=df_daily.index.max(),
    freq='D'
)

# Reindex to daily frequency (this fills missing dates with NaN)
df_daily = df_daily.reindex(full_date_range)

# Count missing days
missing_days = df_daily['value'].isna().sum()
total_days = len(df_daily)

print(f"   📊 Time series summary:")
print(f"      Total days: {total_days}")
print(f"      Days with data: {total_days - missing_days}")
print(f"      Missing days: {missing_days} ({missing_days/total_days*100:.1f}%)")

# Fill missing values using multiple strategies
if missing_days > 0:
    print("\n7️⃣ Filling missing values...")
    
    # Strategy 1: Linear interpolation for small gaps (≤7 days)
    df_daily['value'] = df_daily['value'].interpolate(method='linear', limit=7)
    
    # Strategy 2: Forward fill for remaining gaps (≤14 days)
    df_daily['value'] = df_daily['value'].fillna(method='ffill', limit=14)
    
    # Strategy 3: Backward fill for any remaining gaps
    df_daily['value'] = df_daily['value'].fillna(method='bfill', limit=14)
    
    # Check if any NaN remain
    remaining_nan = df_daily['value'].isna().sum()
    
    if remaining_nan > 0:
        print(f"   ⚠️ {remaining_nan} days still missing (large gaps)")
        print("   Filling with seasonal mean...")
        
        # Fill with mean of same day of week
        df_daily['day_of_week'] = df_daily.index.dayofweek
        for dow in range(7):
            mask = (df_daily['day_of_week'] == dow) & df_daily['value'].isna()
            mean_value = df_daily[df_daily['day_of_week'] == dow]['value'].mean()
            df_daily.loc[mask, 'value'] = mean_value
        
        # Final fallback: overall mean
        df_daily['value'] = df_daily['value'].fillna(df_daily['value'].mean())
    
    print(f"   ✅ All gaps filled")

# Final check: ensure no NaN values remain
if df_daily['value'].isna().any():
    print("   ⚠️ Some NaN values remain. Dropping...")
    df_daily = df_daily.dropna(subset=['value'])

# Get clean time series
ts = df_daily['value']

# Summary statistics
print("\n📊 Final Time Series Summary:")
print(f"   Length: {len(ts)} days ({len(ts)/365:.1f} years)")
print(f"   Date range: {ts.index.min()} to {ts.index.max()}")
print(f"   Mean: {ts.mean():.2f}")
print(f"   Std Dev: {ts.std():.2f}")
print(f"   Min: {ts.min():.2f}")
print(f"   Max: {ts.max():.2f}")

return ts

# ============================================
# 4️⃣ SEASONALITY DETECTION
# ============================================
def detect_seasonality(ts):
    """
    Detect if data has seasonal patterns
    Returns: (has_seasonality, seasonal_period)
    """
    print("\n🔍 STEP 4: Seasonality Detection")
    print("-"*70)
    
    # Need at least 2 years of data for reliable seasonal detection
    if len(ts) < 730:
        print("   ⚠️ Insufficient data for seasonal analysis (<2 years)")
        return False, None
    
    # Use autocorrelation to detect seasonality
    from statsmodels.tsa.stattools import acf
    
    # Calculate autocorrelation
    acf_values = acf(ts, nlags=min(365, len(ts)//2), fft=True)
    
    # Check for peaks at weekly (7 days) and yearly (365 days) periods
    weekly_acf = acf_values[7] if len(acf_values) > 7 else 0
    monthly_acf = acf_values[30] if len(acf_values) > 30 else 0
    yearly_acf = acf_values[365] if len(acf_values) > 365 else 0
    
    print(f"   Autocorrelation analysis:")
    print(f"      Weekly (lag=7): {weekly_acf:.3f}")
    print(f"      Monthly (lag=30): {monthly_acf:.3f}")
    if len(acf_values) > 365:
        print(f"      Yearly (lag=365): {yearly_acf:.3f}")
    
    # Decision logic
    threshold = 0.3  # ACF threshold for seasonality
    
    if yearly_acf > threshold and len(ts) > 730:
        print(f"   ✅ Strong yearly seasonality detected (ACF={yearly_acf:.3f})")
        print(f"   📌 Recommendation: Use SARIMA with m=365 (yearly)")
        return True, 365
    elif monthly_acf > threshold:
        print(f"   ✅ Monthly seasonality detected (ACF={monthly_acf:.3f})")
        print(f"   📌 Recommendation: Use SARIMA with m=30 (monthly)")
        return True, 30
    elif weekly_acf > threshold:
        print(f"   ✅ Weekly seasonality detected (ACF={weekly_acf:.3f})")
        print(f"   📌 Recommendation: Use SARIMA with m=7 (weekly)")
        return True, 7
    else:
        print(f"   ℹ️ No strong seasonality detected")
        print(f"   📌 Recommendation: Use ARIMA (non-seasonal)")
        return False, None

# ============================================
# 5️⃣ MODEL SELECTION & TRAINING
# ============================================
def train_model(ts, has_seasonality, seasonal_period):
    """
    Train ARIMA or SARIMA based on seasonality detection
    """
    print("\n🤖 STEP 5: Model Training")
    print("-"*70)
    
    # Split into train/test (80/20)
    train_size = int(len(ts) * 0.8)
    train = ts[:train_size]
    test = ts[train_size:]
    
    print(f"   Training set: {len(train)} days")
    print(f"   Test set: {len(test)} days")
    
    if has_seasonality and seasonal_period:
        print(f"\n   🔄 Training SARIMA with seasonal period m={seasonal_period}...")
        print("   (This may take 3-10 minutes depending on data size...)")
        
        try:
            # For large seasonal periods (365), use simpler seasonal orders
            if seasonal_period >= 365:
                model_auto = auto_arima(
                    train,
                    seasonal=True,
                    m=seasonal_period,
                    start_p=0, start_q=0, max_p=3, max_q=3,
                    start_P=0, start_Q=0, max_P=1, max_Q=1,  # Simplified seasonal
                    d=None, D=None,
                    trace=False,
                    error_action='ignore',
                    suppress_warnings=True,
                    stepwise=True,
                    n_jobs=-1,
                    maxiter=50
                )
            else:
                model_auto = auto_arima(
                    train,
                    seasonal=True,
                    m=seasonal_period,
                    start_p=0, start_q=0, max_p=5, max_q=5,
                    start_P=0, start_Q=0, max_P=2, max_Q=2,
                    d=None, D=None,
                    trace=False,
                    error_action='ignore',
                    suppress_warnings=True,
                    stepwise=True,
                    n_jobs=-1
                )
        except Exception as e:
            print(f"   ⚠️ SARIMA training failed: {str(e)}")
            print("   📌 Falling back to non-seasonal ARIMA...")
            has_seasonality = False
    
    if not has_seasonality:
        print(f"\n   🔄 Training ARIMA (non-seasonal)...")
        print("   (This may take 1-3 minutes...)")
        
        model_auto = auto_arima(
            train,
            seasonal=False,
            start_p=0, start_q=0, max_p=5, max_q=5,
            d=None,
            trace=False,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True,
            n_jobs=-1
        )
    
    # Get model parameters
    order = model_auto.order
    seasonal_order = model_auto.seasonal_order if has_seasonality else (0,0,0,0)
    
    print(f"\n   ✅ Model training complete!")
    print(f"      Model type: {'SARIMA' if has_seasonality else 'ARIMA'}")
    print(f"      Order (p,d,q): {order}")
    if has_seasonality:
        print(f"      Seasonal order (P,D,Q,m): {seasonal_order}")
    print(f"      AIC: {model_auto.aic():.2f}")
    print(f"      BIC: {model_auto.bic():.2f}")
    
    # Fit final model on full data
    print(f"\n   🔄 Refitting on full dataset...")
    
    model_final = SARIMAX(
        ts,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    
    results = model_final.fit(disp=False)
    
    # Validate on test set
    print(f"\n   📊 Validating on test set...")
    test_predictions = results.predict(start=train_size, end=len(ts)-1)
    
    rmse_test = np.sqrt(mean_squared_error(test, test_predictions))
    mae_test = mean_absolute_error(test, test_predictions)
    r2_test = r2_score(test, test_predictions)
    
    print(f"      RMSE: {rmse_test:.3f}m")
    print(f"      MAE: {mae_test:.3f}m")
    print(f"      R²: {r2_test:.3f}")
    
    if r2_test > 0.8:
        print("      🌟 Excellent model performance!")
    elif r2_test > 0.6:
        print("      ✅ Good model performance")
    elif r2_test > 0.4:
        print("      ⚠️ Moderate performance - consider LSTM")
    else:
        print("      ❌ Poor performance - try LSTM or more data")
    
    return results, order, seasonal_order, (rmse_test, mae_test, r2_test)

# ============================================
# 6️⃣ FORECASTING
# ============================================
def generate_forecasts(model_results, ts, forecast_days=[7, 30]):
    """
    Generate forecasts for multiple horizons
    """
    print("\n🔮 STEP 6: Generating Forecasts")
    print("-"*70)
    
    forecasts = {}
    
    for days in forecast_days:
        print(f"\n   Forecasting {days} days ahead...")
        
        # Generate forecast
        forecast = model_results.get_forecast(steps=days)
        forecast_mean = forecast.predicted_mean
        forecast_ci = forecast.conf_int()
        
        # Create forecast dates
        last_date = ts.index[-1]
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=days,
            freq='D'
        )
        
        # Create DataFrame
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'forecast': forecast_mean.values,
            'lower_ci': forecast_ci.iloc[:, 0].values,
            'upper_ci': forecast_ci.iloc[:, 1].values
        })
        
        forecasts[days] = forecast_df
        
        print(f"      Day 1: {forecast_mean.iloc[0]:.2f}m")
        print(f"      Day {days}: {forecast_mean.iloc[-1]:.2f}m")
        
        # Trend analysis
        trend = forecast_mean.iloc[-1] - ts.iloc[-1]
        if abs(trend) > 0.5:
            direction = "rising" if trend > 0 else "declining"
            print(f"      ⚠️ Trend: {direction} by {abs(trend):.2f}m")
    
    return forecasts

# ============================================
# 7️⃣ VISUALIZATION
# ============================================
def plot_forecasts(ts, forecasts, save_name="forecast"):
    """
    Create visualization of forecasts
    """
    print("\n📊 Generating visualizations...")
    
    fig, axes = plt.subplots(len(forecasts), 1, figsize=(15, 5*len(forecasts)))
    
    if len(forecasts) == 1:
        axes = [axes]
    
    for idx, (days, forecast_df) in enumerate(forecasts.items()):
        ax = axes[idx]
        
        # Plot historical data (last 180 days for clarity)
        lookback = min(180, len(ts))
        ax.plot(ts.index[-lookback:], ts.values[-lookback:],
                label='Historical', color='blue', linewidth=2)
        
        # Plot forecast
        ax.plot(forecast_df['date'], forecast_df['forecast'],
                label=f'{days}-day Forecast', color='red',
                linewidth=2, linestyle='--', marker='o', markersize=4)
        
        # Plot confidence interval
        ax.fill_between(forecast_df['date'],
                        forecast_df['lower_ci'],
                        forecast_df['upper_ci'],
                        color='red', alpha=0.2,
                        label='95% Confidence Interval')
        
        ax.set_title(f'{days}-Day Groundwater Level Forecast', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Groundwater Level (m)', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()  # Lower values (more water) at top
    
    plt.tight_layout()
    plt.savefig(f'{save_name}.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Plot saved: {save_name}.png")
    plt.show()

# ============================================
# 8️⃣ SAVE RESULTS
# ============================================
def save_results(forecasts, model_info, save_name="forecast_results"):
    """
    Save forecasts to CSV
    """
    print("\n💾 Saving results...")
    
    for days, forecast_df in forecasts.items():
        filename = f'{save_name}_{days}day.csv'
        forecast_df.to_csv(filename, index=False)
        print(f"   ✅ Saved: {filename}")
    
    # Save model info
    with open(f'{save_name}_model_info.txt', 'w') as f:
        f.write("="*50 + "\n")
        f.write("MODEL INFORMATION\n")
        f.write("="*50 + "\n")
        f.write(f"Model Type: {model_info['type']}\n")
        f.write(f"Order: {model_info['order']}\n")
        f.write(f"Seasonal Order: {model_info['seasonal_order']}\n")
        f.write(f"Test RMSE: {model_info['test_metrics'][0]:.3f}m\n")
        f.write(f"Test MAE: {model_info['test_metrics'][1]:.3f}m\n")
        f.write(f"Test R²: {model_info['test_metrics'][2]:.3f}\n")
    
    print(f"   ✅ Saved: {save_name}_model_info.txt")

# ============================================
# 🚀 MAIN EXECUTION
# ============================================
def main():
    """
    Main execution pipeline
    """
    print("\n" + "🌊"*35)
    print("   GROUNDWATER FORECASTING - ARIMA/SARIMA")
    print("   Smart India Hackathon 2024")
    print("🌊"*35 + "\n")
    
    # Step 1: Load data
    df, col_map = load_and_map_columns()
    if df is None:
        return
    
    # Step 2: Select location
    df_filtered = select_location(df, col_map)
    if df_filtered is None or len(df_filtered) == 0:
        print("❌ No data after filtering. Exiting.")
        return
    
    # Step 3: Preprocess
    ts = preprocess_timeseries(df_filtered, col_map)
    if ts is None or len(ts) < 365:
        print("❌ Insufficient data after preprocessing. Need at least 1 year.")
        return
    
    # Step 4: Detect seasonality
    has_seasonality, seasonal_period = detect_seasonality(ts)
    
    # Step 5: Train model
    model_results, order, seasonal_order, test_metrics = train_model(
        ts, has_seasonality, seasonal_period
    )
    
    # Step 6: Generate forecasts
    forecast_horizons = [7, 30]  # Default: 7 and 30 days
    custom = input("\nUse custom forecast horizons? (yes/no): ").strip().lower()
    if custom == 'yes':
        days_input = input("Enter forecast days (comma-separated, e.g., 7,15,30,90): ")
        forecast_horizons = [int(d.strip()) for d in days_input.split(',')]
    
    forecasts = generate_forecasts(model_results, ts, forecast_horizons)
    
    # Step 7: Visualize
    show_plots = input("\nGenerate plots? (yes/no): ").strip().lower()
    if show_plots == 'yes':
        save_name = input("Enter base filename for plots (e.g., 'alappuzha_forecast'): ").strip()
        plot_forecasts(ts, forecasts, save_name)
    
    # Step 8: Save results
    save = input("\nSave forecast results to CSV? (yes/no): ").strip().lower()
    if save == 'yes':
        save_name = input("Enter base filename (e.g., 'alappuzha'): ").strip()
        model_info = {
            'type': 'SARIMA' if has_seasonality else 'ARIMA',
            'order': order,
            'seasonal_order': seasonal_order,
            'test_metrics': test_metrics
        }
        save_results(forecasts, model_info, save_name)
    
    # Print final summary
    print("\n" + "="*70)
    print("✅ FORECAST GENERATION COMPLETE!")
    print("="*70)
    print(f"Model: {'SARIMA' + str(seasonal_order) if has_seasonality else 'ARIMA' + str(order)}")
    print(f"Test RMSE: {test_metrics[0]:.3f}m | MAE: {test_metrics[1]:.3f}m | R²: {test_metrics[2]:.3f}")
    print(f"Forecasts generated for: {', '.join([str(d) + ' days' for d in forecasts.keys()])}")
    print("="*70)

# ============================================
# RUN THE PROGRAM
# ============================================
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Program interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()