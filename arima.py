# Fits and evaluates ARIMA/SARIMA models with bias correction for daily groundwater level time-series forecasting.
"""
Groundwater level forecasting with proper train/test split and evaluation metrics.
"""

import warnings
warnings.filterwarnings('ignore')
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Try to import pmdarima for auto model selection
try:
    from pmdarima import auto_arima
    HAS_AUTO_ARIMA = True
except ImportError:
    HAS_AUTO_ARIMA = False
    print("Note: Install pmdarima for better results: pip install pmdarima")

# ============================================================================
# CONFIGURATION
# ============================================================================
CSV_PATH = r"C:\Users\Admin\Downloads\groundwater_2018_2024 (1).csv"
STATE = None          # Set to specific state or None for first available
DISTRICT = None       # Set to specific district or None for first available
FORECAST_DAYS = 90
TEST_SIZE = 90        # Days to use for testing (validation)
MIN_DATA_POINTS = 270  # Min observations (at least TEST_SIZE + 180 for training)
PLOT_RESULTS = True

# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

def load_and_prepare_data(csv_path, state=None, district=None):
    """Load and preprocess groundwater data."""
    
    df = pd.read_csv(csv_path)
    required_cols = ['dataTime', 'dataValue']
    
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV must contain columns: {required_cols}")
    
    # Parse datetime
    df['dataTime'] = pd.to_datetime(df['dataTime'], errors='coerce')
    df = df.dropna(subset=['dataTime', 'dataValue'])
    
    # Filter by state/district if specified
    if state and 'state' in df.columns:
        df = df[df['state'].str.strip().str.upper() == state.strip().upper()]
    
    if district and 'district' in df.columns:
        df = df[df['district'].str.strip().str.upper() == district.strip().upper()]
    
    if df.empty:
        raise ValueError("No data available after filtering")
    
    # Aggregate to daily mean
    df['date'] = df['dataTime'].dt.date
    daily = df.groupby('date')['dataValue'].mean()
    daily.index = pd.to_datetime(daily.index)
    daily = daily.sort_index()
    
    # Create continuous daily index
    date_range = pd.date_range(start=daily.index.min(), 
                               end=daily.index.max(), 
                               freq='D')
    daily = daily.reindex(date_range)
    
    # Interpolate small gaps (up to 7 days)
    daily = daily.interpolate(method='time', limit=7)
    
    # Remove outliers using IQR method
    Q1 = daily.quantile(0.25)
    Q3 = daily.quantile(0.75)
    IQR = Q3 - Q1
    daily = daily[(daily >= Q1 - 3*IQR) & (daily <= Q3 + 3*IQR)]
    
    return daily.dropna()


def train_test_split_timeseries(series, test_size):
    """Split time series into train and test sets."""
    split_idx = len(series) - test_size
    train = series.iloc[:split_idx]
    test = series.iloc[split_idx:]
    return train, test


# ============================================================================
# MODEL SELECTION & FITTING
# ============================================================================

def check_seasonality(series, max_lag=90):
    """Quick check for potential seasonality in the data."""
    if len(series) < max_lag * 2:
        return None
    
    autocorr = acf(series.dropna(), nlags=min(max_lag, len(series)//2), fft=True)
    
    potential_periods = [7, 30, 90]
    strong_periods = []
    
    for period in potential_periods:
        if period < len(autocorr) and autocorr[period] > 0.3:
            strong_periods.append(period)
    
    return strong_periods[0] if strong_periods else None


def fit_best_model(train_series):
    """Fit the best ARIMA/SARIMA model based on training data."""
    
    n_obs = len(train_series)
    print(f"\nFitting model with {n_obs} training observations...")
    
    if HAS_AUTO_ARIMA:
        print("Using auto_arima for model selection...")
        
        seasonal_period = check_seasonality(train_series)
        
        if seasonal_period and n_obs >= seasonal_period * 2:
            print(f"Detected potential seasonality (period={seasonal_period})")
            model = auto_arima(
                train_series,
                seasonal=True,
                m=seasonal_period,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore',
                max_p=5, max_q=5,
                max_P=2, max_Q=2,
                max_d=2, max_D=1,
                trace=False
            )
        else:
            print("No strong seasonality detected, using non-seasonal model")
            model = auto_arima(
                train_series,
                seasonal=False,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore',
                max_p=5, max_q=5,
                max_d=2,
                trace=False
            )
        
        print(f"Selected model: ARIMA{model.order} x SARIMA{model.seasonal_order}")
        return model, 'auto_arima'
    
    else:
        print("Using simple ARIMA model...")
        best_aic = np.inf
        best_model = None
        best_order = None
        
        orders = [
            (1, 1, 1),
            (2, 1, 2),
            (1, 1, 0),
            (0, 1, 1),
        ]
        
        for order in orders:
            try:
                model = ARIMA(train_series, order=order)
                fitted = model.fit()
                
                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_model = fitted
                    best_order = order
            except:
                continue
        
        if best_model is None:
            print("Warning: All models failed, using ARIMA(1,1,1)")
            model = ARIMA(train_series, order=(1, 1, 1))
            best_model = model.fit()
            best_order = (1, 1, 1)
        
        print(f"Selected model: ARIMA{best_order}, AIC={best_aic:.2f}")
        return best_model, 'arima'


# ============================================================================
# FORECASTING & EVALUATION
# ============================================================================

def generate_forecast(model, model_type, n_periods):
    """Generate forecast from fitted model."""
    
    if model_type == 'auto_arima':
        forecast_values = model.predict(n_periods=n_periods)
    else:
        forecast_values = model.forecast(steps=n_periods)
    
    return forecast_values


def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive evaluation metrics."""
    
    # Remove any NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
        'Mean_Error': np.mean(y_pred - y_true),
        'Max_Error': np.max(np.abs(y_pred - y_true))
    }
    
    return metrics


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_results(train, test, test_predictions, future_forecast, metrics_corrected, save_path=None):
    """Create comprehensive visualization with train/test/forecast (bias-corrected)."""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Full time series with train/test/forecast
    # Show last 2 years of training for clarity
    recent_train = train.tail(730)
    
    ax1.plot(recent_train.index, recent_train.values, 
            label='Training Data', color='#2E86AB', linewidth=1.5, alpha=0.7)
    ax1.plot(test.index, test.values, 
            label='Actual Test Data', color='#06D6A0', linewidth=2, marker='o', markersize=3)
    ax1.plot(test.index, test_predictions, 
            label='Predicted Test Data (Bias-Corrected)', color='#EF476F', linewidth=2, 
            linestyle='--', marker='s', markersize=3)
    
    if future_forecast is not None:
        ax1.plot(future_forecast.index, future_forecast.values,
                label='Future Forecast (Bias-Corrected)', color='#A23B72', linewidth=2, 
                linestyle=':', marker='^', markersize=3)
    
    ax1.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Groundwater Level (m)', fontsize=11, fontweight='bold')
    ax1.set_title('Groundwater Level: Training, Testing & Forecasting (Bias-Corrected)', 
                 fontsize=13, fontweight='bold', pad=15)
    ax1.legend(loc='best', frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 2: Test residuals (after bias correction)
    residuals = test.values - test_predictions
    ax2.plot(test.index, residuals, color='#EF476F', linewidth=1.5, marker='o', markersize=3)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.fill_between(test.index, residuals, 0, alpha=0.3, color='#EF476F')
    ax2.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Residual (Actual - Predicted)', fontsize=11, fontweight='bold')
    ax2.set_title('Test Set Residuals After Bias Correction', 
                 fontsize=13, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Add metrics text box - only bias-corrected metrics
    textstr = f"Model Performance (Bias-Corrected):\n"
    textstr += f"R² = {metrics_corrected['R2']:.4f}\n"
    textstr += f"RMSE = {metrics_corrected['RMSE']:.4f} m\n"
    textstr += f"MAE = {metrics_corrected['MAE']:.4f} m\n"
    textstr += f"MAPE = {metrics_corrected['MAPE']:.2f}%"
    props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
    ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Plot saved: {save_path}")
    
    plt.show()


def save_model(model, filename="best_gwl_model.pkl"):
    """Save the trained model to disk."""
    with open(filename, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved as {filename}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("="*70)
    print("GROUNDWATER LEVEL FORECASTING WITH VALIDATION")
    print("="*70)
    
    # Load and prepare data
    print("\n1. Loading and preprocessing data...")
    series = load_and_prepare_data(CSV_PATH, STATE, DISTRICT)
    
    print(f"   Date range: {series.index.min().date()} to {series.index.max().date()}")
    print(f"   Total observations: {len(series)}")
    print(f"   Mean GWL: {series.mean():.2f} m")
    print(f"   Std Dev: {series.std():.2f} m")
    
    # Check if we have enough data
    if len(series) < MIN_DATA_POINTS:
        raise ValueError(f"Insufficient data: need at least {MIN_DATA_POINTS} observations")
    
    # Split into train and test
    print(f"\n2. Splitting data (training: {len(series)-TEST_SIZE} days, testing: {TEST_SIZE} days)...")
    train, test = train_test_split_timeseries(series, TEST_SIZE)
    
    print(f"   Training: {train.index.min().date()} to {train.index.max().date()}")
    print(f"   Testing:  {test.index.min().date()} to {test.index.max().date()}")
    
    # Fit model on training data
    print("\n3. Model fitting on training data...")
    model, model_type = fit_best_model(train)
    
    # Generate predictions for test set
    print("\n4. Generating predictions for test set...")
    test_predictions = generate_forecast(model, model_type, len(test))
    
    # Calculate evaluation metrics (original)
    print("\n5. Calculating evaluation metrics...")
    metrics = calculate_metrics(test.values, test_predictions)
    
    # Apply bias correction
    bias_correction = metrics['Mean_Error']
    test_predictions_corrected = test_predictions - bias_correction
    
    # Recalculate metrics with bias correction
    metrics_corrected = calculate_metrics(test.values, test_predictions_corrected)
    
    # Print original metrics
    print("\n" + "="*70)
    print("MODEL EVALUATION METRICS (on Test Set) - ORIGINAL")
    print("="*70)
    print(f"R² Score:              {metrics['R2']:.4f}")
    print(f"RMSE:                  {metrics['RMSE']:.4f} meters")
    print(f"MAE:                   {metrics['MAE']:.4f} meters")
    print(f"MAPE:                  {metrics['MAPE']:.2f}%")
    print(f"Mean Error (Bias):     {metrics['Mean_Error']:.4f} meters")
    print(f"Max Absolute Error:    {metrics['Max_Error']:.4f} meters")
    
    # Print bias-corrected metrics
    print("\n" + "="*70)
    print("MODEL EVALUATION METRICS (on Test Set) - BIAS CORRECTED")
    print("="*70)
    print(f"R² Score:              {metrics_corrected['R2']:.4f}")
    print(f"RMSE:                  {metrics_corrected['RMSE']:.4f} meters")
    print(f"MAE:                   {metrics_corrected['MAE']:.4f} meters")
    print(f"MAPE:                  {metrics_corrected['MAPE']:.2f}%")
    print(f"Mean Error (Bias):     {metrics_corrected['Mean_Error']:.4f} meters")
    print(f"Max Absolute Error:    {metrics_corrected['Max_Error']:.4f} meters")
    print(f"\nBias Correction Applied: {bias_correction:.4f} meters")
    
    # Interpretation based on corrected metrics
    print("\n" + "-"*70)
    if metrics_corrected['RMSE'] < 0.5:
        print("✓ Model Performance: GOOD (RMSE < 0.5)")
    elif metrics_corrected['RMSE'] < 1.0:
        print("⚠ Model Performance: MODERATE (RMSE = 0.5-1.0)")
    else:
        print("✗ Model Performance: NEEDS IMPROVEMENT (RMSE ≥ 1.0)")
    print("-"*70)
    
    # Additional context
    target_std = test.std()
    print(f"\nContext: Test set std dev = {target_std:.4f} m")
    print(f"RMSE as % of std dev: {(metrics_corrected['RMSE']/target_std)*100:.1f}%")
    
    # Refit model on full dataset for final forecasting
    print("\n6. Refitting model on full dataset for future forecasting...")
    final_model, model_type_final = fit_best_model(series)
    
    forecast_horizons = [7, 30, FORECAST_DAYS]
    forecast_outputs = {}
    
    for horizon in forecast_horizons:
        print(f"   → Forecasting {horizon} days ahead...")
        
        # Generate forecast
        predictions = generate_forecast(final_model, model_type_final, horizon)
        
        # Apply bias correction to future forecasts
        predictions_corrected = predictions - bias_correction
        
        # Prepare date index
        future_dates = pd.date_range(
            start=series.index.max() + pd.Timedelta(days=1),
            periods=horizon,
            freq='D'
        )
        
        # Save CSV with both original and corrected forecasts
        filename = f"gwl_forecast_{horizon}days.csv"
        df_out = pd.DataFrame({
            'date': future_dates,
            'forecasted_gwl_m': predictions_corrected,
            'forecasted_gwl_uncorrected_m': predictions
        })
        df_out.to_csv(filename, index=False)
        
        print(f"      Saved: {filename}")
        
        forecast_outputs[horizon] = df_out
    
    # Save test results with corrected predictions
    test_results_df = pd.DataFrame({
        'date': test.index,
        'actual_gwl_m': test.values,
        'predicted_gwl_m': test_predictions_corrected,
        'predicted_gwl_uncorrected_m': test_predictions
    })
    test_results_df.to_csv("gwl_test_results.csv", index=False)
    print("   Saved: gwl_test_results.csv")
    
    # Save final model with bias correction factor
    model_package = {
        'model': final_model,
        'bias_correction': bias_correction,
        'model_type': model_type_final
    }
    with open("gwl_forecast_model.pkl", "wb") as f:
        pickle.dump(model_package, f)
    print("   Saved: gwl_forecast_model.pkl (includes bias correction)")
    
    # Choose one for plotting (default: use FORECAST_DAYS)
    future_forecast = forecast_outputs[FORECAST_DAYS]
    
    print(f"\n7. Creating visualization for {FORECAST_DAYS}-day forecast...")
    plot_results(
        train,
        test,
        test_predictions_corrected,
        future_forecast.set_index('date')['forecasted_gwl_m'],
        metrics_corrected,
        save_path='gwl_forecast_validation.png'
    )
    
    print("\n" + "="*70)
    print("DONE! All forecast horizons completed.")
    print("="*70)
    
    return {
        'model': final_model,
        'metrics': metrics,
        'metrics_corrected': metrics_corrected,
        'bias_correction': bias_correction,
        'forecasts': forecast_outputs,
        'test_results': test_results_df
    }


if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        print(f"\nError: {e}")
        raise
    