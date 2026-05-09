import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF info/warning messages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import warnings
warnings.filterwarnings("ignore")

print("="*70)
print("     LSTM TRAINING - GWL + RAINFALL FORECASTING")
print("="*70)

# ============================================
# CONFIGURATION
# ============================================

# Feature selection
INPUT_FEATURES = [
    'gwl',
    'rainfall',
    'rainfall_7day_ma',
    'gwl_7day_ma',
    'gwl_30day_ma',
    'day_of_year',
    'month',
    'is_monsoon'
]

TARGET_FEATURE = 'gwl'

LOOKBACK_DAYS = 30       # Use past 30 days to predict future
FORECAST_HORIZON = 7     # Predict next 7 days
TEST_SPLIT = 0.2         # 80% train, 20% test

# ============================================
# STEP 1: LOAD DATA
# ============================================

def load_data(file_path):
    """Load preprocessed CSV"""
    print(f"\n📂 Loading data...")
    print("-"*70)
    
    df = pd.read_csv(file_path)
    # Parse dates with dayfirst=True to handle DD-MM-YYYY format
    df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=True)
    
    print(f"   ✅ Loaded {len(df):,} rows")
    print(f"   📍 Districts: {df['district'].nunique()}")
    print(f"   📅 Range: {df['date'].min()} to {df['date'].max()}")
    print(f"   📊 Columns: {list(df.columns)}")
    
    return df


# ============================================
# STEP 2: PREPARE DATA PER DISTRICT
# ============================================

def prepare_district_data(df, district_name, input_features, target_feature, 
                          lookback, forecast_horizon):
    """
    Prepare sequences for one district
    """
    print(f"\n🔧 Preparing data for: {district_name}")
    print("-"*70)
    
    # Filter district
    dist_df = df[df['district'] == district_name].copy()
    dist_df = dist_df.sort_values('date').reset_index(drop=True)
    
    if len(dist_df) < lookback + forecast_horizon + 100:
        print(f"   ⚠️  Insufficient data: {len(dist_df)} days (need {lookback + forecast_horizon + 100}+)")
        return None, None, None, None, None, None, None
    
    print(f"   📦 Data points: {len(dist_df)}")
    print(f"   📅 Range: {dist_df['date'].min()} to {dist_df['date'].max()}")
    
    # Check for missing values in features
    for col in input_features:
        if col not in dist_df.columns:
            print(f"   ❌ Missing column: {col}")
            return None, None, None, None, None, None, None
        
        missing = dist_df[col].isna().sum()
        if missing > 0:
            print(f"   ⚠️  {col}: {missing} missing values (filling with forward fill)")
            dist_df[col] = dist_df[col].fillna(method='ffill').fillna(method='bfill')
    
    # Extract features
    X_data = dist_df[input_features].values
    y_data = dist_df[target_feature].values
    
    # Create sequences
    X, y = [], []
    for i in range(lookback, len(dist_df) - forecast_horizon + 1):
        X.append(X_data[i-lookback:i, :])
        y.append(y_data[i:i+forecast_horizon])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"   ✅ Created {len(X):,} sequences")
    print(f"   📦 X shape: {X.shape} (samples, lookback={lookback}, features={len(input_features)})")
    print(f"   📦 y shape: {y.shape} (samples, forecast_horizon={forecast_horizon})")
    
    # Train-test split (temporal - no shuffle!)
    split_idx = int(len(X) * (1 - TEST_SPLIT))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"   📊 Train: {len(X_train)} | Test: {len(X_test)}")
    
    # Normalize
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    
    # Fit on train only
    n_samples_train, n_timesteps, n_features = X_train.shape
    X_train_reshaped = X_train.reshape(-1, n_features)
    X_train_scaled = scaler_X.fit_transform(X_train_reshaped)
    X_train_scaled = X_train_scaled.reshape(n_samples_train, n_timesteps, n_features)
    
    # Transform test
    n_samples_test = X_test.shape[0]
    X_test_reshaped = X_test.reshape(-1, n_features)
    X_test_scaled = scaler_X.transform(X_test_reshaped)
    X_test_scaled = X_test_scaled.reshape(n_samples_test, n_timesteps, n_features)
    
    # Scale y
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    
    print(f"   ✅ Normalization complete")
    
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y, dist_df


# ============================================
# STEP 3: BUILD LSTM MODEL
# ============================================

def build_lstm(input_shape, forecast_horizon):
    """
    Build LSTM architecture
    """
    print(f"\n🏗️  Building LSTM model...")
    print("-"*70)
    
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        
        Dense(16, activation='relu'),
        Dense(forecast_horizon)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    print(model.summary())
    
    return model


# ============================================
# STEP 4: TRAIN MODEL
# ============================================

def train_model(model, X_train, y_train, X_test, y_test, district_name, 
                epochs=100, batch_size=32):
    """
    Train LSTM with callbacks
    """
    print(f"\n🔥 Training LSTM for {district_name}...")
    print("-"*70)
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            f'models/lstm_{district_name}.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=0.00001,
            verbose=1
        )
    ]
    
    # Create models directory
    import os
    os.makedirs('models', exist_ok=True)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"\n   ✅ Training complete!")
    print(f"   Final train loss: {history.history['loss'][-1]:.6f}")
    print(f"   Final val loss: {history.history['val_loss'][-1]:.6f}")
    
    return history


# ============================================
# STEP 5: EVALUATE MODEL
# ============================================

def evaluate_model(model, X_test, y_test, scaler_y, district_name):
    """
    Calculate metrics on test set
    """
    print(f"\n📈 Evaluating {district_name}...")
    print("-"*70)
    
    # Predict
    y_pred_scaled = model.predict(X_test, verbose=0)
    
    # Inverse transform
    y_test_actual = scaler_y.inverse_transform(y_test)
    y_pred_actual = scaler_y.inverse_transform(y_pred_scaled)
    
    # Overall metrics
    rmse = np.sqrt(mean_squared_error(y_test_actual.flatten(), y_pred_actual.flatten()))
    mae = mean_absolute_error(y_test_actual.flatten(), y_pred_actual.flatten())
    r2 = r2_score(y_test_actual.flatten(), y_pred_actual.flatten())
    
    # Per-day RMSE
    per_day_rmse = []
    for day in range(y_test_actual.shape[1]):
        day_rmse = np.sqrt(mean_squared_error(y_test_actual[:, day], y_pred_actual[:, day]))
        per_day_rmse.append(day_rmse)
    
    print(f"\n   📊 Overall Metrics:")
    print(f"      RMSE: {rmse:.3f}m")
    print(f"      MAE: {mae:.3f}m")
    print(f"      R²: {r2:.3f}")
    
    print(f"\n   📊 Per-Day RMSE:")
    for i, day_rmse in enumerate(per_day_rmse, 1):
        print(f"      Day {i}: {day_rmse:.3f}m")
    
    # Save metrics
    metrics = {
        'district': district_name,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'per_day_rmse': per_day_rmse
    }
    
    return metrics, y_test_actual, y_pred_actual


# ============================================
# STEP 6: GENERATE FUTURE FORECAST
# ============================================

def forecast_future(model, dist_df, input_features, scaler_X, scaler_y, 
                    lookback, forecast_horizon):
    """
    Generate future forecast
    """
    print(f"\n🔮 Generating {forecast_horizon}-day forecast...")
    print("-"*70)
    
    # Get last lookback days
    last_sequence = dist_df[input_features].values[-lookback:]
    
    # Normalize
    last_sequence_scaled = scaler_X.transform(last_sequence)
    last_sequence_scaled = last_sequence_scaled.reshape(1, lookback, len(input_features))
    
    # Predict
    forecast_scaled = model.predict(last_sequence_scaled, verbose=0)
    
    # Denormalize
    forecast = scaler_y.inverse_transform(forecast_scaled)[0]
    
    # Create forecast dataframe
    last_date = dist_df['date'].iloc[-1]
    forecast_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=forecast_horizon,
        freq='D'
    )
    
    forecast_df = pd.DataFrame({
        'date': forecast_dates,
        'forecast_gwl': forecast
    })
    
    current_gwl = dist_df[TARGET_FEATURE].iloc[-1]
    trend = forecast[-1] - current_gwl
    
    print(f"   ✅ Forecast complete!")
    print(f"      Current GWL: {current_gwl:.2f}m")
    print(f"      Day 1 forecast: {forecast[0]:.2f}m")
    print(f"      Day {forecast_horizon} forecast: {forecast[-1]:.2f}m")
    if abs(trend) > 0.5:
        direction = "rising ⬆️" if trend > 0 else "declining ⬇️"
        print(f"      ⚠️  Trend: {direction} by {abs(trend):.2f}m")
    
    return forecast_df


# ============================================
# STEP 7: VISUALIZATIONS
# ============================================

def plot_results(history, district_name, dist_df, forecast_df, 
                y_test_actual, y_pred_actual, target_feature):
    """
    Create visualizations
    """
    print(f"\n📊 Creating plots...")
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # Plot 1: Training history
    axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_title(f'Training History - {district_name}', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Test predictions
    sample_size = min(50, len(y_test_actual))
    axes[1].plot(range(sample_size), y_test_actual[-sample_size:, 0], 
                label='Actual', color='blue', linewidth=2, marker='o', markersize=4)
    axes[1].plot(range(sample_size), y_pred_actual[-sample_size:, 0], 
                label='Predicted', color='red', linewidth=2, marker='x', markersize=4)
    axes[1].set_title('Test Set: Actual vs Predicted (Day 1)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Sample Index')
    axes[1].set_ylabel('GWL (m)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Future forecast
    lookback_window = 180
    axes[2].plot(dist_df['date'].iloc[-lookback_window:], 
                dist_df[target_feature].iloc[-lookback_window:],
                label='Historical', color='blue', linewidth=2)
    axes[2].plot(forecast_df['date'], forecast_df['forecast_gwl'],
                label='Forecast', color='red', linewidth=2, 
                linestyle='--', marker='o', markersize=6)
    axes[2].set_title(f'7-Day Forecast - {district_name}', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Date')
    axes[2].set_ylabel('GWL (m)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(f'results/lstm_{district_name}_results.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: results/lstm_{district_name}_results.png")
    plt.close()


# ============================================
# MAIN: TRAIN ALL DISTRICTS
# ============================================

def main():
    print("\n" + "🌊"*35)
    print("   LSTM TRAINING - ALL DISTRICTS")
    print("🌊"*35 + "\n")
    
    # Load data
    data_file = input("Enter path to preprocessed CSV: ").strip()
    df = load_data(data_file)
    
    # Get districts
    districts = df['district'].unique()
    print(f"\n📍 Found {len(districts)} districts")
    
    # Optional: Filter districts by minimum data
    min_days = int(input("\nMinimum days required per district (recommend 730): ").strip() or "730")
    
    valid_districts = []
    for district in districts:
        district_days = len(df[df['district'] == district])
        if district_days >= min_days:
            valid_districts.append(district)
    
    print(f"   ✅ {len(valid_districts)} districts have ≥{min_days} days")
    print(f"   Districts to train: {valid_districts}")
    
    # Create output directories
    import os
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('forecasts', exist_ok=True)
    
    # Train each district
    all_metrics = []
    
    for i, district in enumerate(valid_districts, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{len(valid_districts)}] TRAINING: {district}")
        print(f"{'='*70}")
        
        try:
            # Prepare data
            X_train, X_test, y_train, y_test, scaler_X, scaler_y, dist_df = \
                prepare_district_data(df, district, INPUT_FEATURES, TARGET_FEATURE, 
                                     LOOKBACK_DAYS, FORECAST_HORIZON)
            
            if X_train is None:
                print(f"   ⚠️  Skipping {district}")
                continue
            
            # Build model
            model = build_lstm(
                input_shape=(LOOKBACK_DAYS, len(INPUT_FEATURES)),
                forecast_horizon=FORECAST_HORIZON
            )
            
            # Train
            history = train_model(model, X_train, y_train, X_test, y_test, 
                                 district, epochs=100, batch_size=32)
            
            # Evaluate
            metrics, y_test_actual, y_pred_actual = \
                evaluate_model(model, X_test, y_test, scaler_y, district)
            all_metrics.append(metrics)
            
            # Forecast
            forecast_df = forecast_future(model, dist_df, INPUT_FEATURES, 
                                         scaler_X, scaler_y, 
                                         LOOKBACK_DAYS, FORECAST_HORIZON)
            
            # Save forecast
            forecast_df.to_csv(f'forecasts/{district}_forecast.csv', index=False)
            
            # Plot
            plot_results(history, district, dist_df, forecast_df, 
                        y_test_actual, y_pred_actual, TARGET_FEATURE)
            
        except Exception as e:
            print(f"   ❌ Error training {district}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save summary
    if all_metrics:
        summary_df = pd.DataFrame(all_metrics)
        summary_df = summary_df.sort_values('rmse')
        summary_df.to_csv('results/training_summary.csv', index=False)
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE!")
        print("="*70)
        print(f"\nSuccessfully trained: {len(all_metrics)} districts")
        print(f"\n📊 Top 5 Best Models (by RMSE):")
        print(summary_df[['district', 'rmse', 'mae', 'r2']].head().to_string(index=False))
        
        print(f"\n📊 Overall Statistics:")
        print(f"   Mean RMSE: {summary_df['rmse'].mean():.3f}m")
        print(f"   Mean MAE: {summary_df['mae'].mean():.3f}m")
        print(f"   Mean R²: {summary_df['r2'].mean():.3f}")
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
