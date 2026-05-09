"""
COMPLETE GROUNDWATER RECHARGE ESTIMATION MODEL - TRAINING PIPELINE
Aquifer-Specific ANN for Daily Groundwater Data
Outputs: PKL files ready for deployment
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. AQUIFER CLASSIFIER
# ============================================

class AquiferClassifier:
    """
    Classify stations into aquifer types for ALL Indian states
    """
    
    AQUIFER_PROPERTIES = {
        'alluvial': {
            'specific_yield': 0.20,
            'description': 'Indo-Gangetic Plains - High porosity, high permeability',
            'states': ['Punjab', 'Haryana', 'Uttar Pradesh', 'Bihar', 
                      'Uttarakhand', 'Delhi', 'Chandigarh', 'Himachal Pradesh',
                      'Jammu and Kashmir', 'Ladakh', 'Assam', 'Tripura', 
                      'Manipur', 'Meghalaya', 'Nagaland', 'Arunachal Pradesh',
                      'Mizoram', 'Sikkim']
        },
        'hard_rock': {
            'specific_yield': 0.02,
            'description': 'Peninsular India - Crystalline rocks, low porosity',
            'states': ['Maharashtra', 'Karnataka', 'Telangana', 'Andhra Pradesh',
                      'Tamil Nadu', 'Madhya Pradesh', 'Chhattisgarh', 'Odisha',
                      'Jharkhand', 'Rajasthan', 'Gujarat', 'Goa', 'Kerala']
        },
        'coastal': {
            'specific_yield': 0.15,
            'description': 'Coastal aquifers - Mixed sediments, saline intrusion risk',
            'coastal_states': ['Gujarat', 'Maharashtra', 'Goa', 'Karnataka', 
                             'Kerala', 'Tamil Nadu', 'Andhra Pradesh', 
                             'Odisha', 'West Bengal'],
            'coastal_districts': {
                'Gujarat': ['Kachchh', 'Jamnagar', 'Porbandar', 'Junagadh', 'Amreli', 
                           'Bhavnagar', 'Ahmedabad', 'Bharuch', 'Surat', 'Navsari', 
                           'Valsad', 'Dang'],
                'Maharashtra': ['Thane', 'Mumbai Suburban', 'Mumbai', 'Raigad', 
                              'Ratnagiri', 'Sindhudurg'],
                'Goa': ['North Goa', 'South Goa'],
                'Karnataka': ['Uttara Kannada', 'Udupi', 'Dakshina Kannada'],
                'Kerala': ['Kasaragod', 'Kannur', 'Kozhikode', 'Malappuram', 
                          'Thrissur', 'Ernakulam', 'Alappuzha', 'Kollam', 
                          'Thiruvananthapuram'],
                'Tamil Nadu': ['Thiruvallur', 'Chennai', 'Kanchipuram', 'Villupuram',
                             'Cuddalore', 'Nagapattinam', 'Thiruvarur', 'Thanjavur',
                             'Pudukkottai', 'Ramanathapuram', 'Tuticorin', 
                             'Tirunelveli', 'Kanyakumari'],
                'Andhra Pradesh': ['Srikakulam', 'Vizianagaram', 'Visakhapatnam', 
                                 'East Godavari', 'West Godavari', 'Krishna', 
                                 'Guntur', 'Prakasam', 'Nellore'],
                'Odisha': ['Balasore', 'Bhadrak', 'Kendrapara', 'Jagatsinghpur', 
                          'Puri', 'Ganjam'],
                'West Bengal': ['South 24 Parganas', 'North 24 Parganas', 
                               'Purba Medinipur', 'Howrah', 'Hooghly']
            }
        },
        'mixed': {
            'specific_yield': 0.10,
            'description': 'Transitional zones with multiple aquifer types',
            'states': []
        }
    }
    
    @staticmethod
    def classify(state, district):
        """Determine aquifer type from location"""
        coastal_props = AquiferClassifier.AQUIFER_PROPERTIES['coastal']
        
        # Check if district is coastal
        if state in coastal_props['coastal_districts']:
            if district in coastal_props['coastal_districts'][state]:
                return 'coastal'
        
        # Check alluvial states
        if state in AquiferClassifier.AQUIFER_PROPERTIES['alluvial']['states']:
            return 'alluvial'
        
        # Check hard rock states
        if state in AquiferClassifier.AQUIFER_PROPERTIES['hard_rock']['states']:
            return 'hard_rock'
        
        return 'mixed'
    
    @staticmethod
    def get_specific_yield(aquifer_type):
        """Get specific yield for aquifer type"""
        return AquiferClassifier.AQUIFER_PROPERTIES.get(
            aquifer_type, {}
        ).get('specific_yield', 0.15)

# ============================================
# 2. FEATURE ENGINEERING
# ============================================

class DailyRechargeFeatureEngineer:
    """Enhanced feature engineering for daily groundwater data"""
    
    @staticmethod
    def create_lag_features(df, column='rainfall', lag_days=[7, 14, 30, 60, 90]):
        """Create lag features for daily data"""
        for lag in lag_days:
            df[f'{column}_lag_{lag}d'] = df[column].shift(lag)
        return df
    
    @staticmethod
    def create_rolling_features(df):
        """Create rolling window features"""
        # Short-term
        df['rainfall_14day_ma'] = df['rainfall'].rolling(window=14).mean()
        
        # Medium-term
        df['rainfall_30day_ma'] = df['rainfall'].rolling(window=30).mean()
        df['rainfall_60day_ma'] = df['rainfall'].rolling(window=60).mean()
        
        # Long-term
        df['rainfall_90day_sum'] = df['rainfall'].rolling(window=90).sum()
        
        # ET rolling averages
        df['et_30day_ma'] = df['et'].rolling(window=30).mean()
        
        # Effective recharge cumulative
        df['effective_recharge_30day_sum'] = df['effective_recharge'].rolling(window=30).sum()
        df['effective_recharge_60day_sum'] = df['effective_recharge'].rolling(window=60).sum()
        
        return df
    
    @staticmethod
    def create_gwl_trend_features(df):
        """Create GWL trend indicators"""
        # Rate of change over different windows
        df['gwl_change_7d'] = df['gwl'].diff(7)
        df['gwl_change_30d'] = df['gwl'].diff(30)
        
        # Cumulative change
        df['gwl_trend_30d'] = df['gwl_change'].rolling(window=30).sum()
        
        return df
    
    @staticmethod
    def create_seasonal_features(df):
        """Enhanced seasonal encoding"""
        # Cyclical encoding
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        # Monsoon indicator
        df['is_monsoon'] = df['month'].isin([6, 7, 8, 9]).astype(int)
        
        return df
    
    @staticmethod
    def create_interaction_features(df):
        """Create interaction features"""
        # Rainfall intensity during monsoon
        df['rainfall_monsoon_interaction'] = df['rainfall'] * df['is_monsoon']
        
        # ET efficiency ratio
        df['et_rainfall_ratio'] = df['et'] / (df['rainfall'] + 1)
        
        # GWL response to rainfall
        df['gwl_rainfall_interaction'] = df['gwl_change'] * df['rainfall_7day_ma']
        
        return df
    
    @staticmethod
    def engineer_all_features(df):
        """Apply all feature engineering steps"""
        print("Creating lag features...")
        df = DailyRechargeFeatureEngineer.create_lag_features(df, 'rainfall')
        df = DailyRechargeFeatureEngineer.create_lag_features(df, 'effective_recharge')
        
        print("Creating rolling features...")
        df = DailyRechargeFeatureEngineer.create_rolling_features(df)
        
        print("Creating GWL trend features...")
        df = DailyRechargeFeatureEngineer.create_gwl_trend_features(df)
        
        print("Creating seasonal features...")
        df = DailyRechargeFeatureEngineer.create_seasonal_features(df)
        
        print("Creating interaction features...")
        df = DailyRechargeFeatureEngineer.create_interaction_features(df)
        
        # Remove rows with NaN
        initial_len = len(df)
        df = df.dropna()
        print(f"Removed {initial_len - len(df)} rows with NaN values")
        
        return df

# ============================================
# 3. MODEL ARCHITECTURE
# ============================================

class DailyRechargeModelBuilder:
    """Build ANN for daily groundwater data"""
    
    CONFIGS = {
        'alluvial': {
            'hidden_units': [32, 24, 16],
            'dropout': 0.2,
            'learning_rate': 0.0008,
            'activation': 'relu',
            'batch_size': 64
        },
        'hard_rock': {
            'hidden_units': [24, 16, 8],
            'dropout': 0.25,
            'learning_rate': 0.0005,
            'activation': 'relu',
            'batch_size': 32
        },
        'coastal': {
            'hidden_units': [28, 20, 12],
            'dropout': 0.22,
            'learning_rate': 0.0008,
            'activation': 'tanh',
            'batch_size': 64
        }
    }
    
    @staticmethod
    def build(aquifer_type, input_dim):
        """Build aquifer-specific ANN"""
        config = DailyRechargeModelBuilder.CONFIGS.get(
            aquifer_type,
            DailyRechargeModelBuilder.CONFIGS['alluvial']
        )
        
        model = Sequential(name=f'Daily_Recharge_ANN_{aquifer_type}')
        
        # Input layer
        model.add(Input(shape=(input_dim,), name='input_features'))
        
        # Hidden layers with batch normalization
        for i, units in enumerate(config['hidden_units']):
            model.add(Dense(
                units,
                activation=config['activation'],
                kernel_regularizer=l2(0.001),
                name=f'hidden_{i+1}'
            ))
            model.add(BatchNormalization(name=f'bn_{i+1}'))
            model.add(Dropout(config['dropout'], name=f'dropout_{i+1}'))
        
        # Output layer
        model.add(Dense(1, activation='linear', name='gwl_output'))
        
        # Compile
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate']),
            loss='mean_squared_error',
            metrics=['mae', 'mse']
        )
        
        return model, config

# ============================================
# 4. TRAINING PIPELINE
# ============================================

class DailyRechargeModelTrainer:
    """Training pipeline for daily groundwater recharge models"""
    
    def __init__(self, data, feature_cols, target_col='gwl'):
        self.data = data
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.models = {}
        self.scalers = {}
        self.metrics = {}
    
    def train_all_aquifer_models(self, output_dir='models'):
        """Train separate models for each aquifer type"""
        os.makedirs(output_dir, exist_ok=True)
        
        for aquifer_type in ['alluvial', 'hard_rock', 'coastal']:
            print(f"\n{'='*70}")
            print(f"TRAINING MODEL FOR: {aquifer_type.upper()}")
            print(f"{'='*70}")
            
            subset = self.data[self.data['aquifer_type'] == aquifer_type].copy()
            
            if len(subset) < 365:
                print(f"⚠️ Insufficient data for {aquifer_type}: {len(subset)} samples")
                print(f"   Minimum required: 365 samples (1 year). Skipping...")
                continue
            
            print(f"✓ Dataset size: {len(subset)} samples ({len(subset)/365:.1f} years)")
            
            model, scaler, metrics = self._train_single_model(subset, aquifer_type)
            
            self.models[aquifer_type] = model
            self.scalers[aquifer_type] = scaler
            self.metrics[aquifer_type] = metrics
            
            self._save_model_pkl(model, scaler, metrics, aquifer_type, output_dir)
        
        self._save_complete_package(output_dir)
        
        return self.models, self.scalers, self.metrics
    
    def _train_single_model(self, subset, aquifer_type):
        """Train model for single aquifer type"""
        X = subset[self.feature_cols].copy()
        y = subset[self.target_col].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        print(f"✓ Train samples: {len(X_train)} ({len(X_train)/365:.1f} years)")
        print(f"✓ Test samples: {len(X_test)} ({len(X_test)/365:.1f} years)")
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model, config = DailyRechargeModelBuilder.build(aquifer_type, len(self.feature_cols))
        
        print(f"\nModel Architecture:")
        print(f"  Hidden layers: {config['hidden_units']}")
        print(f"  Dropout: {config['dropout']}")
        print(f"  Learning rate: {config['learning_rate']}")
        print(f"  Batch size: {config['batch_size']}")
        
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=25,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        )
        
        print(f"\nTraining started...")
        history = model.fit(
            X_train_scaled, y_train,
            validation_data=(X_test_scaled, y_test),
            epochs=150,
            batch_size=config['batch_size'],
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        metrics = self._evaluate_model(model, scaler, X_test, y_test, aquifer_type)
        
        metrics['history'] = {
            'loss': history.history['loss'],
            'val_loss': history.history['val_loss']
        }
        
        return model, scaler, metrics
    
    def _evaluate_model(self, model, scaler, X_test, y_test, aquifer_type):
        """Comprehensive model evaluation"""
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled, verbose=0).flatten()
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        nse = 1 - (np.sum((y_test - y_pred)**2) / np.sum((y_test - y_test.mean())**2))
        
        if nse > 0.75:
            rating = "EXCELLENT"
        elif nse > 0.65:
            rating = "GOOD"
        elif nse > 0.50:
            rating = "SATISFACTORY"
        else:
            rating = "UNSATISFACTORY"
        
        print(f"\n{'='*70}")
        print(f"VALIDATION RESULTS - {aquifer_type.upper()}")
        print(f"{'='*70}")
        print(f"R² Score: {r2:.4f}")
        print(f"Mean Absolute Error: {mae:.3f} m")
        print(f"Root Mean Square Error: {rmse:.3f} m")
        print(f"Nash-Sutcliffe Efficiency: {nse:.4f}")
        print(f"Performance Rating: {rating}")
        print(f"{'='*70}")
        
        return {'r2': r2, 'mae': mae, 'rmse': rmse, 'nse': nse, 'rating': rating}
    
    def _save_model_pkl(self, model, scaler, metrics, aquifer_type, output_dir):
        """Save model as pickle file"""
        model_package = {
            'model': model,
            'scaler': scaler,
            'metrics': metrics,
            'aquifer_type': aquifer_type,
            'feature_cols': self.feature_cols,
            'target_col': self.target_col,
            'specific_yield': AquiferClassifier.get_specific_yield(aquifer_type),
            'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_version': '1.0'
        }
        
        pkl_path = os.path.join(output_dir, f'gw_recharge_{aquifer_type}.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(model_package, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"✓ Model saved as PKL: {pkl_path}")
    
    def _save_complete_package(self, output_dir):
        """Save complete package with all models"""
        complete_package = {
            'models': self.models,
            'scalers': self.scalers,
            'metrics': self.metrics,
            'feature_cols': self.feature_cols,
            'target_col': self.target_col,
            'aquifer_classifier': AquiferClassifier,
            'training_summary': {
                'total_models': len(self.models),
                'aquifer_types': list(self.models.keys()),
                'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model_version': '1.0'
            }
        }
        
        complete_path = os.path.join(output_dir, 'gw_recharge_complete_package.pkl')
        with open(complete_path, 'wb') as f:
            pickle.dump(complete_package, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"\n{'='*70}")
        print(f"✓ Complete package saved: {complete_path}")
        print(f"  Contains all {len(self.models)} trained models")
        print(f"{'='*70}")

# ============================================
# 5. DATA LOADING & PREPROCESSING
# ============================================

def load_and_prepare_data(csv_path):
    """
    Load CSV data and prepare for training
    Expected columns: date, state, district, gwl, rainfall, et, 
                     effective_recharge, rainfall_7day_ma, gwl_change, 
                     gwl_7day_ma, day_of_year, month, year
    """
    print(f"\n{'='*70}")
    print("LOADING DATA")
    print(f"{'='*70}")
    print(f"Reading: {csv_path}")
    
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"✓ Loaded {len(df)} records")
    print(f"✓ Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"✓ States: {df['state'].nunique()}")
    print(f"✓ Districts: {df['district'].nunique()}")
    
    # Classify aquifer types
    print("\nClassifying aquifer types...")
    df['aquifer_type'] = df.apply(
        lambda row: AquiferClassifier.classify(row['state'], row['district']),
        axis=1
    )
    
    print("\nAquifer distribution:")
    print(df['aquifer_type'].value_counts())
    
    return df

# ============================================
# 6. MAIN TRAINING FUNCTION
# ============================================

def train_with_your_data(csv_path, output_dir='models'):
    """
    Complete training pipeline
    
    Args:
        csv_path: Path to your CSV file
        output_dir: Directory to save models
        
    Returns:
        models, scalers, metrics, processed_data
    """
    print(f"\n{'='*70}")
    print("GROUNDWATER RECHARGE MODEL TRAINING PIPELINE")
    print(f"{'='*70}")
    
    # Load data
    df = load_and_prepare_data(csv_path)
    
    # Feature engineering
    print(f"\n{'='*70}")
    print("FEATURE ENGINEERING")
    print(f"{'='*70}")
    df = DailyRechargeFeatureEngineer.engineer_all_features(df)
    
    # Define feature columns
    feature_cols = [
        'rainfall', 'et', 'effective_recharge',
        'rainfall_7day_ma', 'gwl_change', 'gwl_7day_ma',
        'month', 'day_of_year',
        'rainfall_lag_7d', 'rainfall_lag_14d', 'rainfall_lag_30d', 
        'rainfall_lag_60d', 'rainfall_lag_90d',
        'effective_recharge_lag_7d', 'effective_recharge_lag_14d', 
        'effective_recharge_lag_30d', 'effective_recharge_lag_60d', 
        'effective_recharge_lag_90d',
        'rainfall_14day_ma', 'rainfall_30day_ma', 'rainfall_60day_ma', 
        'rainfall_90day_sum',
        'et_30day_ma',
        'effective_recharge_30day_sum', 'effective_recharge_60day_sum',
        'gwl_change_7d', 'gwl_change_30d', 'gwl_trend_30d',
        'month_sin', 'month_cos', 'day_sin', 'day_cos', 'is_monsoon',
        'rainfall_monsoon_interaction', 'et_rainfall_ratio', 
        'gwl_rainfall_interaction'
    ]
    
    print(f"✓ Total features: {len(feature_cols)}")
    print(f"✓ Final dataset: {len(df)} records")
    
    # Train models
    print(f"\n{'='*70}")
    print("TRAINING AQUIFER-SPECIFIC MODELS")
    print(f"{'='*70}")
    
    trainer = DailyRechargeModelTrainer(df, feature_cols, target_col='gwl')
    models, scalers, metrics = trainer.train_all_aquifer_models(output_dir)
    
    # Summary
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE - SUMMARY")
    print(f"{'='*70}")
    for aquifer_type, metric in metrics.items():
        print(f"\n{aquifer_type.upper()}:")
        print(f"  R²: {metric['r2']:.4f}")
        print(f"  RMSE: {metric['rmse']:.3f} m")
        print(f"  NSE: {metric['nse']:.4f}")
        print(f"  Rating: {metric['rating']}")
    
    print(f"\n{'='*70}")
    print("MODELS SAVED SUCCESSFULLY")
    print(f"{'='*70}")
    print(f"Location: {output_dir}/")
    print("Files created:")
    print("  - gw_recharge_alluvial.pkl")
    print("  - gw_recharge_hard_rock.pkl")
    print("  - gw_recharge_coastal.pkl")
    print("  - gw_recharge_complete_package.pkl")
    
    return models, scalers, metrics, df

# ============================================
# 7. RECHARGE CALCULATOR
# ============================================

class RechargeCalculator:
    """
    Calculate actual recharge rates from model predictions
    Using Water Table Fluctuation (WTF) Method
    """
    
    @staticmethod
    def calculate_daily(predicted_gwl, actual_gwl, aquifer_type, time_period_days=1):
        """
        Calculate recharge rate using WTF method
        
        Formula: R = Sy × (Δh / Δt) × 1000
        where:
        - R = Recharge rate (mm/period)
        - Sy = Specific yield (aquifer-dependent)
        - Δh = Change in water level (m)
        - Δt = Time period (days)
        """
        sy = AquiferClassifier.get_specific_yield(aquifer_type)
        
        # Calculate water level change (positive if level rose)
        delta_h = actual_gwl - predicted_gwl
        
        # Calculate recharge (convert m to mm)
        recharge_mm_per_day = sy * (delta_h / time_period_days) * 1000
        recharge_mm_per_month = recharge_mm_per_day * 30  # Approximate monthly
        
        # Classify intensity
        if recharge_mm_per_day > 50:
            category = "High Recharge"
            color = "#0066CC"
        elif recharge_mm_per_day > 20:
            category = "Moderate Recharge"
            color = "#66B2FF"
        elif recharge_mm_per_day > 5:
            category = "Low Recharge"
            color = "#FFD700"
        elif recharge_mm_per_day > 0:
            category = "Minimal Recharge"
            color = "#FFA500"
        else:
            category = "Depletion (Extraction > Recharge)"
            color = "#FF4444"
        
        return {
            'recharge_mm_per_day': recharge_mm_per_day,
            'recharge_mm_per_month': recharge_mm_per_month,
            'category': category,
            'color': color,
            'specific_yield': sy,
            'delta_h_m': delta_h,
            'predicted_gwl': predicted_gwl,
            'actual_gwl': actual_gwl
        }

# ============================================
# 8. MODEL LOADER FOR DEPLOYMENT
# ============================================

class ModelLoader:
    """Load trained models for prediction"""
    
    @staticmethod
    def load_model(aquifer_type, models_dir='models'):
        """Load a single aquifer model"""
        pkl_path = os.path.join(models_dir, f'gw_recharge_{aquifer_type}.pkl')
        
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Model not found: {pkl_path}")
        
        with open(pkl_path, 'rb') as f:
            package = pickle.load(f)
        
        print(f"✓ Loaded {aquifer_type} model")
        print(f"  Trained: {package['training_date']}")
        print(f"  NSE: {package['metrics']['nse']:.4f}")
        
        return package
    
    @staticmethod
    def load_all_models(models_dir='models'):
        """Load all aquifer models"""
        complete_path = os.path.join(models_dir, 'gw_recharge_complete_package.pkl')
        
        if os.path.exists(complete_path):
            with open(complete_path, 'rb') as f:
                package = pickle.load(f)
            print(f"✓ Loaded complete package with {package['training_summary']['total_models']} models")
            return package
        else:
            # Load individual models
            models = {}
            for aquifer_type in ['alluvial', 'hard_rock', 'coastal']:
                try:
                    models[aquifer_type] = ModelLoader.load_model(aquifer_type, models_dir)
                except FileNotFoundError:
                    print(f"⚠️ {aquifer_type} model not found")
            return models
    
    @staticmethod
    def predict(model_package, features_df):
        """Make prediction using loaded model"""
        model = model_package['model']
        scaler = model_package['scaler']
        feature_cols = model_package['feature_cols']
        
        # Ensure features are in correct order
        X = features_df[feature_cols].values
        X_scaled = scaler.transform(X)
        
        predictions = model.predict(X_scaled, verbose=0).flatten()
        
        return predictions

# ============================================
# 9. EXAMPLE WORKFLOW WITH SAMPLE DATA
# ============================================

def create_sample_data(n_days=730):
    """
    Create sample data for testing (2 years of daily data)
    In production, replace with actual India-WRIS data
    """
    print("\n" + "="*70)
    print("CREATING SAMPLE DATA FOR DEMONSTRATION")
    print("="*70)
    print(f"Generating {n_days} days of synthetic data...")
    
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', periods=n_days, freq='D')
    
    # Create data for multiple stations across different aquifer types
    stations = [
        {'id': 'DWLR_001', 'state': 'Punjab', 'district': 'Ludhiana'},
        {'id': 'DWLR_002', 'state': 'Punjab', 'district': 'Amritsar'},
        {'id': 'DWLR_003', 'state': 'Maharashtra', 'district': 'Pune'},
        {'id': 'DWLR_004', 'state': 'Maharashtra', 'district': 'Nagpur'},
        {'id': 'DWLR_005', 'state': 'Kerala', 'district': 'Thiruvananthapuram'},
        {'id': 'DWLR_006', 'state': 'Tamil Nadu', 'district': 'Chennai'},
    ]
    
    all_data = []
    
    for station in stations:
        for date in dates:
            month = date.month
            day_of_year = date.dayofyear
            
            # Simulate monsoon patterns (June-September)
            is_monsoon = 1 if month in [6, 7, 8, 9] else 0
            
            # Simulate rainfall with seasonal patterns
            if is_monsoon:
                rainfall = np.random.gamma(3, 20)  # Higher rainfall during monsoon
            else:
                rainfall = np.random.gamma(1, 5)   # Lower rainfall otherwise
            
            # Simulate ET (higher in summer)
            et = 3 + 2 * np.sin(2 * np.pi * day_of_year / 365) + np.random.normal(0, 0.5)
            et = max(0, et)
            
            # Calculate effective recharge
            effective_recharge = max(0, rainfall - et)
            
            # Simulate GWL with cumulative effects
            base_gwl = 15.0
            seasonal_effect = 3 * np.sin(2 * np.pi * day_of_year / 365 + np.pi)
            random_noise = np.random.normal(0, 0.3)
            gwl = base_gwl + seasonal_effect + random_noise
            
            all_data.append({
                'date': date,
                'state': station['state'],
                'district': station['district'],
                'station_id': station['id'],
                'gwl': gwl,
                'rainfall': rainfall,
                'et': et,
                'effective_recharge': effective_recharge,
                'month': month,
                'day_of_year': day_of_year,
                'year': date.year
            })
    
    df = pd.DataFrame(all_data)
    
    # Calculate 7-day moving averages and changes
    df = df.sort_values(['station_id', 'date'])
    
    for station_id in df['station_id'].unique():
        mask = df['station_id'] == station_id
        df.loc[mask, 'rainfall_7day_ma'] = df.loc[mask, 'rainfall'].rolling(window=7).mean()
        df.loc[mask, 'gwl_7day_ma'] = df.loc[mask, 'gwl'].rolling(window=7).mean()
        df.loc[mask, 'gwl_change'] = df.loc[mask, 'gwl'].diff()
    
    df = df.dropna()
    
    print(f"✓ Created {len(df)} records")
    print(f"✓ Stations: {df['station_id'].nunique()}")
    print(f"✓ Date range: {df['date'].min()} to {df['date'].max()}")
    
    return df

# ============================================
# 10. MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("GROUNDWATER RECHARGE MODEL - COMPLETE TRAINING PIPELINE")
    print("="*70)
    print("\nOptions:")
    print("1. Train with sample data (demonstration)")
    print("2. Train with your CSV file")
    
    # OPTION 1: Run with sample data for demonstration
    print("\n" + "="*70)
    print("RUNNING DEMONSTRATION WITH SAMPLE DATA")
    print("="*70)
    
    # Create sample data
    df = create_sample_data(n_days=730)  # 2 years of daily data
    
    # Save to CSV for reference
    sample_csv = r"C:\codes\PyTorchTest2\i2final_lstm.csv" 
    df.to_csv(sample_csv, index=False)
    print(f"\n✓ Sample data saved to: {sample_csv}")
    
    # Train models
    print("\n" + "="*70)
    print("STARTING TRAINING PROCESS")
    print("="*70)
    
    models, scalers, metrics, processed_df = train_with_your_data(sample_csv)
    
    # Test prediction on sample data
    print("\n" + "="*70)
    print("TESTING PREDICTIONS")
    print("="*70)
    
    for aquifer_type in models.keys():
        print(f"\n{aquifer_type.upper()} - Sample Prediction:")
        
        # Get sample data for this aquifer
        aquifer_data = processed_df[processed_df['aquifer_type'] == aquifer_type]
        
        if len(aquifer_data) == 0:
            print(f"  No data available for {aquifer_type}")
            continue
        
        # Load model package
        model_package = ModelLoader.load_model(aquifer_type, 'models')
        
        # Get a sample row
        sample_features = aquifer_data[model_package['feature_cols']].iloc[[0]]
        actual_gwl = aquifer_data['gwl'].iloc[0]
        
        # Predict
        predicted_gwl = ModelLoader.predict(model_package, sample_features)[0]
        
        # Calculate recharge
        recharge_result = RechargeCalculator.calculate_daily(
            predicted_gwl, actual_gwl, aquifer_type
        )
        
        print(f"  Predicted GWL: {predicted_gwl:.2f} m")
        print(f"  Actual GWL: {actual_gwl:.2f} m")
        print(f"  Recharge Rate: {recharge_result['recharge_mm_per_day']:.1f} mm/day")
        print(f"  Monthly Recharge: {recharge_result['recharge_mm_per_month']:.1f} mm/month")
        print(f"  Category: {recharge_result['category']}")
    
    # OPTION 2: Instructions for training with your own data
    print("\n" + "="*70)
    print("TO TRAIN WITH YOUR OWN DATA")
    print("="*70)
    print("\n1. Prepare your CSV with these columns:")
    print("   date, state, district, gwl, rainfall, et,")
    print("   effective_recharge, rainfall_7day_ma, gwl_change,")
    print("   gwl_7day_ma, day_of_year, month, year")
    print("\n2. Run:")
    print("   models, scalers, metrics, df = train_with_your_data('your_data.csv')")
    print("\n3. Models will be saved in 'models/' directory as PKL files")
    print("\n4. Load and use models:")
    print("   package = ModelLoader.load_model('alluvial', 'models')")
    print("   predictions = ModelLoader.predict(package, features_df)")
    print("   recharge = RechargeCalculator.calculate_daily(pred, actual, 'alluvial')")
    print("\n" + "="*70)
    print("TRAINING COMPLETE - MODELS READY FOR DEPLOYMENT!")
    print("="*70)