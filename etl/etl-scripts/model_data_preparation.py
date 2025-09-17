#!/usr/bin/env python3
"""
Model Data Preparation Script

This script prepares the enriched passenger data for machine learning modeling.
It performs feature engineering, handles missing values, creates time-based features,
and outputs model-ready datasets for different prediction tasks.

Input: passengers_enriched_2023.csv
Output: Multiple model-ready datasets for different ML tasks
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PROCESSED_DIR = DATA_DIR / "processed"
INPUT_FILE = PROCESSED_DIR / "passengers_enriched_2023.csv"
OUTPUT_DIR = PROCESSED_DIR / "model_ready"

def load_and_explore_data():
    """Load the enriched dataset and perform initial exploration"""
    print("Loading enriched passenger data...")
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Unique stations: {df['station_name'].nunique()}")
    print(f"Missing values per column:")
    print(df.isnull().sum().sort_values(ascending=False).head(10))
    
    return df

def create_time_features(df):
    """Create time-based features for modeling"""
    print("Creating time-based features...")
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # Time features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['quarter'] = df['date'].dt.quarter
    
    # Weekend/weekday
    df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)
    df['is_monday'] = (df['date'].dt.dayofweek == 0).astype(int)
    df['is_friday'] = (df['date'].dt.dayofweek == 4).astype(int)
    
    # Holiday indicators (basic UK holidays)
    df['is_new_year'] = ((df['month'] == 1) & (df['day'] == 1)).astype(int)
    df['is_christmas_period'] = ((df['month'] == 12) & (df['day'].between(24, 31))).astype(int)
    
    # Seasonal features
    df['season'] = df['month'].map({
        12: 'winter', 1: 'winter', 2: 'winter',
        3: 'spring', 4: 'spring', 5: 'spring',
        6: 'summer', 7: 'summer', 8: 'summer',
        9: 'autumn', 10: 'autumn', 11: 'autumn'
    })
    
    return df

def create_lag_features(df, target_cols=['entrytapcount', 'exittapcount'], lags=[1, 7]):
    """Create lag features for time series modeling"""
    print("Creating lag features...")
    
    df = df.copy()
    df = df.sort_values(['station_name', 'line', 'date'])
    
    for col in target_cols:
        for lag in lags:
            df[f'{col}_lag_{lag}'] = df.groupby(['station_name', 'line'])[col].shift(lag)
    
    # Rolling averages
    for col in target_cols:
        df[f'{col}_rolling_7d'] = df.groupby(['station_name', 'line'])[col].rolling(7, min_periods=1).mean().reset_index(0, drop=True)
        df[f'{col}_rolling_30d'] = df.groupby(['station_name', 'line'])[col].rolling(30, min_periods=1).mean().reset_index(0, drop=True)
    
    return df

def create_weather_features(df):
    """Engineer weather-related features"""
    print("Creating weather features...")
    
    df = df.copy()
    
    # Temperature features
    df['temp_range'] = df['max_temp'] - df['min_temp']
    df['is_hot_day'] = (df['max_temp'] > 25).astype(int)
    df['is_cold_day'] = (df['min_temp'] < 5).astype(int)
    
    # Precipitation features
    df['has_rain'] = (df['precipitation_amount'] > 0).astype(int)
    df['heavy_rain'] = (df['precipitation_amount'] > 10).astype(int)
    
    # Weather comfort index (simple)
    df['weather_comfort'] = (
        (df['mean_temp'].between(15, 22)) & 
        (df['precipitation_amount'] < 1) & 
        (df['relative_humidity'] < 80)
    ).astype(int)
    
    return df

def create_event_features(df):
    """Engineer event-related features"""
    print("Creating event features...")
    
    df = df.copy()
    
    # Event indicators
    df['has_event'] = (~df['event_name'].isna()).astype(int)
    df['event_attendance_category'] = pd.cut(
        df['expected_attendance'], 
        bins=[0, 1000, 10000, 50000, np.inf], 
        labels=['none', 'small', 'medium', 'large'],
        include_lowest=True
    )
    
    return df

def create_station_features(df):
    """Engineer station-related features"""
    print("Creating station features...")
    
    df = df.copy()
    
    # Station amenities score
    amenity_cols = ['wifi', 'airportinterchange', 'bluebadgecarparking', 'toilet_isaccessible']
    for col in amenity_cols:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(int)
    
    df['amenities_score'] = df[amenity_cols].sum(axis=1)
    
    # Zone features
    df['is_central_zone'] = (df['farezones'].astype(str).str.contains('1')).astype(int)
    df['is_outer_zone'] = (df['farezones'].astype(str).str.contains('[456]')).astype(int)
    
    return df

def handle_missing_values(df):
    """Handle missing values appropriately for modeling"""
    print("Handling missing values...")
    
    df = df.copy()
    
    # Numeric columns - fill with median or 0
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            if 'count' in col.lower() or 'attendance' in col.lower():
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna(df[col].median())
    
    # Categorical columns - fill with 'unknown' or mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna('unknown')
    
    return df

def encode_categorical_features(df):
    """Encode categorical features for modeling"""
    print("Encoding categorical features...")
    
    df = df.copy()
    
    # Label encode categorical columns
    categorical_cols = ['dayofweek', 'mode', 'line', 'season', 'event_type', 'toilet_type']
    label_encoders = {}
    
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    
    # One-hot encode high cardinality categoricals (optional)
    # For station_name, we might want to use target encoding instead due to high cardinality
    
    return df, label_encoders

def create_target_variables(df):
    """Create different target variables for various modeling tasks"""
    print("Creating target variables...")
    
    df = df.copy()
    
    # Total passenger flow
    df['total_passengers'] = df['entrytapcount'] + df['exittapcount']
    
    # Net flow (entry - exit)
    df['net_flow'] = df['entrytapcount'] - df['exittapcount']
    
    # Passenger flow categories
    df['flow_category'] = pd.cut(
        df['total_passengers'],
        bins=[0, 100, 500, 1000, 5000, np.inf],
        labels=['very_low', 'low', 'medium', 'high', 'very_high']
    )
    
    # Peak hour indicator (if we had hourly data, this would be more meaningful)
    # For now, we'll use day of week as proxy
    df['is_peak_day'] = df['dayofweek'].isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']).astype(int)
    
    return df

def prepare_modeling_datasets(df):
    """Prepare different datasets for various modeling tasks"""
    print("Preparing modeling datasets...")
    
    # Remove rows with missing target values
    df_clean = df.dropna(subset=['entrytapcount', 'exittapcount']).copy()
    
    # Select features for modeling
    feature_cols = [
        # Time features
        'year', 'month', 'day', 'day_of_year', 'week_of_year', 'quarter',
        'is_weekend', 'is_monday', 'is_friday', 'is_new_year', 'is_christmas_period',
        'season_encoded',
        
        # Weather features
        'max_temp', 'min_temp', 'mean_temp', 'temp_range', 'precipitation_amount',
        'relative_humidity', 'cloud_cover', 'sunshine_duration', 'sea_level_pressure',
        'is_hot_day', 'is_cold_day', 'has_rain', 'heavy_rain', 'weather_comfort',
        
        # Station features
        'farezones', 'amenities_score', 'is_central_zone', 'is_outer_zone',
        'mode_encoded', 'line_encoded',
        
        # Event features
        'has_event', 'expected_attendance',
        
        # Performance features (if available)
        'service_operated_allweek_pct', 'service_operated_weekday_pct', 
        'service_operated_weekend_pct', 'kilometres_operated'
    ]
    
    # Filter to existing columns
    available_features = [col for col in feature_cols if col in df_clean.columns]
    
    # Dataset 1: Passenger count prediction (regression)
    regression_features = df_clean[available_features].copy()
    regression_targets = df_clean[['entrytapcount', 'exittapcount', 'total_passengers']].copy()
    
    # Dataset 2: Flow category prediction (classification)
    classification_features = df_clean[available_features].copy()
    classification_target = df_clean['flow_category'].copy()
    
    # Dataset 3: Time series dataset (with lags)
    lag_features = [col for col in df_clean.columns if 'lag_' in col or 'rolling_' in col]
    timeseries_features = df_clean[available_features + lag_features].dropna().copy()
    timeseries_targets = df_clean[['entrytapcount', 'exittapcount']].loc[timeseries_features.index].copy()
    
    return {
        'regression': (regression_features, regression_targets),
        'classification': (classification_features, classification_target),
        'timeseries': (timeseries_features, timeseries_targets)
    }

def save_datasets(datasets, metadata):
    """Save the prepared datasets"""
    print("Saving model-ready datasets...")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    for task_name, (features, targets) in datasets.items():
        # Save features and targets
        features.to_csv(OUTPUT_DIR / f"{task_name}_features.csv", index=False)
        
        if isinstance(targets, pd.DataFrame):
            targets.to_csv(OUTPUT_DIR / f"{task_name}_targets.csv", index=False)
        else:
            targets.to_csv(OUTPUT_DIR / f"{task_name}_target.csv", index=False)
        
        print(f"  {task_name}: {features.shape[0]} samples, {features.shape[1]} features")
    
    # Save metadata
    metadata_df = pd.DataFrame([metadata])
    metadata_df.to_csv(OUTPUT_DIR / "dataset_metadata.csv", index=False)
    
    print(f"\nDatasets saved to: {OUTPUT_DIR}")

def main():
    """Main pipeline for model data preparation"""
    print("=" * 60)
    print("MODEL DATA PREPARATION PIPELINE")
    print("=" * 60)
    
    # Load data
    df = load_and_explore_data()
    
    # Feature engineering
    df = create_time_features(df)
    df = create_weather_features(df)
    df = create_event_features(df)
    df = create_station_features(df)
    df = create_target_variables(df)
    
    # Create lag features (for time series)
    df = create_lag_features(df)
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Encode categorical features
    df, label_encoders = encode_categorical_features(df)
    
    # Prepare datasets for different tasks
    datasets = prepare_modeling_datasets(df)
    
    # Create metadata
    metadata = {
        'creation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'source_file': str(INPUT_FILE),
        'total_samples': len(df),
        'date_range_start': df['date'].min(),
        'date_range_end': df['date'].max(),
        'unique_stations': df['station_name'].nunique(),
        'unique_lines': df['line'].nunique()
    }
    
    # Save datasets
    save_datasets(datasets, metadata)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    # Print summary
    print("\nDataset Summary:")
    for task_name, (features, targets) in datasets.items():
        print(f"  {task_name.upper()}:")
        print(f"    - Features: {features.shape}")
        print(f"    - Targets: {targets.shape}")
        print(f"    - Use case: {get_use_case_description(task_name)}")
    
    return df, datasets, label_encoders

def get_use_case_description(task_name):
    """Get description of use case for each dataset"""
    descriptions = {
        'regression': 'Predict passenger counts (entry/exit/total)',
        'classification': 'Predict passenger flow categories',
        'timeseries': 'Time series forecasting with lag features'
    }
    return descriptions.get(task_name, 'Unknown')

if __name__ == "__main__":
    df, datasets, encoders = main()
