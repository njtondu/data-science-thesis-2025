#!/usr/bin/env python3
"""
Data Preparation for Modeling Script

This script prepares the passengers enriched data for machine learning modeling by:
1. Dropping redundant fields
2. Converting date to numerical and cyclical features
3. Creating weekend boolean field
4. Converting event type to boolean
5. Handling null values in toilet fields
6. Filling null values in service and parking fields with appropriate statistics

Input: passengers_enriched_2023.csv
Output: passengers_enriched_2023_prepared.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PROCESSED_DIR = DATA_DIR / "processed"
INPUT_FILE = PROCESSED_DIR / "passengers_enriched_2023.csv"
OUTPUT_FILE = PROCESSED_DIR / "passengers_enriched_2023_prepared.csv"

def load_data():
    """Load the enriched passenger data"""
    print("Loading enriched passenger data...")
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Column names: {df.columns.tolist()}")
    print(f"Data types summary:")
    print(df.dtypes.value_counts())
    
    return df

def drop_redundant_fields(df):
    """Drop redundant fields as specified"""
    print("\n1. Dropping redundant fields...")
    
    fields_to_drop = [
        'airportinterchange', 
        'hubnaptancode', 
        'station_name', 
        'mode', 
        'event_name', 
        'toilet_type'
    ]

print(f"Fields to drop: {fields_to_drop}")
print(f"Shape before dropping: {df.shape}")

# Drop the fields
    df = df.drop(columns=fields_to_drop, errors='ignore')

print(f"Shape after dropping: {df.shape}")
print(f"Remaining columns: {df.columns.tolist()}")

    return df

def convert_date_features(df):
    """Convert date to numerical and cyclical features"""
    print("\n2. Converting date to numerical and cyclical features...")

    df = df.copy()

# Convert date to datetime
df['date'] = pd.to_datetime(df['date'])

# Extract numerical features from date
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['day_of_year'] = df['date'].dt.dayofyear

# Create cyclical features for month and day of year
# Month cyclical features (12 months)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Day of year cyclical features (365/366 days)
df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

    # Drop the original date column as it's no longer needed
    df = df.drop(columns=['date'])

    print(f"Added cyclical features. New shape: {df.shape}")
    print("New date-related columns:", [col for col in df.columns if any(x in col for x in ['year', 'month', 'day', 'sin', 'cos'])])

    return df

def create_weekend_boolean(df):
    """Create weekend boolean field"""
    print("\n3. Creating weekend boolean field...")

    df = df.copy()

# Create is_weekend boolean field
# Weekend is Saturday (5) and Sunday (6) in pandas weekday (Monday=0)
    # But dayofweek column contains day names, so we need to map them
    weekend_days = ['Saturday', 'Sunday']
    df['is_weekend'] = df['dayofweek'].isin(weekend_days).astype(int)
    
    # Verify the mapping
    weekend_counts = df.groupby(['dayofweek', 'is_weekend']).size().unstack(fill_value=0)
    print("Weekend mapping verification:")
    print(weekend_counts)
    
    return df

def convert_event_type_to_boolean(df):
    """Convert event type to boolean"""
    print("\n4. Converting event type to boolean...")
    
    df = df.copy()
    
    # Create has_event boolean (1 if there's an event, 0 if no event)
    df['has_event'] = (~df['event_type'].isna()).astype(int)
    
    # Check the distribution
    event_distribution = df['has_event'].value_counts()
print("Event distribution:")
    print(f"No event (0): {event_distribution.get(0, 0)}")
    print(f"Has event (1): {event_distribution.get(1, 0)}")
    
    # Drop the original event_type column
    df = df.drop(columns=['event_type'])
    
    return df

def handle_toilet_nulls(df):
    """Handle null values in toilet fields"""
    print("\n5. Handling null values in toilet fields...")
    
    df = df.copy()
    
    # Check null counts before processing
    toilet_cols = ['toilet_isaccessible', 'toilet_isfeecharged']
    print("Null counts before processing:")
    for col in toilet_cols:
        if col in df.columns:
            print(f"{col}: {df[col].isnull().sum()}")
    
    # Fill toilet_isaccessible nulls with 'FALSE' (assuming no toilet info means not accessible)
    if 'toilet_isaccessible' in df.columns:
        df['toilet_isaccessible'] = df['toilet_isaccessible'].fillna('FALSE')
        # Convert to boolean
        df['toilet_isaccessible'] = (df['toilet_isaccessible'] == 'TRUE').astype(int)
    
    # Fill toilet_isfeecharged nulls with 'FALSE' (assuming no toilet info means no fee)
    if 'toilet_isfeecharged' in df.columns:
        df['toilet_isfeecharged'] = df['toilet_isfeecharged'].fillna('FALSE')
        # Convert to boolean
        df['toilet_isfeecharged'] = (df['toilet_isfeecharged'] == 'TRUE').astype(int)
    
    # Check null counts after processing
    print("Null counts after processing:")
    for col in toilet_cols:
        if col in df.columns:
            print(f"{col}: {df[col].isnull().sum()}")
    
    return df

def fill_service_and_parking_nulls(df):
    """Fill null values in service and parking fields with appropriate statistics"""
    print("\n6. Filling null values in service and parking fields...")
    
    df = df.copy()
    
    # Service performance columns
    service_cols = [
        'service_operated_allweek_pct',
        'service_operated_weekday_pct', 
        'service_operated_weekend_pct',
        'kilometres_operated'
    ]
    
    # Parking columns
    parking_cols = ['bluebadgecarparkspaces']
    
    # Fill service columns with median (as they are percentages/continuous values)
    for col in service_cols:
        if col in df.columns:
            null_count_before = df[col].isnull().sum()
            if null_count_before > 0:
                median_value = df[col].median()
                df[col] = df[col].fillna(median_value)
                print(f"{col}: filled {null_count_before} nulls with median {median_value:.2f}")
    
    # Fill parking spaces with 0 (assuming no info means no spaces)
    for col in parking_cols:
        if col in df.columns:
            null_count_before = df[col].isnull().sum()
            if null_count_before > 0:
                df[col] = df[col].fillna(0)
                print(f"{col}: filled {null_count_before} nulls with 0")
    
    # Fill expected_attendance with 0 (no event means no expected attendance)
    if 'expected_attendance' in df.columns:
        null_count_before = df['expected_attendance'].isnull().sum()
        if null_count_before > 0:
            df['expected_attendance'] = df['expected_attendance'].fillna(0)
            print(f"expected_attendance: filled {null_count_before} nulls with 0")
    
    return df

def final_data_check(df):
    """Perform final data quality checks"""
    print("\n7. Final data quality check...")
    
    print(f"Final dataset shape: {df.shape}")
    
    # Check for remaining null values
    null_counts = df.isnull().sum()
    remaining_nulls = null_counts[null_counts > 0]
    
    if len(remaining_nulls) > 0:
        print("Remaining null values:")
        print(remaining_nulls)
else:
        print("✓ No remaining null values")
    
    # Check data types
    print("\nFinal data types:")
    print(df.dtypes.value_counts())
    
    # Display sample of final data
    print("\nSample of final processed data:")
    print(df.head())
    
    return df

def save_processed_data(df):
    """Save the processed data"""
    print(f"\n8. Saving processed data to {OUTPUT_FILE}...")
    
    # Ensure output directory exists
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the processed data
    df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"✓ Processed data saved successfully!")
    print(f"Final dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    
    return OUTPUT_FILE

def main():
    """Main data preparation pipeline"""
    print("=" * 60)
    print("DATA PREPARATION FOR MODELING PIPELINE")
    print("=" * 60)
    
    try:
        # Load data
        df = load_data()
        
        # Create a copy for processing
        df = df.copy()
        
        # Apply all transformations
        df = drop_redundant_fields(df)
        df = convert_date_features(df)
        df = create_weekend_boolean(df)
        df = convert_event_type_to_boolean(df)
        df = handle_toilet_nulls(df)
        df = fill_service_and_parking_nulls(df)
        
        # Final checks
        df = final_data_check(df)
        
        # Save processed data
        output_path = save_processed_data(df)
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        return df, output_path
        
    except Exception as e:
        print(f"\n❌ Error in data preparation pipeline: {e}")
        raise

if __name__ == "__main__":
    df_processed, output_file = main()