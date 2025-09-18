"""
Anomaly Detection Utilities for London Underground Passenger Monitoring

This module provides utilities for preparing data and detecting anomalies in passenger counts
using a linear regression model as the expected baseline.
"""

import pandas as pd
import numpy as np
from typing import Dict, Union


def ensure_daily(df: pd.DataFrame, 
                target: str = "entrytapcount", 
                station: str = "station_uid", 
                date: str = "date") -> pd.DataFrame:
    """
    Aggregate data to daily granularity with proper handling of different column types.
    
    Args:
        df: Input DataFrame with passenger data
        target: Column name for entry tap count (default: "entrytapcount")
        station: Column name for station identifier (default: "station_uid")
        date: Column name for date (default: "date")
        
    Returns:
        DataFrame aggregated to daily level, one row per (station, date)
        
    Raises:
        ValueError: If required columns are missing or invalid
    """
    # Input validation
    if df is None or df.empty:
        raise ValueError("Input DataFrame cannot be None or empty")
    
    if station not in df.columns:
        raise ValueError(f"Station column '{station}' not found in DataFrame")
    
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame")
    
    # Create a copy to avoid modifying original
    df_work = df.copy()
    
    # Handle date column - build from year, month, day if date is missing
    if date not in df_work.columns:
        required_date_cols = ['year', 'month', 'day']
        missing_cols = [col for col in required_date_cols if col not in df_work.columns]
        if missing_cols:
            raise ValueError(f"Date column '{date}' missing and cannot build from year/month/day. Missing: {missing_cols}")
        
        print(f"Building '{date}' column from year, month, day")
        df_work[date] = pd.to_datetime(df_work[['year', 'month', 'day']])
    else:
        # Ensure date is datetime
        df_work[date] = pd.to_datetime(df_work[date])
    
    # Keep only finite target values
    initial_count = len(df_work)
    df_work = df_work[np.isfinite(df_work[target])]
    filtered_count = len(df_work)
    
    if filtered_count < initial_count:
        print(f"Filtered out {initial_count - filtered_count} rows with non-finite {target} values")
    
    if df_work.empty:
        raise ValueError(f"No finite values found in {target} column")
    
    # Identify column types for aggregation
    numeric_cols = df_work.select_dtypes(include=[np.number]).columns.tolist()
    boolean_cols = df_work.select_dtypes(include=['bool']).columns.tolist()
    categorical_cols = df_work.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Remove grouping columns from aggregation lists
    grouping_cols = [station, date]
    numeric_cols = [col for col in numeric_cols if col not in grouping_cols]
    boolean_cols = [col for col in boolean_cols if col not in grouping_cols]
    categorical_cols = [col for col in categorical_cols if col not in grouping_cols]
    
    # Build aggregation dictionary
    agg_dict = {}
    
    # Target gets summed
    if target in numeric_cols:
        agg_dict[target] = 'sum'
        numeric_cols.remove(target)  # Remove from numeric list to avoid duplicate
    
    # Handle binary integer columns (like is_event) - use max instead of mean
    binary_int_cols = []
    for col in numeric_cols:
        unique_vals = set(df_work[col].dropna().unique())
        if unique_vals.issubset({0, 1, 0.0, 1.0}) and len(unique_vals) <= 2:
            binary_int_cols.append(col)
            agg_dict[col] = 'max'  # Use max for binary columns
    
    # Remove binary columns from numeric list
    numeric_cols = [col for col in numeric_cols if col not in binary_int_cols]
    
    # Other numeric columns get mean
    for col in numeric_cols:
        agg_dict[col] = 'mean'
    
    # Boolean columns get max (if any record is True/1, the day is True/1)
    for col in boolean_cols:
        agg_dict[col] = 'max'
    
    # Categorical columns get first
    for col in categorical_cols:
        agg_dict[col] = 'first'
    
    print(f"Aggregating {len(df_work)} rows to daily level...")
    print(f"  Target '{target}': sum")
    print(f"  Binary integer columns ({len(binary_int_cols)}): max")
    print(f"  Numeric columns ({len(numeric_cols)}): mean")
    print(f"  Boolean columns ({len(boolean_cols)}): max")
    print(f"  Categorical columns ({len(categorical_cols)}): first")
    
    # Group and aggregate
    df_daily = df_work.groupby([station, date]).agg(agg_dict).reset_index()
    
    # Sort by station and date
    df_daily = df_daily.sort_values([station, date]).reset_index(drop=True)
    
    print(f"Result: {len(df_daily)} daily records for {df_daily[station].nunique()} stations")
    
    return df_daily


def time_split(df_daily: pd.DataFrame, 
               date: str = "date", 
               train_ratio: float = 0.7, 
               val_ratio: float = 0.15) -> Dict[str, pd.DataFrame]:
    """
    Split daily data chronologically into train/validation/test sets with no date overlap.
    
    Args:
        df_daily: Daily aggregated DataFrame
        date: Column name for date (default: "date")
        train_ratio: Proportion for training set (default: 0.7)
        val_ratio: Proportion for validation set (default: 0.15)
        
    Returns:
        Dictionary with keys 'train', 'val', 'test' containing respective DataFrames
        
    Raises:
        ValueError: If ratios are invalid or required columns missing
    """
    # Input validation
    if df_daily is None or df_daily.empty:
        raise ValueError("Input DataFrame cannot be None or empty")
    
    if date not in df_daily.columns:
        raise ValueError(f"Date column '{date}' not found in DataFrame")
    
    if not (0 < train_ratio < 1):
        raise ValueError("train_ratio must be between 0 and 1")
    
    if not (0 < val_ratio < 1):
        raise ValueError("val_ratio must be between 0 and 1")
    
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1")
    
    test_ratio = 1 - train_ratio - val_ratio
    
    print(f"Splitting data chronologically:")
    print(f"  Train: {train_ratio:.1%}, Val: {val_ratio:.1%}, Test: {test_ratio:.1%}")
    
    # Create a copy and ensure date is datetime
    df_work = df_daily.copy()
    df_work[date] = pd.to_datetime(df_work[date])
    
    # Get unique dates and sort them
    unique_dates = sorted(df_work[date].unique())
    total_dates = len(unique_dates)
    
    print(f"  Date range: {unique_dates[0].strftime('%Y-%m-%d')} to {unique_dates[-1].strftime('%Y-%m-%d')}")
    print(f"  Total unique dates: {total_dates}")
    
    # Calculate split points
    train_end_idx = int(total_dates * train_ratio)
    val_end_idx = int(total_dates * (train_ratio + val_ratio))
    
    # Define date ranges
    train_dates = unique_dates[:train_end_idx]
    val_dates = unique_dates[train_end_idx:val_end_idx]
    test_dates = unique_dates[val_end_idx:]
    
    print(f"  Train dates: {len(train_dates)} ({train_dates[0].strftime('%Y-%m-%d')} to {train_dates[-1].strftime('%Y-%m-%d')})")
    print(f"  Val dates: {len(val_dates)} ({val_dates[0].strftime('%Y-%m-%d')} to {val_dates[-1].strftime('%Y-%m-%d')})")
    print(f"  Test dates: {len(test_dates)} ({test_dates[0].strftime('%Y-%m-%d')} to {test_dates[-1].strftime('%Y-%m-%d')})")
    
    # Split the data
    df_train = df_work[df_work[date].isin(train_dates)].copy()
    df_val = df_work[df_work[date].isin(val_dates)].copy()
    df_test = df_work[df_work[date].isin(test_dates)].copy()
    
    # Sort each split by date
    df_train = df_train.sort_values(date).reset_index(drop=True)
    df_val = df_val.sort_values(date).reset_index(drop=True)
    df_test = df_test.sort_values(date).reset_index(drop=True)
    
    # Verify no date overlap
    max_train_date = df_train[date].max()
    min_val_date = df_val[date].min()
    max_val_date = df_val[date].max()
    min_test_date = df_test[date].min()
    
    assert max_train_date < min_val_date, f"Date overlap: max train ({max_train_date}) >= min val ({min_val_date})"
    assert max_val_date < min_test_date, f"Date overlap: max val ({max_val_date}) >= min test ({min_test_date})"
    
    print(f"  ✅ No date overlap verified")
    print(f"  Train: {len(df_train)} records")
    print(f"  Val: {len(df_val)} records") 
    print(f"  Test: {len(df_test)} records")
    
    return {
        "train": df_train,
        "val": df_val,
        "test": df_test
    }


# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    print("Creating sample data for testing...")
    
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    stations = ['LUL001', 'LUL002', 'LUL003']
    
    # Create sample data with multiple records per day per station
    sample_data = []
    for date in dates[:10]:  # Use first 10 days for testing
        for station in stations:
            # Multiple records per day to test aggregation
            for hour in [8, 12, 18]:  # Morning, noon, evening
                sample_data.append({
                    'station_uid': station,
                    'date': date,
                    'year': date.year,
                    'month': date.month,
                    'day': date.day,
                    'entrytapcount': np.random.randint(100, 1000),
                    'temperature': np.random.normal(15, 5),
                    'is_weekend': date.weekday() >= 5,
                    'line': f'Line_{station[-1]}'
                })
    
    df_sample = pd.DataFrame(sample_data)
    print(f"Sample data created: {len(df_sample)} records")
    
    # Test ensure_daily
    print("\n" + "="*50)
    print("Testing ensure_daily function...")
    df_daily = ensure_daily(df_sample)
    print(f"Daily aggregation result: {df_daily.shape}")
    print(df_daily.head())
    
    # Test time_split
    print("\n" + "="*50)
    print("Testing time_split function...")
    splits = time_split(df_daily)
    
    print(f"\nSplit results:")
    for split_name, split_df in splits.items():
        print(f"  {split_name}: {split_df.shape}")
