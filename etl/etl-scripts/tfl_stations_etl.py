import requests
import zipfile
import io
import pandas as pd
import os
from pathlib import Path


def connect_to_tfl_api():
    """
    Connect to TFL API and download station data
    Returns a dictionary with the required DataFrames
    """
    print("Connecting to TFL API...")
    
    url = "https://api.tfl.gov.uk/stationdata/tfl-stationdata-detailed.zip"
    r = requests.get(url)
    
    if r.status_code != 200:
        raise Exception(f"Failed to download data from TFL API. Status code: {r.status_code}")
    
    # Load DataFrames from the ZIP file
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        print("Files available in ZIP:", z.namelist())
        
        # Load required CSV files
        stations_df = pd.read_csv(z.open("Stations.csv"))
        modes_lines_df = pd.read_csv(z.open("ModesAndLines.csv"))
        platform_services_df = pd.read_csv(z.open("PlatformServices.csv"))
        toilets_df = pd.read_csv(z.open("Toilets.csv"))
    
    print("Successfully loaded data from TFL API")
    print(f"Stations: {stations_df.shape[0]} rows, {stations_df.shape[1]} columns")
    print(f"Modes and Lines: {modes_lines_df.shape[0]} rows, {modes_lines_df.shape[1]} columns")
    print(f"Platform Services: {platform_services_df.shape[0]} rows, {platform_services_df.shape[1]} columns")
    print(f"Toilets: {toilets_df.shape[0]} rows, {toilets_df.shape[1]} columns")
    
    return {
        'stations': stations_df,
        'modes_lines': modes_lines_df,
        'platform_services': platform_services_df,
        'toilets': toilets_df
    }


def transform_station_data(dataframes):
    """
    Transform the raw data into the final stations dimension table
    """
    print("Transforming station data...")
    
    stations_df = dataframes['stations']
    modes_lines_df = dataframes['modes_lines']
    platform_services_df = dataframes['platform_services']
    toilets_df = dataframes['toilets']
    
    # Step 1: Extract station-line relationships from PlatformServices
    station_line_df = platform_services_df[["StopAreaNaptanCode", "Line"]].drop_duplicates()
    print(f"Step 1: Created station-line mapping with {station_line_df.shape[0]} unique combinations")
    
    # Step 2: Map lines to modes using ModesAndLines
    line_mapping_df = station_line_df.merge(
        modes_lines_df,
        left_on="Line",
        right_on="Name",
        how="left"
    )
    line_mapping_df = line_mapping_df[["StopAreaNaptanCode", "Mode", "Line"]].drop_duplicates()
    print(f"Step 2: Added mode information, resulting in {line_mapping_df.shape[0]} records")
    
    # Step 3: Select relevant columns from stations
    station_cols_df = stations_df[[
        "UniqueId",
        "Name",
        "FareZones",
        "HubNaptanCode",
        "Wifi",
        "AirportInterchange",
        "BlueBadgeCarParking",
        "BlueBadgeCarParkSpaces"
    ]]
    print(f"Step 3: Selected {station_cols_df.shape[1]} columns from stations data")
    
    # Step 4: Join station-line mapping with station information
    station_line_joined_df = line_mapping_df.merge(
        station_cols_df,
        left_on="StopAreaNaptanCode",
        right_on="UniqueId",
        how="left"
    )
    print(f"Step 4: Joined with station data, resulting in {station_line_joined_df.shape[0]} records")
    
    # Step 5: Add toilet information
    stations_dim_df = station_line_joined_df.merge(
        toilets_df[["StationUniqueId", "IsAccessible", "IsFeeCharged", "Id", "Type"]],
        left_on="UniqueId",
        right_on="StationUniqueId",
        how="left"
    )
    print(f"Step 5: Added toilet information, final table has {stations_dim_df.shape[0]} records and {stations_dim_df.shape[1]} columns")
    print(f"Columns after toilet merge: {list(stations_dim_df.columns)}")
    
    # Step 6: Clean up BlueBadgeCarParking/BlueBadgeCarParkSpaces relationship
    # When BlueBadgeCarParking is False, set BlueBadgeCarParkSpaces to 0
    stations_dim_df.loc[stations_dim_df['BlueBadgeCarParking'] == False, 'BlueBadgeCarParkSpaces'] = 0
    print(f"Step 6: Cleaned up BlueBadgeCarParking/BlueBadgeCarParkSpaces relationship")
    
    return stations_dim_df


def save_to_csv(dataframe, output_path):
    """
    Save the final dataframe to CSV file
    """
    print(f"Saving stations dimension table to {output_path}...")
    
    # Create directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Save to CSV
    print(f"Columns being saved: {list(dataframe.columns)}")
    dataframe.to_csv(output_path, index=False)
    print(f"Successfully saved {dataframe.shape[0]} records to {output_path}")


def save_raw_data(dataframes, raw_data_dir="../data/raw-data/api-station-data"):
    """
    Optionally save raw data files for data lineage
    """
    print(f"Saving raw data files to {raw_data_dir}...")
    
    # Create raw data directory if it doesn't exist
    if not os.path.exists(raw_data_dir):
        os.makedirs(raw_data_dir)
        print(f"Created directory: {raw_data_dir}")
    
    # Save each raw dataframe
    for name, df in dataframes.items():
        file_path = os.path.join(raw_data_dir, f"{name}.csv")
        df.to_csv(file_path, index=False)
        print(f"Saved {name}.csv: {df.shape[0]} rows, {df.shape[1]} columns")
    
    print("Raw data files saved successfully")


def main():
    """
    Main function to orchestrate the ETL process
    """
    try:
        # Step 1: Connect to TFL API
        dataframes = connect_to_tfl_api()
        
        # Step 2: Save raw data (optional - for data lineage)
        save_raw_data(dataframes)
        
        # Step 3: Transform the data
        stations_dim_df = transform_station_data(dataframes)
        
        # Step 4: Save to CSV in the processed data directory
        output_path = "/Users/NTondu/Desktop/data-science-thesis-2025/data/processed/stations_dimension_table.csv"
        save_to_csv(stations_dim_df, output_path)
        
        print("\nETL process completed successfully!")
        print(f"Final dataset: {stations_dim_df.shape[0]} rows, {stations_dim_df.shape[1]} columns")
        
        # Display sample of the final data
        print("\nSample of the final dataset:")
        print(stations_dim_df.head())
        
    except Exception as e:
        print(f"Error during ETL process: {str(e)}")
        raise


if __name__ == "__main__":
    main()
