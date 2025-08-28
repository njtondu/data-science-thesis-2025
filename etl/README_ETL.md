# TFL Stations ETL Script

This script extracts, transforms, and loads TFL (Transport for London) station data from their API to create a comprehensive stations dimension table.

## Overview

The script performs the following operations:
1. **Extract**: Connects to the TFL API and downloads station data
2. **Transform**: Processes and joins multiple data sources to create a unified stations dimension table
3. **Load**: Saves the final dataset as a CSV file

## Files

- `tfl_stations_etl.py`: Main ETL script
- `requirements.txt`: Python dependencies
- `data/stations_dimension_table.csv`: Output file (generated after running the script)

## Installation

1. Create a virtual environment:
```bash
python3 -m venv venv
```

2. Activate the virtual environment:
```bash
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate     # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the script with the virtual environment activated:

```bash
source venv/bin/activate
python tfl_stations_etl.py
```

## Output

The script generates a CSV file at `data/stations_dimension_table.csv` containing:

- **StopAreaNaptanCode**: Unique station identifier
- **Mode**: Transport mode (e.g., tube, overground, elizabeth-line)
- **Line**: Specific line name
- **UniqueId**: Station unique ID
- **Name**: Station name
- **FareZones**: Fare zone information
- **HubNaptanCode**: Hub station code
- **Wifi**: WiFi availability
- **AirportInterchange**: Airport interchange information
- **BlueBadgeCarParking**: Blue badge parking availability
- **BlueBadgeCarParkSpaces**: Number of blue badge parking spaces
- **StationUniqueId**: Station unique ID (from toilets data)
- **IsAccessible**: Accessibility information
- **IsFeeCharged**: Whether fees are charged
- **Id**: Toilet facility ID

## Data Sources

The script combines data from multiple TFL API endpoints:
- **Stations.csv**: Basic station information
- **ModesAndLines.csv**: Transport modes and line mappings
- **PlatformServices.csv**: Station-line relationships
- **Toilets.csv**: Toilet facility information

## Functions

### `connect_to_tfl_api()`
- Downloads data from TFL API
- Returns a dictionary containing all required DataFrames

### `transform_station_data(dataframes)`
- Processes and joins the raw data
- Creates the final stations dimension table
- Returns the transformed DataFrame

### `save_to_csv(dataframe, output_path)`
- Saves the final DataFrame to CSV format
- Creates output directory if it doesn't exist

## Example Output

The script processes approximately:
- 509 stations
- 23 transport modes/lines
- 1771 platform services
- 398 toilet facilities

Final output: ~946 records with 15 columns

## Error Handling

The script includes error handling for:
- API connection failures
- Missing data files
- Directory creation issues
- Data transformation errors
