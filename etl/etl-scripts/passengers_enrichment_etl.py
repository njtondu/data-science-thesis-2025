import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_DIR = DATA_DIR / "raw-data" / "eda_data"
PROCESSED_DIR = DATA_DIR / "processed"

PASSENGERS_FILE = RAW_DIR / "station_passenger_counts_2023.csv"
STATIONS_FILE = PROCESSED_DIR / "stations_dimension_table.csv"
EVENTS_FILE = PROCESSED_DIR / "event_sessions_2023.csv"
WEATHER_FILE = PROCESSED_DIR / "london_weather_2023_clean.csv"
PERF_FILE = PROCESSED_DIR / "monthly_tube_performance_2023.csv"
OUTPUT_FILE = PROCESSED_DIR / "passengers_enriched_2023.csv"


def normalize_passengers(df: pd.DataFrame) -> pd.DataFrame:
	df = df.copy()
	df.columns = df.columns.str.lower().str.replace(' ', '_')
	df['traveldate'] = pd.to_datetime(df['traveldate'].astype(str), format='%Y%m%d')
	df = df.rename(columns={'station': 'station_name'})
	return df


def normalize_stations(df: pd.DataFrame) -> pd.DataFrame:
	df = df.copy()
	df.columns = df.columns.str.lower().str.replace(' ', '_')
	print(f"Columns before dropping: {list(df.columns)}")
	df = df.drop(columns=['stopareanaptancode', 'stationuniqueid'], errors='ignore')
	print(f"Columns after dropping: {list(df.columns)}")
	df = df.rename(columns={
		'uniqueid': 'station_uid', 
		'name': 'station_name',
		'type': 'toilet_type',
		'isaccessible': 'toilet_isaccessible',
		'isfeecharged': 'toilet_isfeecharged'
	})
	print(f"Columns after renaming: {list(df.columns)}")
	
	# Handle toilet information aggregation to avoid duplicates
	# Group by station and aggregate toilet information
	def aggregate_toilets(group):
		# Get the first row for non-toilet columns
		result = group.iloc[0].copy()
		
		# Handle toilet information
		toilet_types = group['toilet_type'].dropna().unique()
		toilet_accessible = group['toilet_isaccessible'].dropna().unique()
		toilet_feecharged = group['toilet_isfeecharged'].dropna().unique()
		
		# Aggregate toilet information
		if len(toilet_types) > 0:
			# If multiple types, concatenate them or use a representative value
			result['toilet_type'] = ', '.join(sorted([str(t) for t in toilet_types if pd.notna(t)]))
			if result['toilet_type'] == '':
				result['toilet_type'] = None
		else:
			result['toilet_type'] = None
			
		# For accessibility, if any toilet is accessible, mark as accessible
		if len(toilet_accessible) > 0:
			accessible_values = [str(v).lower() for v in toilet_accessible if pd.notna(v)]
			result['toilet_isaccessible'] = 'True' if 'true' in accessible_values else accessible_values[0] if accessible_values else None
		else:
			result['toilet_isaccessible'] = None
			
		# For fee charged, use the most common value or first non-null
		if len(toilet_feecharged) > 0:
			feecharged_values = [str(v).lower() for v in toilet_feecharged if pd.notna(v)]
			result['toilet_isfeecharged'] = feecharged_values[0] if feecharged_values else None
		else:
			result['toilet_isfeecharged'] = None
		
		return result
	
	# Group by station identifiers and aggregate
	groupby_cols = ['station_name', 'station_uid', 'mode', 'line']
	# Only group if we have duplicates, otherwise return as is
	if df.duplicated(subset=groupby_cols).any():
		print(f"Found duplicates, aggregating by: {groupby_cols}")
		df_agg = df.groupby(groupby_cols, as_index=False).apply(aggregate_toilets).reset_index(drop=True)
		print(f"Shape before aggregation: {df.shape}, after: {df_agg.shape}")
		return df_agg
	else:
		print("No duplicates found, returning original dataframe")
		return df


def normalize_events(df: pd.DataFrame) -> pd.DataFrame:
	df = df.copy()
	df['date'] = pd.to_datetime(df['date']).dt.date
	df['affected_line_norm'] = df['affected_lines'].astype(str).str.lower().str.replace(r'[^a-z]', '', regex=True)
	return df


def normalize_weather(df: pd.DataFrame) -> pd.DataFrame:
	df = df.copy()
	df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
	df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d').dt.date
	keep = ['date','max_temp','min_temp','mean_temp','precipitation_amount','relative_humidity','cloud_cover','sunshine_duration','sea_level_pressure']
	return df[keep]


def normalize_performance(df: pd.DataFrame) -> pd.DataFrame:
	df = df.copy()
	df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
	df['line_name'] = df['line_name'].replace({'C&H': 'Circle Hammersmith & City'})
	
	# Remove duplicates from performance data first
	print(f"Performance data shape before dedup: {df.shape}")
	df = df.drop_duplicates(subset=['year', 'month', 'line_name'])
	print(f"Performance data shape after dedup: {df.shape}")
	
	mask = df['line_name'].astype(str).str.lower().str.contains('circle') & df['line_name'].astype(str).str.lower().str.contains('hammersmith')
	combined = df[mask]
	if not combined.empty:
		dup1 = combined.copy(); dup1['line_name'] = 'Circle'
		dup2 = combined.copy(); dup2['line_name'] = 'Hammersmith & City'
		df = pd.concat([df[~mask], dup1, dup2], ignore_index=True)
	df['line_norm'] = df['line_name'].astype(str).str.lower().str.replace(r'[^a-z]', '', regex=True)
	df['year'] = df['year'].astype(int)
	df['month'] = df['month'].astype(int)
	keep = ['year','month','line_norm','service_operated_allweek_pct','service_operated_weekday_pct','service_operated_weekend_pct','kilometres_operated']
	return df[keep]


def run_pipeline() -> Path:
	# Load
	p_src = pd.read_csv(PASSENGERS_FILE)
	s_src = pd.read_csv(STATIONS_FILE)
	e_src = pd.read_csv(EVENTS_FILE)
	w_src = pd.read_csv(WEATHER_FILE)
	perf_src = pd.read_csv(PERF_FILE)

	# Normalize
	p = normalize_passengers(p_src)
	s = normalize_stations(s_src)
	e = normalize_events(e_src)
	w = normalize_weather(w_src)
	perf = normalize_performance(perf_src)

	# Join passengers + stations
	p_s = p.merge(s, on='station_name', how='inner')
	p_s['date'] = p_s['traveldate'].dt.date
	p_s['line_norm'] = p_s['line'].astype(str).str.lower().str.replace(r'[^a-z]', '', regex=True)

	# Add events (date + line)
	ps_e = p_s.merge(
		e[['date','affected_line_norm','event_type','event_name','expected_attendance']],
		left_on=['date','line_norm'], right_on=['date','affected_line_norm'], how='left'
	).drop(columns=['affected_line_norm'])

	# Add weather (date)
	ps_e_w = ps_e.merge(w, on='date', how='left')

	# Add monthly performance (year, month, line)
	ps_e_w['year'] = pd.to_datetime(ps_e_w['date']).dt.year
	ps_e_w['month'] = pd.to_datetime(ps_e_w['date']).dt.month
	ps_e_w_perf = ps_e_w.merge(perf, on=['year','month','line_norm'], how='left')

	# Final columns order (drop traveldate)
	id_cols = ['date','dayofweek','station_name','station_uid','mode','line','farezones','hubnaptancode']
	counts_cols = ['entrytapcount','exittapcount']
	event_cols = ['event_type','event_name','expected_attendance']
	weather_cols = ['max_temp','min_temp','mean_temp','precipitation_amount','relative_humidity','cloud_cover','sunshine_duration','sea_level_pressure']
	perf_cols = ['service_operated_allweek_pct','service_operated_weekday_pct','service_operated_weekend_pct','kilometres_operated']
	other_station_cols = ['wifi','airportinterchange','bluebadgecarparking','bluebadgecarparkspaces','toilet_isaccessible','toilet_isfeecharged','toilet_type']

	ps_e_w_perf = ps_e_w_perf.drop(columns=['traveldate'], errors='ignore')
	ordered_cols = id_cols + counts_cols + event_cols + weather_cols + perf_cols + other_station_cols
	ordered_cols = [c for c in ordered_cols if c in ps_e_w_perf.columns]

	final_df = ps_e_w_perf[ordered_cols].copy()

	# Final deduplication step to ensure no duplicates remain
	print(f"Final dataset shape before dedup: {final_df.shape}")
	duplicates_before = final_df.duplicated(subset=['date', 'station_name', 'line']).sum()
	print(f"Duplicates before final dedup: {duplicates_before}")
	
	if duplicates_before > 0:
		final_df = final_df.drop_duplicates(subset=['date', 'station_name', 'line'])
		print(f"Final dataset shape after dedup: {final_df.shape}")
		duplicates_after = final_df.duplicated(subset=['date', 'station_name', 'line']).sum()
		print(f"Duplicates after final dedup: {duplicates_after}")

	# Save
	OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
	final_df.to_csv(OUTPUT_FILE, index=False)
	return OUTPUT_FILE


if __name__ == "__main__":
	out = run_pipeline()
	print(f"Saved passengers enrichment to: {out}")
