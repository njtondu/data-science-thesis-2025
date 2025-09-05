import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_EVENTS_FILE = PROCESSED_DIR / "raw_event_sessions_2023.csv"
OUTPUT_FILE = PROCESSED_DIR / "event_sessions_2023.csv"


def build_event_sessions():
	# Load raw events (already curated list)
	events_df = pd.read_csv(RAW_EVENTS_FILE)

	# Normalize core fields
	events_df['date'] = pd.to_datetime(events_df['Fecha']).dt.date
	events_df = events_df.rename(columns={
		'Tipo': 'event_type',
		'Evento/Festividad': 'event_name',
		'Asistentes estimados': 'expected_attendance',
	})

	# Explode affected lines
	rows = []
	for _, row in events_df.iterrows():
		affected = row.get('Líneas afectadas (parseadas)')
		if pd.isna(affected) or affected == 'N/A (Impacto general en la red)':
			new_row = row.copy()
			new_row['affected_lines'] = 'General'
			rows.append(new_row)
		else:
			for ln in str(affected).split(','):
				new_row = row.copy()
				new_row['affected_lines'] = ln.strip()
				rows.append(new_row)

	events_expanded = pd.DataFrame(rows)

	# Select final columns
	final_cols = ['date', 'event_type', 'event_name', 'affected_lines', 'expected_attendance']
	event_sessions = events_expanded[final_cols].copy()

	# Save
	OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
	event_sessions.to_csv(OUTPUT_FILE, index=False)
	return OUTPUT_FILE


if __name__ == "__main__":
	out = build_event_sessions()
	print(f"Saved event sessions to: {out}")
