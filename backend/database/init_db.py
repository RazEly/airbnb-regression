import csv
import os
import sqlite3
from datetime import datetime

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "database.db")
CSV_PATH = os.path.join(os.path.dirname(BASE_DIR), "calendar.csv")


def init_db():
    # Ensure we are in the right directory or use absolute paths if needed
    # But based on the environment, relative to root is fine or we check existence
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Drop old table if exists (for schema update)
    cursor.execute("DROP TABLE IF EXISTS locations")

    # Create new table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS locations (
            location TEXT,
            date TEXT,
            color TEXT,
            PRIMARY KEY (location, date)
        )
    """
    )

    print(f"Reading data from {CSV_PATH}...")

    data = []
    years = [2026, 2027]

    with open(CSV_PATH, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            city = row["city"].strip()
            mm_dd = row["date"].strip()  # Format: MM-DD
            color_name = row["color"].strip()

            # Map color name to CSS class
            color_class = f"airbnb-day-{color_name}"

            for year in years:
                date_str = f"{year}-{mm_dd}"
                data.append((city, date_str, color_class))

    print(f"Inserting {len(data)} entries into the database...")

    # Insert data
    cursor.executemany(
        "INSERT OR REPLACE INTO locations (location, date, color) VALUES (?, ?, ?)",
        data,
    )

    conn.commit()
    conn.close()
    print("Database initialized successfully with CSV data.")


if __name__ == "__main__":
    init_db()
