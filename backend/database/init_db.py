"""
Database Initialization Script

Loads calendar data from parquet file and populates SQLite database.
Uses MM-DD date format (no year) for year-agnostic matching.
"""

import os
import sqlite3
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "listings.db")
CALENDAR_PATH = os.path.join(project_root, "models", "production", "calendar")


def init_db():
    """
    Initialize database with calendar data from parquet file.

    Schema:
        locations (location TEXT, date TEXT, color TEXT, price_relative REAL)
        PRIMARY KEY (location, date)

    Date format: MM-DD (e.g., "01-01", "12-25")
    """
    print("=" * 70)
    print("DATABASE INITIALIZATION")
    print("=" * 70)

    # Check if calendar path exists
    if not os.path.exists(CALENDAR_PATH):
        print(f"✗ Error: Calendar path not found: {CALENDAR_PATH}")
        print(f"  Please ensure the calendar parquet file exists.")
        return

    print(f"\nCalendar path: {CALENDAR_PATH}")
    print(f"Database path: {DB_PATH}")

    # Load parquet file
    try:
        import pandas as pd

        print(f"\nLoading calendar parquet...")
        df = pd.read_parquet(CALENDAR_PATH)
        print(f"✓ Loaded {len(df)} rows from parquet")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Unique cities: {df['city'].nunique()}")
    except ImportError:
        print("✗ Error: pandas not installed. Run: pip install pandas pyarrow")
        return
    except Exception as e:
        print(f"✗ Error loading parquet: {e}")
        return

    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print(f"\nDropping old 'locations' table (if exists)...")
    cursor.execute("DROP TABLE IF EXISTS locations")

    print(f"Creating new 'locations' table...")
    cursor.execute(
        """
        CREATE TABLE locations (
            location TEXT,
            date TEXT,
            color TEXT,
            price_relative REAL,
            PRIMARY KEY (location, date)
        )
        """
    )
    print(f"✓ Table created")

    # Prepare data for insertion
    print(f"\nPreparing data for insertion...")
    data = []
    for _, row in df.iterrows():
        city = row["city"]
        mm_dd = row["date"]  # Already in MM-DD format
        stoplight = row["stoplight"]  # green/yellow/orange/red
        price_relative = row["price_relative"]

        # Map stoplight to CSS class
        color_class = f"airbnb-day-{stoplight}"

        data.append((city, mm_dd, color_class, price_relative))

    print(f"✓ Prepared {len(data)} entries")

    # Insert data
    print(f"\nInserting data into database...")
    cursor.executemany(
        "INSERT OR REPLACE INTO locations (location, date, color, price_relative) VALUES (?, ?, ?, ?)",
        data,
    )

    conn.commit()
    print(f"✓ Inserted {len(data)} entries")

    # Verify insertion
    cursor.execute("SELECT COUNT(*) FROM locations")
    count = cursor.fetchone()[0]
    print(f"\nVerification: {count} rows in database")

    # Show sample data
    print(f"\nSample data:")
    cursor.execute("SELECT * FROM locations WHERE location='london' LIMIT 5")
    rows = cursor.fetchall()
    print(f"  {'Location':<15} {'Date':<8} {'Color':<20} {'Price Relative'}")
    print(f"  {'-' * 15} {'-' * 8} {'-' * 20} {'-' * 15}")
    for row in rows:
        print(f"  {row[0]:<15} {row[1]:<8} {row[2]:<20} {row[3]:>8.4f}")

    conn.close()

    print("\n" + "=" * 70)
    print("DATABASE INITIALIZATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    init_db()
