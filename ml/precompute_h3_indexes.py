import pandas as pd
import os
from ml.utils.geo import add_h3_index

def precompute_and_save_h3_indexes():
    """
    This script performs a one-time pre-computation of H3 indexes.
    It reads the source CSV files, adds the H3 index column, and saves them
    as Parquet files in the production model directory, overwriting existing files.
    """
    h3_resolution = 3
    
    # Define paths
    backend_dir = 'backend'
    models_dir = 'models/production'
    
    source_files = {
        'airport.csv': 'airports.parquet',
        'train_stations.csv': 'train_stations.parquet'
    }
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    for csv_file, parquet_file in source_files.items():
        source_path = os.path.join(backend_dir, csv_file)
        target_path = os.path.join(models_dir, parquet_file)
        
        try:
            print(f"Processing {source_path}...")
            
            # Read the source CSV
            df = pd.read_csv(source_path)
            
            # Add H3 index
            print(f"  Adding H3 index with resolution {h3_resolution}...")
            df_indexed = add_h3_index(df, h3_resolution=h3_resolution)
            
            # Save as Parquet, overwriting if it exists
            print(f"  Saving indexed data to {target_path}...")
            df_indexed.to_parquet(target_path, index=False)
            
            print(f"  Successfully created {target_path}")

        except FileNotFoundError:
            print(f"  ! Warning: Source file not found at {source_path}. Skipping.")
        except Exception as e:
            print(f"  ! Error processing {csv_file}: {e}")

if __name__ == "__main__":
    precompute_and_save_h3_indexes()
