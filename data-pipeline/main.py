import sys
import os
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import re

from remove_outliers import process_zip_file_outliers
from trim import process_zip_file


def trim():
    # Setup paths
    data_dirs = [Path('./data/Timo')] 
    # data_dirs.append(['./data/Nathaniel']) # just using timo data for now.
    output_dir = Path('./data_trimmed')
    output_dir.mkdir(exist_ok=True)

    # Create subdirectories
    for data_dir in data_dirs:
        if data_dir.exists():
            (output_dir / data_dir.name).mkdir(exist_ok=True)

    all_results = []
    variance_deltas = []

    # Process all zip files
    for data_dir in data_dirs:
        if not data_dir.exists():
            print(f"Warning: {data_dir} does not exist, skipping...")
            continue
            
        output_subdir = output_dir / data_dir.name
        
        for zip_file in data_dir.glob('*.zip'):
            print(f"Processing {zip_file}...")
            results = process_zip_file(zip_file, output_subdir)
            all_results.append(results)
            
            # Calculate variance deltas for each CSV file
            for csv_name in results['original_var'].keys():
                orig_overall = results['original_var'][csv_name].get('overall', 0)
                trim_overall = results['trimmed_var'][csv_name].get('overall', 0)
                delta = (orig_overall - trim_overall) / orig_overall * 100  # Percentage reduction
                variance_deltas.append({
                    'file': f"{results['file']}_{csv_name}",
                    'delta_percent': delta,
                    'sensor_type': csv_name.replace('.csv', '')
                })

    # Print summary statistics
    print(f"\nProcessed {len(all_results)} zip files")
    print(f"Variance reduction statistics:")
    deltas = [d['delta_percent'] for d in variance_deltas]
    print(f"  Mean reduction: {np.mean(deltas):.2f}%")
    print(f"  Median reduction: {np.median(deltas):.2f}%")
    print(f"  Std deviation: {np.std(deltas):.2f}%")

    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(deltas, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Variance Reduction (%)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Variance Reduction After Trimming First/Last Seconds')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='No change')
    plt.axvline(x=np.mean(deltas), color='green', linestyle='-', alpha=0.7, 
                label=f'Mean: {np.mean(deltas):.1f}%')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'variance_reduction_histogram.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Save detailed results
    with open(output_dir / 'variance_analysis.json', 'w') as f:
        json.dump({
            'summary': {
                'mean_reduction_percent': float(np.mean(deltas)),
                'median_reduction_percent': float(np.median(deltas)),
                'std_reduction_percent': float(np.std(deltas))
            },
            'detailed_results': variance_deltas
        }, f, indent=2)

    print(f"\nResults saved to {output_dir}")
    print("Histogram saved as 'variance_reduction_histogram.png'")
    print("Detailed analysis saved as 'variance_analysis.json'")



def outlier_analysis():
    zip_folder = "./data_trimmed/Timo/"
    output_folder = "./data_cleaned/Timo/"

    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    zip_files = [f for f in os.listdir(zip_folder) if f.endswith('.zip')]

    all_results = {}
    for zip_file in zip_files:
        zip_path = os.path.join(zip_folder, zip_file)
        output_zip_path = os.path.join(output_folder, zip_file)
        
        print(f"\n{'='*50}")
        print(f"Processing: {zip_file}")
        
        # Process the zip file
        result = process_zip_file_outliers(zip_path, output_folder, zip_file)
        if result is None:
            continue
            
        all_results[zip_file] = result
        cleaned_data = result['cleaned_data']
        
        # Create new zip file with cleaned data
        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as new_zip:
            # Copy meta folder from original zip
            with zipfile.ZipFile(zip_path, 'r') as orig_zip:
                for file_info in orig_zip.infolist():
                    if file_info.filename.startswith('meta/'):
                        new_zip.writestr(file_info, orig_zip.read(file_info.filename))
            
            # Write cleaned CSV files
            for sensor_type, cleaned_df in cleaned_data.items():
                csv_filename = f"{sensor_type}.csv"
                csv_content = cleaned_df.to_csv(index=False)
                new_zip.writestr(csv_filename, csv_content)
        
        print(f"Created cleaned zip: {output_zip_path}")


def combine():
    # Configuration: folder containing the zip files
    DATA_PARENT_DIR = './data_cleaned/'
    SOURCE_DIR = 'Timo'
    DATA_DIR = os.path.join(DATA_PARENT_DIR, SOURCE_DIR)

    # NEW: define where to write the Parquet outputs
    # e.g. OUTPUT_DIR = '../data_processed/parquet_outputs'
    OUTPUT_DIR = os.path.join(DATA_PARENT_DIR, 'parquet_outputs', SOURCE_DIR)

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Pattern to identify and parse filenames
    FNAME_REGEX = re.compile(r'^(?P<label>base|(?P<intake>\d+)ml)\s+(?P<date>\d{4}-\d{2}-\d{2})')

    # Sensor file names inside each zip
    SENSOR_FILES = {
        'Accelerometer.csv': 'accelerometer',
        'Gyroscope.csv': 'gyroscope',
        'Linear Acceleration.csv': 'linear_acceleration'
    }

    # Collect data per experiment date
    daily_data = {}

    # Walk through DATA_DIR to include subfolders (sources)
    for root, dirs, files in os.walk(DATA_DIR):
        source = SOURCE_DIR
        print(f"Processing source: {source} in {root}")

        for fname in files:
            if not fname.lower().endswith('.zip'):
                continue

            match = FNAME_REGEX.match(fname)
            if not match:
                print(f"Skipping unknown-format file: {fname} in {root}")
                continue

            info = match.groupdict()
            date_key = info['date']
            is_base = 1 if info['label'] == 'base' else 0
            intake = 0 if is_base else int(info['intake'])

            zip_path = os.path.join(root, fname)
            dfs = []

            with zipfile.ZipFile(zip_path, 'r') as z:
                for internal_name, sensor_key in SENSOR_FILES.items():
                    try:
                        with z.open(internal_name) as f:
                            df = pd.read_csv(f)
                    except KeyError:
                        print(f"  - Warning: {internal_name} not found in {fname}")
                        continue

                    # Rename x/y/z columns to include sensor prefix
                    rename_map = {
                        col: f"{sensor_key}_{col}"
                        for col in ['x', 'y', 'z']
                        if col in df.columns
                    }
                    if rename_map:
                        df = df.rename(columns=rename_map)

                    dfs.append(df)

            if not dfs:
                print(f"No sensor data for {fname}, skipping.")
                continue

            # Merge all sensor data on 'time'
            merged = dfs[0]
            for df in dfs[1:]:
                merged = pd.merge(merged, df, on='time', how='outer')

            # Add experiment metadata
            merged['experiment_id'] = date_key
            merged['base'] = is_base
            merged['caffeine_ml'] = intake
            merged['source'] = source

            # Reorder columns
            cols = [
                'experiment_id', 'base', 'caffeine_ml', 'source', 'time'
            ] + [
                c for c in merged.columns
                if c not in ['experiment_id', 'base', 'caffeine_ml', 'source', 'time']
            ]
            merged = merged[cols]

            daily_data.setdefault(date_key, []).append(merged)

    # Write combined Parquet per day into OUTPUT_DIR
    for date_key, frames in daily_data.items():
        combined = pd.concat(frames, ignore_index=True)
        out_fname = f"output-{date_key}.parquet"
        out_path = os.path.join(OUTPUT_DIR, out_fname)
        combined.to_parquet(out_path, index=False, compression='snappy')
        print(f"Wrote {out_path}")

        out_csv = os.path.join(OUTPUT_DIR, f"output-{date_key}.csv")
        combined.to_csv(out_csv, index=False)
        print(f"Wrote {out_csv}")


    # 2) Write master Parquet and master CSV combining all days
    all_frames = [df for frames in daily_data.values() for df in frames]
    if all_frames:
        all_combined = pd.concat(all_frames, ignore_index=True)

        # Master Parquet
        master_parquet = os.path.join(OUTPUT_DIR, 'output-all.parquet')
        all_combined.to_parquet(master_parquet, index=False, compression='snappy')
        print(f"Wrote master Parquet: {master_parquet}")

        # Master CSV
        master_csv = os.path.join(OUTPUT_DIR, 'output-all.csv')
        all_combined.to_csv(master_csv, index=False)
        print(f"Wrote master CSV: {master_csv}")
    else:
        print("No data available to write master files.")

if __name__ == '__main__':
    # Trim all Timo data
    trim()
    
    # Remove all outliers from Timo data
    outlier_analysis()
    
    # Combine all Timo data
    combine()