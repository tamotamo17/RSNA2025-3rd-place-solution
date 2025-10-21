import os
from pathlib import Path
from tqdm import tqdm
import copy
import ast
from collections import defaultdict, Counter
from glob import glob

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import List, Optional

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


def calculate_intensity_percentiles(file_path: Path, percentiles: List[float] = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]):
    """
    Calculate intensity percentiles for a volume
    
    Args:
        file_path: Path to the .npy file
        percentiles: List of percentiles to calculate
        
    Returns:
        Dictionary with percentile values or None if error
    """
    try:
        # Load volume using memory mapping for efficiency
        volume = np.load(str(file_path), mmap_mode='r')
        
        # Calculate percentiles
        percentile_values = np.percentile(volume, percentiles)
        
        # Create dictionary with percentile names as keys
        result = {f'intensity_p{int(p)}': val for p, val in zip(percentiles, percentile_values)}
        
        # Add some additional statistics
        result['intensity_mean'] = np.mean(volume)
        result['intensity_std'] = np.std(volume)
        result['intensity_min'] = np.min(volume)
        result['intensity_max'] = np.max(volume)
        
        return result
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_batch(df_batch: pd.DataFrame, percentiles: List[float]):
    """Process a batch of rows to calculate intensity features"""
    results = []
    for idx, row in df_batch.iterrows():
        file_path = Path(row['file_name'])
        if file_path.exists():
            intensity_features = calculate_intensity_percentiles(file_path, percentiles)
            if intensity_features:
                intensity_features['index'] = idx
                results.append(intensity_features)
        else:
            print(f"File not found: {file_path}")
    return results


    

if __name__=='__main__':
    root = '/mnt/project/brain/aneurysm/tamoto/RSNA2025/data/npy_float32'
    df = pd.read_csv("/home/tamoto/kaggle/RSNA2025/data/train_add_metadata_v5_roi.csv")

    # df["file_name"] = df.apply(asign_filename_npy, axis=1)
    series_to_remove = [
        "1.2.826.0.1.3680043.8.498.75712554178574230484227682423862727306",  # 読み込みエラー
        "1.2.826.0.1.3680043.8.498.82768897201281605198635077495114055892",  # 読み込みエラー
        "1.2.826.0.1.3680043.8.498.75712554178574230484227682423862727306",  # 読み込みエラー
        "1.2.826.0.1.3680043.8.498.10063454172499468887877935052136698373",  # z_spacingエラー
        "1.2.826.0.1.3680043.8.498.22157965342587174310173115980837533982",  # 読み込みエラー
    ]
    df["sorted_files"] = df["sorted_files"].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else x)
    df = df[~df["SeriesInstanceUID"].isin(series_to_remove)].reset_index(drop=True)
    df['file_name'] = df['SeriesInstanceUID'].apply(lambda x: Path(root)/x/(x+'.npy'))
    
    # Define percentiles to calculate
    percentiles = [0, 1, 5, 10, 90, 95, 99, 100]

    # Process in parallel for efficiency
    print(f"Processing {len(df)} volumes to extract intensity features...")

    # Use ThreadPoolExecutor for I/O bound task
    batch_size = 100
    num_batches = (len(df) + batch_size - 1) // batch_size

    all_results = []

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        
        for i in range(0, len(df), batch_size):
            df_batch = df.iloc[i:i+batch_size]
            future = executor.submit(process_batch, df_batch, percentiles)
            futures.append(future)
        
        # Collect results with progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batches"):
            batch_results = future.result()
            all_results.extend(batch_results)

    # Convert results to DataFrame
    if all_results:
        intensity_df = pd.DataFrame(all_results)
        intensity_df = intensity_df.set_index('index')
        
        # Merge with original dataframe
        df_with_intensity = df.join(intensity_df)
        
        # Fill NaN values for failed calculations
        intensity_columns = [col for col in df_with_intensity.columns if col.startswith('intensity_')]
        for col in intensity_columns:
            df_with_intensity[col] = df_with_intensity[col].fillna(-1)  # Use -1 as indicator of failed calculation
        
        # Save the updated dataframe
        output_path = "/home/tamoto/kaggle/RSNA2025/data/train_add_metadata_v5_with_intensity_roi.csv"
        df_with_intensity.to_csv(output_path, index=False)
        print(f"Saved dataframe with intensity features to: {output_path}")
        
        # Print summary statistics
        print("\nIntensity Feature Summary:")
        print("-" * 50)
        for col in intensity_columns:
            valid_values = df_with_intensity[df_with_intensity[col] != -1][col]
            if len(valid_values) > 0:
                print(f"{col:20s}: mean={valid_values.mean():.2f}, std={valid_values.std():.2f}")
        
        print(f"\nTotal rows: {len(df_with_intensity)}")
        print(f"Successfully processed: {len(df_with_intensity[df_with_intensity['intensity_mean'] != -1])}")
        print(f"Failed: {len(df_with_intensity[df_with_intensity['intensity_mean'] == -1])}")
    else:
        print("No results to save")