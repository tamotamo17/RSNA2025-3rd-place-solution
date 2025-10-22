"""
Spacing Predictor Training Script
Train spacing predictor model for PixelSpacing and SliceThickness prediction
"""

import os
import sys
import pandas as pd
import glob
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.spacing.config import CFG
from src.spacing.utils import init_logger, seed_torch
from src.spacing.model import train_loop


def parse_args():
    parser = argparse.ArgumentParser(description='Train Spacing Predictor')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Path to data directory (default: ./data)')
    parser.add_argument('--csv_path', type=str, default='./data/rsna2025_spacing.csv',
                        help='Path to spacing CSV file (default: ./data/rsna2025_spacing.csv)')
    parser.add_argument('--img_dir', type=str, default='./data/extracted_slice_v2',
                        help='Path to image directory (default: ./data/extracted_slice_v2)')
    parser.add_argument('--output_dir', type=str, default='./exp/spacing_exp002',
                        help='Path to output directory (default: ./exp/spacing_exp002)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode (use only 1000 samples)')
    return parser.parse_args()


def main():
    """
    Main training function
    Prepare: 1.train  2.test  3.submission  4.folds
    """
    args = parse_args()
    
    # Setup
    OUTPUT_DIR = args.output_dir + '/'
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    LOGGER = init_logger(log_file=OUTPUT_DIR+'train.log')
    seed_torch(seed=CFG.seed)
    
    # Override debug mode if specified
    if args.debug:
        CFG.debug = True
        LOGGER.info("Debug mode enabled")
    
    # Load data
    LOGGER.info(f"Loading data from {args.csv_path}")
    train = pd.read_csv(args.csv_path)
    
    # Get image paths
    img_pattern = os.path.join(args.img_dir, '*/*.png')
    LOGGER.info(f"Loading images from {img_pattern}")
    img_paths = glob.glob(img_pattern)
    LOGGER.info(f"Found {len(img_paths)} images")
    
    df_img = pd.DataFrame({"file_path": img_paths})
    df_img["SeriesInstanceUID"] = df_img["file_path"].apply(lambda x: x.split(os.sep)[-2])
    
    # Merge
    df_merged = df_img.merge(train, on="SeriesInstanceUID", how="left")
    LOGGER.info(f"Total samples: {len(df_merged)}")
    
    if CFG.debug:
        df_merged = df_merged.sample(n=1000, random_state=CFG.seed).reset_index(drop=True)
        LOGGER.info(f"Debug mode: Using {len(df_merged)} samples")
    
    # Create folds
    from sklearn.model_selection import KFold
    folds = df_merged.copy()
    kf = KFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
    for fold, (train_idx, val_idx) in enumerate(kf.split(folds)):
        folds.loc[val_idx, 'fold'] = fold
    folds['fold'] = folds['fold'].astype(int)
    
    # Training loop
    oof_df = pd.DataFrame()
    for fold in range(CFG.n_fold):
        if fold in CFG.trn_fold:
            _oof_df = train_loop(folds, fold, OUTPUT_DIR, LOGGER)
            oof_df = pd.concat([oof_df, _oof_df])
            LOGGER.info(f"========== fold: {fold} result ==========")
    
    # Save result
    oof_df.to_csv(OUTPUT_DIR+'oof_df.csv', index=False)
    LOGGER.info("Training completed!")


if __name__ == '__main__':
    main()