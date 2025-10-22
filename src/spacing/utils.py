"""
Utility functions for Spacing Predictor
"""

import os
import time
import random
import numpy as np
import torch
from contextlib import contextmanager


def get_score(y_true, y_pred):
    """
    Calculate MAE for each column and overall average
    y_true, y_pred: shape (N, 3) or DataFrame with same columns
    """
    # Convert to ndarray
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # MAE per column
    mae_each = np.mean(np.abs(y_true - y_pred), axis=0)
    # Overall average MAE
    mae_all = np.mean(np.abs(y_true - y_pred))
    
    print(f"MAE(PixelSpacingX, PixelSpacingY, SliceThickness): {mae_each}")
    print(f"MAE(all): {mae_all:.6f}")
    return mae_all


@contextmanager
def timer(name, logger):
    t0 = time.time()
    logger.info(f'[{name}] start')
    yield
    logger.info(f'[{name}] done in {time.time() - t0:.0f} s.')


def init_logger(log_file='train.log'):
    from logging import getLogger, INFO, FileHandler, Formatter, StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True