"""
Configuration for Spacing Predictor
"""


class CFG:
    debug = False
    apex = False
    print_freq = 100
    num_workers = 4
    model_name = "tf_efficientnetv2_s.in21k_ft_in1k"
    size = 512
    scheduler = 'CosineAnnealingWarmRestarts'  # ['ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']
    epochs = 5
    # factor = 0.2  # ReduceLROnPlateau
    # patience = 4  # ReduceLROnPlateau
    # eps = 1e-6  # ReduceLROnPlateau
    # T_max = 10  # CosineAnnealingLR
    T_0 = epochs  # CosineAnnealingWarmRestarts
    lr = 1e-3
    min_lr = 1e-6
    batch_size = 16
    weight_decay = 1e-6
    gradient_accumulation_steps = 1
    max_grad_norm = 1000
    seed = 42

    target_cols = ['PixelSpacingX', 'PixelSpacingY', 'SliceThickness']
    target_size = len(target_cols)
    target_col = 'target'
    n_fold = 5
    trn_fold = [0]
    train = True
    inference = False