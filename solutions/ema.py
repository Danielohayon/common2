def update_model_ema(cfg, num_gpus, model, model_ema, cur_epoch, cur_iter):
    """Update exponential moving average (ema) of model weights."""
    update_period = cfg.TRAIN.EMA_UPDATE_PERIOD
    if update_period is None or update_period == 0 or cur_iter % update_period != 0:
        return
    # Adjust alpha to be fairly independent of other parameters
    total_batch_size = num_gpus * cfg.DATA.BATCH_SIZE
    adjust = total_batch_size / cfg.TRAIN.EPOCHS * update_period
    # print('ema adjust', adjust)
    alpha = min(1.0, cfg.TRAIN.EMA_ALPHA * adjust)
    # During warmup simply copy over weights instead of using ema
    alpha = 1.0 if cur_epoch < cfg.TRAIN.WARMUP_EPOCHS else alpha
    # Take ema of all parameters (not just named parameters)
    params = model.state_dict()
    for name, param in model_ema.state_dict().items():
        param.copy_(param * (1.0 - alpha) + params[name] * alpha)