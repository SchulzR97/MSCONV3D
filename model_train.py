from model.msconv3d import MSCONV3Ds
from rsp.ml.run import Run
from pathlib import Path
from utils.dataset_helper import DATASET_TYPE, load_datasets
from torch.utils.data import DataLoader
import numpy as np
import rsp.ml.metrics as m
import rsp.common.console as console
import torch
import cv2 as cv

if __name__ == '__main__':
    #region parameter
    INPUT_SIZE = (400, 400)
    USE_DEPTH_DATA = False
    MOVING_AVERAGE_EPOCHS = 0
    BATCHES_PER_EPOCH = 10000000000
    BATCH_SIZE = 8
    LEARNING_RATE = 5e-5
    P_DROPOUT = 0.2
    EPOCHS = 100000
    NUM_WORKERS = 8
    DATASET = DATASET_TYPE.TUCHRI_CS
    DATASET_DIRECTORY = 'datasets'
    FOLD = 1
    
    if torch.cuda.is_available():
        DEVICE = 'cuda'
    elif torch.backends.mps.is_available():
        DEVICE = 'mps'
    else:
        DEVICE = 'cpu'

    USE_DEPTH_DATA = DATASET in [DATASET_TYPE.TUCRID, DATASET_TYPE.UTKINECTACTION3D]
    #endregion

    #region data
    ds_train, ds_val = load_datasets(
        dataset_type=DATASET,
        input_size=INPUT_SIZE,
        dataset_directory=DATASET_DIRECTORY,
        fold=FOLD
    )

    dl_train = DataLoader(
        ds_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
        persistent_workers=NUM_WORKERS>0,
        timeout=9999)
    dl_val = DataLoader(
        ds_val,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
        persistent_workers=NUM_WORKERS>0,
        timeout=9999)


    if len(ds_train.action_labels) > 20:
        action_labels = [f'A{i:0>3}' for i in range(len(ds_train.action_labels))]
    else:
        action_labels = ds_train.action_labels
    #endregion

    #region model 
    msconv3d = MSCONV3Ds(
        use_depth_channel=USE_DEPTH_DATA, 
        sequence_length=ds_train.sequence_length, 
        num_actions=len(ds_train.action_labels), 
        p_dropout=P_DROPOUT
    )
    msconv3d.to(DEVICE)
    #endregion

    #region run
    metrics = [
        m.top_1_accuracy,
        m.top_3_accuracy
    ]
    config = {
        'loss': {'ymin': 0.},
        #m.top_1_accuracy.__name__: {'ymin': 0., 'ymax': 1.},
    }

    run_id = f'{DATASET}/{type(msconv3d).__name__}'
    if DATASET == DATASET_TYPE.TUCRID:
        run_id += ('_rgbd' if USE_DEPTH_DATA else '_rgb')
    if DATASET in [DATASET_TYPE.HMDB51, DATASET_TYPE.UCF101]:
        if FOLD is not None:
            run_id += f'_fold{FOLD}'   
    run = Run(
        id= run_id,
        moving_average_epochs=MOVING_AVERAGE_EPOCHS,
        metrics=metrics,
        device=DEVICE,
        ignore_outliers_in_chart_scaling=True,
        config=config
    )
    run.load_best_state_dict(msconv3d)
    run.recalculate_moving_average()
    run.plot()

    optimizer = torch.optim.Adam(msconv3d.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.functional.binary_cross_entropy
    #endregion

    #region train cycle
    while run.epoch < EPOCHS:
        # compute
        results_train = run.train_epoch(dl_train, msconv3d, optimizer, criterion, BATCHES_PER_EPOCH)
        results_val = run.validate_epoch(dl_val, msconv3d, optimizer, criterion, BATCHES_PER_EPOCH)

        acc_train_avg, acc_val_avg = run.get_avg(m.top_1_accuracy.__name__, 'train'), run.get_avg(m.top_1_accuracy.__name__, 'val')
        acc_train, acc_val = run.get_val(m.top_1_accuracy.__name__, 'train'), run.get_avg(m.top_1_accuracy.__name__, 'val')

        if acc_train < 0.8 * acc_train_avg:
            console.warn(f'Accuracy train dropped: {acc_train:0.6f} -> load best state dict')
            run.load_best_state_dict(msconv3d)
            continue
        if acc_val < 0.8 * acc_val_avg:
            console.warn(f'Accuracy val dropped: {acc_val:0.6f} -> load best state dict')
            run.load_best_state_dict(msconv3d)
            continue
        
        # output
        run.save()
        run.plot()
        run.save_state_dict(msconv3d.state_dict())
        run.save_best_state_dict(msconv3d.state_dict(), acc_val_avg)

        print(f'Epoch: {run.epoch}, acc_train: {acc_train_avg:0.6f}, acc_val: {acc_val_avg:0.6f}')
    #endregion