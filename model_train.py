from rsp.ml.dataset import TUCRID
import rsp.ml.multi_transforms as multi_transforms
from model.msconv3d import MSCONV3Ds
from rsp.ml.run import Run
from torch.utils.data import DataLoader
from pathlib import Path
import rsp.ml.metrics as m
import rsp.common.console as console
import torch
import utils.tensor_helper as tensor_helper
import os
import cv2 as cv

if __name__ == '__main__':
    #region parameter
    INPUT_SIZE = (400, 400)
    SEQUENCE_LENGTH = 30
    USE_DEPTH_DATA = False
    MOVING_AVERAGE_EPOCHS = 50
    BATCHES_PER_EPOCH = 30
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-6
    EPOCHS = 100000
    NUM_CLASSES = 7
    
    if torch.cuda.is_available():
        DEVICE = 'cuda'
    elif torch.backends.mps.is_available():
        DEVICE = 'mps'
    else:
        DEVICE = 'cpu'

    directory_state_dict = Path('state_dict')
    directory_state_dict.mkdir(exist_ok=True)
    #endregion

    #region data
    backgrounds = TUCRID.load_backgrounds(load_depth_data=True)
    for file in os.listdir('backgrounds'):
        img = cv.imread(f'backgrounds/{file}')
        backgrounds.append((img, None))

    tranforms_train = multi_transforms.Compose([
        multi_transforms.ReplaceBackground(
            backgrounds = backgrounds,
            hsv_filter=[(69, 87, 139, 255, 52, 255)],
            p = 0.8,
            rotate=180,
            max_scale=1.5,
            max_noise=0.002
        ),
        multi_transforms.Resize(INPUT_SIZE, auto_crop=False),
        multi_transforms.Color(0.3, p = 0.5),#multi_transforms.Color(0.1, p = 0.2),
        multi_transforms.Brightness(0.5, 1.5),#multi_transforms.Brightness(0.7, 1.3),
        multi_transforms.Satturation(0.5, 1.5),#multi_transforms.Satturation(0.7, 1.3),
        multi_transforms.RandomHorizontalFlip(),
        multi_transforms.GaussianNoise(0.002),
        multi_transforms.RandomCrop(max_scale=1.1),
        multi_transforms.Rotate(max_angle=3),
        multi_transforms.Stack()
    ])
    transforms_val = multi_transforms.Compose([
        multi_transforms.Resize(INPUT_SIZE, auto_crop=False),
        multi_transforms.Stack()
    ])

    ds_train = TUCRID(
        phase='train',
        load_depth_data=USE_DEPTH_DATA,
        sequence_length=SEQUENCE_LENGTH,
        num_classes=NUM_CLASSES,
        transforms=tranforms_train
    )
    sampler_train = ds_train.get_uniform_sampler()
    ds_val = TUCRID(
        phase='val',
        load_depth_data=USE_DEPTH_DATA,
        sequence_length=SEQUENCE_LENGTH,
        num_classes=NUM_CLASSES,
        transforms=transforms_val
    )
    sampler_val = ds_val.get_uniform_sampler()

    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, sampler=sampler_train, num_workers=3, prefetch_factor=2, persistent_workers=True)
    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, sampler=sampler_val, num_workers=3, prefetch_factor=2, persistent_workers=True)
    #endregion

    #region model 
    msconv3d = MSCONV3Ds(use_depth_channel=USE_DEPTH_DATA, sequence_length=SEQUENCE_LENGTH, num_actions=NUM_CLASSES)
    msconv3d.to(DEVICE)
    #endregion

    #region run
    metrics = [
        m.top_1_accuracy,
        m.top_3_accuracy
    ]
    config = {
        m.top_1_accuracy.__name__: {'ymin': 0., 'ymax': 1.},
        m.top_3_accuracy.__name__: {'ymin': 0., 'ymax': 1.},
        'loss': {'ymin': 0.},
    }
    run = Run(
        id=type(ds_train).__name__ + '/' + type(msconv3d).__name__ + ('_rgbd' if USE_DEPTH_DATA else '_rgb'),
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
        results_train = run.train_epoch(dl_train, msconv3d, optimizer, criterion, BATCHES_PER_EPOCH, return_YT=True)
        results_val = run.validate_epoch(dl_val, msconv3d, optimizer, criterion, BATCHES_PER_EPOCH, return_YT=True)

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
        
        # confusion matrix
        results_train['confusion_matrix'] = m.confusion_matrix(results_train['Y'], results_train['T'])
        results_val['confusion_matrix'] = m.confusion_matrix(results_val['Y'], results_val['T'])

        m.plot_confusion_matrix(results_train['confusion_matrix'], labels=ds_train.labels, save_file_name=f'{run.directory_plot}/confusion_matrix_train.jpg')
        m.plot_confusion_matrix(results_val['confusion_matrix'], labels=ds_val.labels, save_file_name=f'{run.directory_plot}/confusion_matrix_val.jpg')

        results_train['confusion_matrix_rel'] = tensor_helper.normalize_dim_1(results_train['confusion_matrix'])
        results_val['confusion_matrix_rel'] = tensor_helper.normalize_dim_1(results_val['confusion_matrix'])

        m.plot_confusion_matrix(results_train['confusion_matrix_rel'], labels=ds_train.labels, save_file_name=f'{run.directory_plot}/confusion_matrix_train_rel.jpg')
        m.plot_confusion_matrix(results_val['confusion_matrix_rel'], labels=ds_val.labels, save_file_name=f'{run.directory_plot}/confusion_matrix_val_rel.jpg')

        # output
        run.save()
        run.plot()
        run.save_state_dict(msconv3d.state_dict())
        run.save_best_state_dict(msconv3d.state_dict(), acc_val_avg)

        # save state dict if average accuracy improved
        if len(run.data['top_1_accuracy']['val']['avg']) >= 2:
            acc_val_avg_prev = run.data['top_1_accuracy']['val']['avg'][-2]
            if acc_val_avg > acc_val_avg_prev:
                sd_file = directory_state_dict.joinpath(f'{run.id}.pt')
                torch.save(msconv3d.state_dict(), sd_file)
                console.success(f'Saved state dict: {sd_file}, epoch: {run.epoch}, acc_val_avg: {acc_val_avg:0.6f}')

        print(f'Epoch: {run.epoch}, acc_train: {acc_train_avg:0.6f}, acc_val: {acc_val_avg:0.6f}')
    #endregion