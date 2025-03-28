from rsp.ml.dataset import TUCRID, TUCHRI, HMDB51, UCF101, Kinetics, UTKinectAction3D
from datasets import load_dataset
from model.msconv3d import MSCONV3Ds
from rsp.ml.run import Run
from torch.utils.data import DataLoader
from pathlib import Path
import torchvision.transforms as transforms
import numpy as np
import rsp.ml.metrics as m
import rsp.common.console as console
import rsp.ml.multi_transforms as multi_transforms
import torch
import torchvision
import utils.transforms_helper as transforms_helper
import os
import cv2 as cv

#region DATASET_TYPES
DATASET_TYPE_TUCRID = 'TUCRID'
DATASET_TYPE_TUCHRI = 'TUCHRI'
DATASET_TYPE_TUCHRI_CS = 'TUCHRI-CS'
DATASET_TYPE_HMDB51 = 'HMDB51'
DATASET_TYPE_UCF101 = 'UCF101'
DATASET_TYPE_KINETICS400 = 'KINETICS400'
DATASET_TYPE_UTKINECTACTION3D = 'UTKinectAction3D'
#endregion

if __name__ == '__main__':
    #region parameter
    INPUT_SIZE = (400, 400)
    USE_DEPTH_DATA = False
    MOVING_AVERAGE_EPOCHS = 0
    BATCHES_PER_EPOCH = 10000000000
    BATCH_SIZE = 8
    LEARNING_RATE = 5e-5
    EPOCHS = 100000
    NUM_WORKERS = 14
    DATASET_TYPE = DATASET_TYPE_UCF101
    DATASET_DIRECTORY = 'datasets'
    FOLD = 1
    
    if torch.cuda.is_available():
        DEVICE = 'cuda'
    elif torch.backends.mps.is_available():
        DEVICE = 'mps'
    else:
        DEVICE = 'cpu'

    USE_DEPTH_DATA = DATASET_TYPE in [DATASET_TYPE_TUCRID, DATASET_TYPE_UTKINECTACTION3D]
    #endregion

    #region data
    if DATASET_TYPE == DATASET_TYPE_TUCRID:
        backgrounds = TUCRID.load_backgrounds(load_depth_data=True)
        for file in os.listdir('backgrounds'):
            img = cv.imread(f'backgrounds/{file}')
            backgrounds.append((img, None))

        transforms_train = multi_transforms.Compose([
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
            sequence_length=30,
            transforms=transforms_train
        )
        sampler_train = ds_train.get_uniform_sampler()
        ds_val = TUCRID(
            phase='val',
            load_depth_data=USE_DEPTH_DATA,
            sequence_length=30,
            transforms=transforms_val
        )
        sampler_val = ds_val.get_uniform_sampler()

        dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, sampler=sampler_train, num_workers=NUM_WORKERS, prefetch_factor=2, persistent_workers=True)
        dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, sampler=sampler_val, num_workers=NUM_WORKERS, prefetch_factor=2, persistent_workers=True)
    elif DATASET_TYPE in [DATASET_TYPE_TUCHRI, DATASET_TYPE_TUCHRI_CS]:
        ds_backgrounds = load_dataset('SchulzR97/backgrounds', split='train')

        backgrounds = []
        for record in ds_backgrounds:
            img = np.array(record['image'])
            f = np.max([375 / img.shape[0], 500 / img.shape[1]])
            img = cv.resize(img, (0, 0), fx=f, fy=f)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

            if img.shape[0] > 375:
                start = (img.shape[0] - 375) // 2
                img = img[start:start+375, :, :]
            if img.shape[1] > 500:
                start = (img.shape[1] - 500) // 2
                img = img[:, start:start+500, :]

            backgrounds.append(img)


        transforms_train = multi_transforms.Compose([
            multi_transforms.ReplaceBackground(
                backgrounds = backgrounds,
                hsv_filter=[
                    (50, 80, 240, 255, 240, 255)
                ],
                p = 0.8,
                rotate=1,
                max_scale=1.1,
                max_noise=0.002
            ),
            multi_transforms.Resize(INPUT_SIZE, auto_crop=False),
            multi_transforms.Color(0.2, p = 0.5),
            multi_transforms.Brightness(0.7, 1.3),
            multi_transforms.Satturation(0.7, 1.3),
            multi_transforms.RandomHorizontalFlip(),
            multi_transforms.GaussianNoise(0.002),
            multi_transforms.RandomCrop(max_scale=1.05),
            multi_transforms.Rotate(max_angle=3),
            multi_transforms.Stack()
        ])
        transforms_val = multi_transforms.Compose([
            multi_transforms.Resize(INPUT_SIZE, auto_crop=False),
            multi_transforms.Stack()
        ])

        ds_train = TUCHRI(
            split='train',
            sequence_length=30,
            transforms=transforms_train,
            cache_dir=Path(DATASET_DIRECTORY).joinpath(DATASET_TYPE) if DATASET_DIRECTORY else None,
            validation_type='cross_subject' if DATASET_TYPE == DATASET_TYPE_TUCHRI_CS else 'default'
        )
        sampler_train = ds_train.get_uniform_sampler()
        ds_val = TUCHRI(
            split='val',
            sequence_length=30,
            transforms=transforms_val,
            cache_dir=Path(DATASET_DIRECTORY).joinpath(DATASET_TYPE) if DATASET_DIRECTORY else None,
            validation_type='cross_subject' if DATASET_TYPE == DATASET_TYPE_TUCHRI_CS else 'default'
        )

        # for _ in range(1000):
        #     i = np.random.randint(len(ds_train))
        #     X, T = ds_train[i]
        #     for x in X:
        #         img = x.permute(1, 2, 0).numpy()
        #         #cv.imshow('img', img)
        #         #cv.waitKey(10)
        #         img = np.array(img*255, dtype=np.uint8)
        #         cv.imwrite('test.png', img)
        #         time.sleep(1)

        dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, sampler=sampler_train, num_workers=NUM_WORKERS, prefetch_factor=2 if NUM_WORKERS > 0 else None, persistent_workers=NUM_WORKERS>0)
        dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, sampler=None, num_workers=NUM_WORKERS, prefetch_factor=2 if NUM_WORKERS > 0 else None, persistent_workers=NUM_WORKERS>0)
    elif DATASET_TYPE == DATASET_TYPE_HMDB51:
        transforms_train = multi_transforms.Compose([
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
        ])

        ds_train = HMDB51(
            split='train',
            fold=FOLD,
            transforms=transforms_train,
            target_size=INPUT_SIZE,
            verbose=False
        )
        ds_val = HMDB51(
            split='val',
            fold=FOLD,
            transforms=transforms_val,
            target_size=INPUT_SIZE,
            verbose=False
        )
        dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, prefetch_factor=2, persistent_workers=True)
        dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, prefetch_factor=2, persistent_workers=True)
    elif DATASET_TYPE == DATASET_TYPE_UCF101:
        USE_DEPTH_DATA = True
        transforms_train = multi_transforms.Compose([
            multi_transforms.Color(0.3, p = 0.5),
            multi_transforms.Brightness(0.7, 1.3),
            multi_transforms.Satturation(0.7, 1.3),
            multi_transforms.RandomHorizontalFlip(),
            multi_transforms.GaussianNoise(0.002),
            multi_transforms.RandomCrop(max_scale=1.05),
            multi_transforms.Rotate(max_angle=3),
            multi_transforms.Stack()
        ])
        transforms_val = multi_transforms.Compose([
            multi_transforms.Stack()
        ])

        ds_train = UCF101(
            split='train',
            fold=FOLD,
            transforms=transforms_train,
            target_size=INPUT_SIZE,
            verbose=False,
            cache_dir=Path(DATASET_DIRECTORY).joinpath('UCF101') if DATASET_DIRECTORY else None
        )
        ds_val = UCF101(
            split='val',
            fold=FOLD,
            transforms=transforms_val,
            target_size=INPUT_SIZE,
            verbose=False,
            cache_dir=Path(DATASET_DIRECTORY).joinpath('UCF101') if DATASET_DIRECTORY else None
        )
        dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, prefetch_factor=2 if NUM_WORKERS > 0 else None, persistent_workers=NUM_WORKERS>0, pin_memory_device=DEVICE if NUM_WORKERS>0 else '', pin_memory=NUM_WORKERS>0)
        dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, prefetch_factor=2 if NUM_WORKERS > 0 else None, persistent_workers=NUM_WORKERS>0, pin_memory_device=DEVICE if NUM_WORKERS>0 else '', pin_memory=NUM_WORKERS>0)

        # for batch_X, batch_T in dl_train:
        #     for X in batch_X:
        #         for x in X.cpu():
        #             img = np.array(x.permute(1, 2, 0).numpy() * 255, dtype=np.uint8)
        #             img_rgb = img[:, :, :3]
        #             img_mask = img[:, :, 3]

        #             cv.imwrite('test_rgb.png', img_rgb)
        #             cv.imwrite('test_mask.png', img_mask)
        #             time.sleep(0.5)
    elif DATASET_TYPE == DATASET_TYPE_UTKINECTACTION3D:      
        transforms_bg = transforms.Compose([
            transforms.Resize((600, 600)),
            transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.5),
            transforms.RandomRotation(180, expand=False),
            transforms.CenterCrop(INPUT_SIZE),
            transforms.RandomGrayscale(p=0.2),
            transforms_helper.ToNumpy()
        ])

        dtd_dataset = torchvision.datasets.DTD(root='DTD', download=True, split='val', transform=transforms_bg)

        transforms_train = multi_transforms.Compose([
            multi_transforms.RemoveBackgroundAI(removed_color=(0, 255, 0)),
            multi_transforms.ReplaceBackground(backgrounds=dtd_dataset, hsv_filter=[(45, 96, 230, 255, 230, 255)]),
            multi_transforms.Color(0.2, p = 0.5),#multi_transforms.Color(0.1, p = 0.2),
            multi_transforms.Brightness(0.8, 1.2),#multi_transforms.Brightness(0.7, 1.3),
            multi_transforms.Satturation(0.8, 1.2),#multi_transforms.Satturation(0.7, 1.3),
            multi_transforms.RandomHorizontalFlip(),
            multi_transforms.GaussianNoise(0.002),
            multi_transforms.RandomCrop(max_scale=1.1),
            multi_transforms.Rotate(max_angle=3),
            multi_transforms.Stack()
        ])
        transforms_val = multi_transforms.Compose([])

        ds_train = UTKinectAction3D(
            split='train',
            transforms=transforms_train,
            sequence_length=20,
            target_size=INPUT_SIZE,
            verbose=False
        )
        ds_val = UTKinectAction3D(
            split='val',
            transforms=transforms_val,
            sequence_length=20,
            target_size=INPUT_SIZE,
            verbose=False
        )

        dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, prefetch_factor=2 if NUM_WORKERS > 0 else None, persistent_workers=NUM_WORKERS>0, timeout=9999)
        dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, prefetch_factor=2 if NUM_WORKERS > 0 else None, persistent_workers=NUM_WORKERS>0, timeout=9999)

        # for batch_X, batch_T in dl_train:
        #     for X, T in zip(batch_X, batch_T):
        #         for x in X:
        #             img_rgb = x.permute(1, 2, 0).numpy()[:, :, :3]
        #             img_depth = x.permute(1, 2, 0).numpy()[:, :, 3]
        #             cv.imshow('rgb', img_rgb)
        #             cv.imshow('depth', img_depth)
        #             cv.waitKey()
    elif DATASET_TYPE == DATASET_TYPE_KINETICS400:
        transforms_train = multi_transforms.Compose([
            multi_transforms.Color(0.1, p = 0.5),#multi_transforms.Color(0.1, p = 0.2),
            multi_transforms.Brightness(0.8, 1.2),#multi_transforms.Brightness(0.7, 1.3),
            multi_transforms.Satturation(0.8, 1.2),#multi_transforms.Satturation(0.7, 1.3),
            multi_transforms.RandomHorizontalFlip(),
            multi_transforms.GaussianNoise(0.002),
            multi_transforms.RandomCrop(max_scale=1.1),
            multi_transforms.Rotate(max_angle=3),
            multi_transforms.Stack()
        ])
        transforms_val = multi_transforms.Compose([
        ])

        ds_train = Kinetics(
            split='train',
            type=400,
            transforms=transforms_train,
            num_threads=4,
            verbose=False
        )
        ds_val = Kinetics(
            split='val',
            type=400,
            transforms=transforms_val,
            num_threads=4,
            verbose=False
        )
        dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, prefetch_factor=2, persistent_workers=True)
        dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, prefetch_factor=2, persistent_workers=True)

    console.success(f'Loaded dataset: {type(ds_train).__name__} ' +
                    f'train: {len(ds_train)} ({len(ds_train)/(len(ds_train)+len(ds_val))*100:0.2f}%), ' +
                    f'val: {len(ds_val)} ({len(ds_val)/(len(ds_train)+len(ds_val))*100:0.2f}%)')

    if len(ds_train.action_labels) > 20:
        action_labels = [f'A{i:0>3}' for i in range(len(ds_train.action_labels))]
    else:
        action_labels = ds_train.action_labels
    #endregion

    #region model 
    msconv3d = MSCONV3Ds(use_depth_channel=USE_DEPTH_DATA, sequence_length=ds_train.sequence_length, num_actions=len(ds_train.action_labels), p_dropout=0.2)
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

    run_id = f'{DATASET_TYPE}/{type(msconv3d).__name__}'
    if DATASET_TYPE == DATASET_TYPE_TUCRID:
        run_id += ('_rgbd' if USE_DEPTH_DATA else '_rgb')
    if DATASET_TYPE in [DATASET_TYPE_HMDB51, DATASET_TYPE_UCF101]:
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