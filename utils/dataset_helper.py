import rsp.common.console as console
import rsp.ml.multi_transforms as multi_transforms
import numpy as np
import cv2 as cv
import torchvision.transforms as transforms
import torchvision
from utils import transforms_helper
from pathlib import Path
from rsp.ml.dataset import TUCHRI, UCF101, HMDB51, Kinetics, UTKinectAction3D
from datasets import load_dataset
from glob import glob

#region DATASET_TYPE
class DATASET_TYPE():
    TUCRID = 'TUCRID'
    TUCHRI = 'TUCHRI'
    TUCHRI_CS = 'TUCHRI-CS'
    HMDB51 = 'HMDB51'
    UCF101 = 'UCF101'
    KINETICS400 = 'KINETICS400'
    UTKINECTACTION3D = 'UTKinectAction3D'
#endregion

#region data
def load_datasets(dataset_type, input_size, dataset_directory, fold, additional_backgrounds_dir):
    if dataset_type in [DATASET_TYPE.TUCHRI, DATASET_TYPE.TUCHRI_CS]:
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

        if additional_backgrounds_dir is not None:
            bg_files = glob(f'{additional_backgrounds_dir}/*_color.jpg')
            
            for bg_file in bg_files:
                img = cv.imread(bg_file)
                img = cv.resize(img, (500, 375))
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
            multi_transforms.Resize(input_size, auto_crop=False),
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
            multi_transforms.Resize(input_size, auto_crop=False),
            multi_transforms.Stack()
        ])

        ds_train = TUCHRI(
            split='train',
            sequence_length=30,
            transforms=transforms_train,
            cache_dir=Path(dataset_directory).joinpath(dataset_type) if dataset_directory else None,
            validation_type='cross_subject' if dataset_type == DATASET_TYPE.TUCHRI_CS else 'default'
        )
        sampler_train = ds_train.get_uniform_sampler()
        ds_val = TUCHRI(
            split='val',
            sequence_length=30,
            transforms=transforms_val,
            cache_dir=Path(dataset_directory).joinpath(dataset_type) if dataset_directory else None,
            validation_type='cross_subject' if dataset_type == DATASET_TYPE.TUCHRI_CS else 'default'
        )

        # for _ in range(1000):
        #     i = np.random.randint(len(ds_train))
        #     X, T = ds_train[i]
        #     for x in X:
        #         img = x.permute(1, 2, 0).numpy()
        #         cv.imshow('img', img)
        #         cv.waitKey(10)
        #         #img = np.array(img*255, dtype=np.uint8)
        #         #cv.imwrite('test.png', img)
        #         #time.sleep(1)

    elif dataset_type == DATASET_TYPE.HMDB51:
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
            fold=fold,
            transforms=transforms_train,
            target_size=input_size,
            verbose=False
        )
        ds_val = HMDB51(
            split='val',
            fold=fold,
            transforms=transforms_val,
            target_size=input_size,
            verbose=False
        )
    elif dataset_type == DATASET_TYPE.UCF101:
        transforms_train = multi_transforms.Compose([
            multi_transforms.Color(0.3, p = 0.5),#multi_transforms.Color(0.1, p = 0.2),
            multi_transforms.Brightness(0.7, 1.3),#multi_transforms.Brightness(0.7, 1.3),
            multi_transforms.Satturation(0.7, 1.3),#multi_transforms.Satturation(0.7, 1.3),
            multi_transforms.RandomHorizontalFlip(),
            multi_transforms.GaussianNoise(0.002),
            multi_transforms.RandomCrop(max_scale=1.05),
            multi_transforms.Rotate(max_angle=3),
            multi_transforms.Stack()
        ])
        transforms_val = multi_transforms.Compose([])

        ds_train = UCF101(
            split='train',
            fold=fold,
            transforms=transforms_train,
            target_size=input_size,
            verbose=False,
            cache_dir=Path(dataset_directory).joinpath('UCF101') if dataset_directory else None,
            load_person_masks=True
        )
        ds_val = UCF101(
            split='val',
            fold=fold,
            transforms=transforms_val,
            target_size=input_size,
            verbose=False,
            cache_dir=Path(dataset_directory).joinpath('UCF101') if dataset_directory else None,
            load_person_masks=True
        )
        
        # for batch_X, batch_T in dl_train:
        #     for X in batch_X:
        #         for x in X.cpu():
        #             img = np.array(x.permute(1, 2, 0).numpy() * 255, dtype=np.uint8)
        #             img_rgb = img[:, :, :3]
        #             img_mask = img[:, :, 3]

        #             cv.imwrite('test_rgb.png', img_rgb)
        #             cv.imwrite('test_mask.png', img_mask)
        #             time.sleep(0.5)
    elif dataset_type == DATASET_TYPE.UTKINECTACTION3D:      
        transforms_bg = transforms.Compose([
            transforms.Resize((600, 600)),
            transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.5),
            transforms.RandomRotation(180, expand=False),
            transforms.CenterCrop(input_size),
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
            target_size=input_size,
            verbose=False
        )
        ds_val = UTKinectAction3D(
            split='val',
            transforms=transforms_val,
            sequence_length=20,
            target_size=input_size,
            verbose=False
        )

        # for batch_X, batch_T in dl_train:
        #     for X, T in zip(batch_X, batch_T):
        #         for x in X:
        #             img_rgb = x.permute(1, 2, 0).numpy()[:, :, :3]
        #             img_depth = x.permute(1, 2, 0).numpy()[:, :, 3]
        #             cv.imshow('rgb', img_rgb)
        #             cv.imshow('depth', img_depth)
        #             cv.waitKey()
    elif dataset_type == DATASET_TYPE.KINETICS400:
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

    console.success(f'Loaded dataset: {dataset_type} ' +
                    f'train: {len(ds_train)} ({len(ds_train)/(len(ds_train)+len(ds_val))*100:0.2f}%), ' +
                    f'val: {len(ds_val)} ({len(ds_val)/(len(ds_train)+len(ds_val))*100:0.2f}%)')
    return ds_train, ds_val