from model.msconv3d import MSCONV3Ds
from pathlib import Path
from rsp.ml.dataset import TUCRID
from threading import Thread
from datetime import datetime
import time
import rsp.common.drawing as drawing
import rsp.common.color as colors
import cv2 as cv
import torch
import numpy as np

def predict(results:dict, X:torch.Tensor, msconv3d:MSCONV3Ds, tucrid:TUCRID, k:int):
    with torch.no_grad():
        Y = msconv3d(X.to(DEVICE))

    Y = Y.squeeze(0)
    action = Y.argmax().item()
    label = tucrid.labels[action]

    for i, y in enumerate(Y.detach().cpu()):
        results['scores'][i] = y.item()

    if action == results['last_action']:
        results['k'] += 1
    else:
        results['k'] = 1

    if results['k'] >= k:
        results['action'] = action
        results['label'] = label
    else:
        results['action'] = None
        results['label'] = 'undefined'

    results['last_action'] = action

if __name__ == '__main__':
    #region parameter
    CAP_DEVICE = 1
    INPUT_SIZE = (400, 400)
    SEQUENCE_LENGTH = 30
    USE_DEPTH_DATA = False
    NUM_CLASSES = 7
    PRED_I = 15
    K = 1
    TARGET_FRAMERATE = 35
    DATASET_IMAGE_SIZE = (375, 500)

    if torch.cuda.is_available():
        DEVICE = 'cuda'
    elif torch.backends.mps.is_available():
        DEVICE = 'mps'
    else:
        DEVICE = 'cpu'
    #endregion

    #region data
    tucrid = TUCRID(phase='val', load_depth_data=USE_DEPTH_DATA, sequence_length=SEQUENCE_LENGTH, num_classes=NUM_CLASSES)
    #endregion

    #region model
    msconv3d = MSCONV3Ds(use_depth_channel=USE_DEPTH_DATA, sequence_length=SEQUENCE_LENGTH, num_actions=NUM_CLASSES)
    msconv3d.eval()
    msconv3d.to(DEVICE)

    id = 'msconv3d_v1' + ('_rgbd' if USE_DEPTH_DATA else '_rgb')
    msconv3d.load_state_dict(torch.load(f'state_dict/{id}.pt'))
    #endregion

    #region cycle
    cap = cv.VideoCapture(CAP_DEVICE)
    results = {
        'action': 0,
        'last_action': 0,
        'label': None,
        'k': 0,
        'scores': {}
    }
    for i in range(NUM_CLASSES):
        results['scores'][i] = 0.

    X = torch.zeros((1, SEQUENCE_LENGTH, 4 if USE_DEPTH_DATA else 3, INPUT_SIZE[0], INPUT_SIZE[1]), dtype=torch.float32)

    attemps = 0
    i = 0
    framerate = TARGET_FRAMERATE
    while True:
        start = datetime.now()
        ret, frame = cap.read()

        if not ret:
            attemps += 1
            if attemps > 10:
                break
            continue

        fy, fx = DATASET_IMAGE_SIZE[0] / frame.shape[0], DATASET_IMAGE_SIZE[1] / frame.shape[1]

        # upscale if input image is too small
        if fx > 1. or fy > 1.:
            f = np.max([fx, fy])
            frame = cv.resize(frame, (0, 0), fx=f, fy=f)

        img_vis = frame.copy()

        # calculate image section to crop
        fy, fx = DATASET_IMAGE_SIZE[0] / frame.shape[0], DATASET_IMAGE_SIZE[1] / frame.shape[1]
        fy, fx = fy / np.max([fy, fx]), fx / np.max([fy, fx]) 
        new_w = int(np.round(fx * frame.shape[1]))
        new_h = int(np.round(fy * frame.shape[0]))

        sx = frame.shape[1] // 2 - new_w // 2
        ex = sx + new_w
        sy = frame.shape[0] // 2 - new_h // 2
        ey = sy + new_h

        frame = frame[sy:ey, :, :]
        frame = frame[:, sx:ex, :]

        frame = cv.resize(frame, INPUT_SIZE)
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        X[0, :-1] = X[0, 1:].clone()
        X[0, -1] = torch.tensor(frame).permute(2, 0, 1).float() / 255

        if i%PRED_I == 0:
            t = Thread(target=predict, args=(results, X, msconv3d, tucrid, K))
            t.start()

        action = results['action']
        label = results['label']

        img_vis = drawing.add_rectangle(img_vis, (0, 0), (img_vis.shape[1], sy), opacity=0.5, color=colors.LIGHT_RED)
        img_vis = drawing.add_rectangle(img_vis, (0, ey), (img_vis.shape[1], img_vis.shape[0]), opacity=0.5, color=colors.LIGHT_RED)

        img_vis = drawing.add_rectangle(img_vis, (0, sy), (sx, ey), opacity=0.5, color=colors.LIGHT_RED)
        img_vis = drawing.add_rectangle(img_vis, (ex, sy), (img_vis.shape[1], img_vis.shape[0]), opacity=0.5, color=colors.LIGHT_RED)

        action_str = f'A{action:0>3} - {label}' if action is not None else 'None - undefined'
        img_vis = drawing.add_text(img_vis, action_str, (10, 10), width=600, margin=5, background=colors.WHITE, background_opacity=0.7, scale=2, text_thickness=3, vertical_align='center')
        img_vis = drawing.add_text(img_vis, f'{framerate:0.2f} FPS', (img_vis.shape[1]-400, 10), width=350, margin=5, background=colors.WHITE, background_opacity=0.7, text_thickness=3, vertical_align='center', scale=2)
        
        # if results['scores'] is not None:
        #     px = 10
        #     width, height = 200, 50
        #     cnt = len(results['scores'])
        #     for a, score in results['scores'].items():
        #         sy = img_vis.shape[0]-10-(cnt-a)*height
        #         w = int(np.round(score * width))
        #         img_vis = drawing.add_rectangle(img_vis, (px, sy), (px+width, sy+height), color=colors.WHITE, opacity=0.5)
        #         img_vis = drawing.add_text(img_vis, f'A{a:0>3}: {score:0.3f}', (px, sy), width=w, height=height,
        #                                    horizontal_align='left', vertical_align='center',
        #                                    background=colors.CORNFLOWER_BLUE, background_opacity=1., scale=1, text_thickness=2)

        cv.imshow('frame', img_vis)
        if cv.waitKey(1) != -1:
            break

        while (datetime.now() - start).total_seconds() < 1 / TARGET_FRAMERATE:
            time.sleep(0.001)

        framerate += 1e-3 * (1 / (datetime.now() - start).total_seconds() - framerate)

        i += 1
        pass
    #endregion