from rsp.ml.model import publish_model, load_model
from model.msconv3d import MSCONV3Ds, MSCONV3Dm
from utils.dataset_helper import DATASET_TYPE
from rsp.ml.run import Run
from pathlib import Path
from glob import glob
import torch
from utils.model_helper import load_model_from_run

if __name__ == '__main__':
    RUN_ID = 'TUCHRI/MSCONV3Ds'
    MODEL_ID = 'MSCONV3Ds'
    NUM_ACTIONS = 11
    USE_DEPTH_CHANNEL = False
    SEQUENCE_LENGTH = 30
    INPUT_SIZE = (400, 400)

    dataset = RUN_ID.split('/')[0]
    
    if dataset == DATASET_TYPE.TUCHRI:
        WEIGHTS_ID = 'TUC-HRI'
    elif dataset == DATASET_TYPE.TUCHRI_CS:
        WEIGHTS_ID = 'TUC-HRI-CS'
    elif dataset == DATASET_TYPE.TUCRID:
        WEIGHTS_ID = 'TUC-RID'
    elif dataset == DATASET_TYPE.HMDB51:
        WEIGHTS_ID = 'HMDB51'
    elif dataset == DATASET_TYPE.UCF101:
        WEIGHTS_ID = 'UCF101'
    elif dataset == DATASET_TYPE.KINETICS400:
        WEIGHTS_ID = 'KINETICS400'
    elif dataset == DATASET_TYPE.UTKINECTACTION3D:
        WEIGHTS_ID = 'UTKINECTACTION3D'
    else:
        WEIGHTS_ID = RUN_ID.split('/')[0] 

    WEIGHTS_ID += '-LAB'   
    
    model = load_model_from_run(RUN_ID, NUM_ACTIONS, USE_DEPTH_CHANNEL, SEQUENCE_LENGTH)

    publish_model(
        model=model,
        user_id='SchulzR97',
        model_id=MODEL_ID,
        weights_id=WEIGHTS_ID,
        hf_token=None,
        input_shape=(1, SEQUENCE_LENGTH, 4 if USE_DEPTH_CHANNEL else 3, INPUT_SIZE[0], INPUT_SIZE[1]),
    )

