from rsp.ml.model import publish_model, load_model
from model.msconv3d import MSCONV3Ds, MSCONV3Dm
from utils.dataset_helper import DATASET_TYPE
from rsp.ml.run import Run
from pathlib import Path
from glob import glob
import torch

def load_model_from_run(run_id, num_actions, use_depth_channel, sequence_lentgh):
    model_dir = Path('runs').joinpath(run_id)

    state_dict_files = [Path(sd_file) for sd_file in glob(f'{model_dir}/state_dict*.pt') if not 'optimizer' in sd_file and 'acc' in sd_file]

    best_acc = 0.
    state_dict_file = None
    for sd_file in state_dict_files:
        s_i = sd_file.stem.index('acc')
        acc = float(sd_file.stem[s_i+4:])

        if acc > best_acc:
            best_acc = acc
            state_dict_file = sd_file

    model = MSCONV3Ds(
        use_depth_channel=use_depth_channel,
        sequence_length=sequence_lentgh,
        num_actions=num_actions,
        p_dropout=0.5
    )
    model.load_state_dict(torch.load(state_dict_file))
    
    return model


if __name__ == '__main__':
    RUN_ID = 'TUCHRI/MSCONV3D'
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
    
    model = load_model_from_run(RUN_ID, NUM_ACTIONS, USE_DEPTH_CHANNEL, SEQUENCE_LENGTH)

    publish_model(
        model=model,
        user_id='SchulzR97',
        model_id=MODEL_ID,
        weights_id=WEIGHTS_ID,
        hf_token=None,
        input_shape=(1, SEQUENCE_LENGTH, 4 if USE_DEPTH_CHANNEL else 3, INPUT_SIZE[0], INPUT_SIZE[1]),
    )

