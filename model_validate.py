from model.msconv3d import MSCONV3Ds
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from rsp.ml.dataset import TUCHRI
import rsp.ml.metrics as m
import rsp.ml.multi_transforms as multi_transforms
import torch
import os
import utils.tensor_helper as tensor_helper
from utils.dataset_helper import DATASET_TYPE, load_datasets
from utils.model_helper import load_model_from_run

if __name__ == '__main__':
    #region parameter
    INPUT_SIZE = (400, 400)
    USE_DEPTH_DATA = False
    BATCH_SIZE = 4
    #RUN_ID = 'TUCHRI/MSCONV3Ds'
    RUN_ID = 'UCF101/MSCONV3Ds'
    DATASET = DATASET_TYPE.UCF101
    DATASET_DIRECTORY = '/home/schulzr/Documents/datasets'
    FOLD = 1

    RUN_ID = DATASET
    if DATASET in [DATASET_TYPE.UCF101] and FOLD is not None:
        RUN_ID += f'_fold{FOLD}'
    RUN_ID += '/MSCONV3Ds'
    
    if torch.cuda.is_available():
        DEVICE = 'cuda'
    elif torch.backends.mps.is_available():
        DEVICE = 'mps'
    else:
        DEVICE = 'cpu'
    #endregion

    #region data
    ds_train, ds_val = load_datasets(
        dataset_type=DATASET,
        input_size=INPUT_SIZE,
        dataset_directory=DATASET_DIRECTORY,
        fold=FOLD,
        additional_backgrounds_dir=None
    )
    dataloader = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=True)

    if len(ds_val.action_labels) <= 20:
        action_labels = ds_val.action_labels
    else:
        action_labels = [f'A{i:0>3}' for i in range(len(ds_val.action_labels))]
    #endregion

    #region run
    msconv3d = load_model_from_run(RUN_ID, len(ds_val.action_labels), USE_DEPTH_DATA, ds_val.sequence_length)
    msconv3d.to(DEVICE)
    msconv3d.eval()
    out_dir = Path('validate').joinpath(RUN_ID)
    os.makedirs(out_dir, exist_ok=True)
    #endregion

    #region cycle
    TP = 0
    confusion_matrix = None
    confusion_matrix_rel = None
    prog = tqdm(dataloader, desc='validate')
    i = 0
    for X, T in prog:
        i += X.shape[0]
        X = X.to(DEVICE)
        T = T.to(DEVICE)
        with torch.no_grad():
            Y = msconv3d(X)

        if confusion_matrix is None:
            confusion_matrix = m.confusion_matrix(Y, T)
        else:
            confusion_matrix += m.confusion_matrix(Y, T)
        confusion_matrix_rel = tensor_helper.normalize_dim_1(confusion_matrix)

        X = X.detach().to('cpu')
        Y = Y.detach().to('cpu')
        T = T.detach().to('cpu')

        true_action = T.argmax(dim=1)
        predicted_action = Y.argmax(dim=1)

        TP += (true_action == predicted_action).sum().item()

        m.plot_confusion_matrix(confusion_matrix, labels=action_labels, save_file_name=str(out_dir.joinpath('confusion_matrix.png')))
        m.plot_confusion_matrix(confusion_matrix_rel, labels=action_labels, save_file_name=str(out_dir.joinpath('confusion_matrix_rel.png')))

        prog.set_description(f'validate - accuracy: {TP / (i + 1):.4f} ({TP}/{i+1})')

        # output
        with open(out_dir.joinpath('results.txt'), 'w') as f:
            f.write(
                f'ACC: {TP / (i + 1):.4f}\n'+
                f'TP: {TP}/{i+1}'
            )
    #endregion