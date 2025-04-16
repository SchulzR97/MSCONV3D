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
from utils.model_helper import load_model_from_run

if __name__ == '__main__':
    #region parameter
    INPUT_SIZE = (400, 400)
    SEQUENCE_LENGTH = 30
    USE_DEPTH_DATA = False
    NUM_CLASSES = 11
    BATCH_SIZE = 4
    RUN_ID = 'TUCHRI/MSCONV3Ds'
    DATASET_DIRECTORY = '/home/schulzr/Documents/datasets/TUCHRI'
    
    if torch.cuda.is_available():
        DEVICE = 'cuda'
    elif torch.backends.mps.is_available():
        DEVICE = 'mps'
    else:
        DEVICE = 'cpu'
    #endregion

    #region data
    transforms = multi_transforms.Compose([
        multi_transforms.Resize(INPUT_SIZE, auto_crop=False),
        multi_transforms.Stack()
    ])
    tuchri = TUCHRI(
        split='val',
        validation_type='default',
        sequence_length=SEQUENCE_LENGTH,
        cache_dir=DATASET_DIRECTORY,
        transforms=multi_transforms.Compose([
            multi_transforms.Resize(INPUT_SIZE, auto_crop=False),
            multi_transforms.Stack()
        ])
    )
    dataloader = DataLoader(tuchri, batch_size=BATCH_SIZE, shuffle=True)
    #endregion

    #region run
    msconv3d = load_model_from_run(RUN_ID, NUM_CLASSES, USE_DEPTH_DATA, SEQUENCE_LENGTH)
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

        m.plot_confusion_matrix(confusion_matrix, labels=tuchri.action_labels, save_file_name=str(out_dir.joinpath('confusion_matrix.png')))
        m.plot_confusion_matrix(confusion_matrix_rel, labels=tuchri.action_labels, save_file_name=str(out_dir.joinpath('confusion_matrix_rel.png')))

        prog.set_description(f'validate - accuracy: {TP / (i + 1):.4f} ({TP}/{i+1})')

        # output
        with open(out_dir.joinpath('results.txt'), 'w') as f:
            f.write(
                f'ACC: {TP / (i + 1):.4f}\n'+
                f'TP: {TP}/{i+1}'
            )
    #endregion