from model.msconv3d import MSCONV3Ds
from rsp.ml.dataset import TUCRID
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
import rsp.ml.metrics as m
import rsp.ml.multi_transforms as multi_transforms
import torch
import os
import utils.tensor_helper as tensor_helper

if __name__ == '__main__':
    #region parameter
    INPUT_SIZE = (400, 400)
    SEQUENCE_LENGTH = 30
    USE_DEPTH_DATA = False
    NUM_CLASSES = 7
    BATCH_SIZE = 4
    
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
    tucrid = TUCRID(
        phase='val',
        load_depth_data=USE_DEPTH_DATA,
        sequence_length=SEQUENCE_LENGTH,
        num_classes=NUM_CLASSES,
        transforms=transforms
    )
    dataloader = DataLoader(tucrid, batch_size=BATCH_SIZE, shuffle=True)
    #endregion

    #region run
    msconv3d = MSCONV3Ds(use_depth_channel=USE_DEPTH_DATA, sequence_length=SEQUENCE_LENGTH, num_actions=NUM_CLASSES)
    msconv3d.to(DEVICE)
    msconv3d.eval()
    id = type(tucrid).__name__ + '/' + type(msconv3d).__name__ + ('_rgbd' if USE_DEPTH_DATA else '_rgb')
    msconv3d.load_state_dict(torch.load(f'state_dict/{id}.pt'))
    out_dir = Path('validate').joinpath(id)
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

        m.plot_confusion_matrix(confusion_matrix, labels=tucrid.labels, save_file_name=out_dir.joinpath('confusion_matrix.png'))
        m.plot_confusion_matrix(confusion_matrix_rel, labels=tucrid.labels, save_file_name=out_dir.joinpath('confusion_matrix_rel.png'))

        prog.set_description(f'validate - accuracy: {TP / (i + 1):.4f} ({TP}/{i+1})')

        # output
        with open(out_dir.joinpath('results.txt'), 'w') as f:
            f.write(
                f'ACC: {TP / (i + 1):.4f}\n'+
                f'TP: {TP}/{i+1}'
            )
    #endregion