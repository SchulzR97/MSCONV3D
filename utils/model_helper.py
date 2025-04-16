from pathlib import Path
from glob import glob
from model.msconv3d import MSCONV3Ds
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