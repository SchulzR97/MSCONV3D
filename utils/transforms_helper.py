from PIL import Image
import numpy as np
import torch

class ToNumpy():
    def __call__(self, x):
        if isinstance(x, Image.Image):
            return np.array(x)
        elif isinstance(x, torch.Tensor):
            return x.permute(1, 2, 0).numpy()
        else:
            raise TypeError('Input must be PIL.Image or torch.Tensor')