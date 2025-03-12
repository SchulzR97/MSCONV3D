import torch

def normalize_dim_1(input:torch.Tensor):
    sum_dim_1 = input.sum(dim = 1)
    output = torch.zeros(input.shape)
    for i in range(input.shape[1]):
        output[:, i] = input[:, i] / sum_dim_1
    output = torch.round(output, decimals=3)
    return output
