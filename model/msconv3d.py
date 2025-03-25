import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(
            self,
            d_q:int = 2,
            d_k:int = 2,
            d_v:int = 4,
            embed_dim:int = 3
    ):
        super().__init__()

        self.d_q = d_q
        self.d_k = d_k
        self.d_v = d_v
        self.embed_dim = embed_dim

        self.W_q = nn.Parameter(torch.randn(embed_dim, d_q))
        self.W_k = nn.Parameter(torch.randn(embed_dim, d_k))
        self.W_v = nn.Parameter(torch.randn(embed_dim, d_v))

    def forward(self, X):
        Z = []
        # iterate over batch_size
        for x in X:
            Q = x @ self.W_q
            K = x @ self.W_k
            V = x @ self.W_v

            omega = Q @ K.T                                 # omega ...unnormalized attention weights
            alpha = F.softmax(omega / self.d_k**0.5, dim=0) # alpha ...normalized attention weights
            z = alpha @ V                                   # z ...context vector -> attention-weighted version of original query input x_i
            Z.append(z)
        
        Z = torch.stack(Z)
        return Z

class MultiHeadSelfAttention(nn.Module):
    def __init__(
            self,
            num_heads:int,
            d_q:int = 2,
            d_k:int = 2,
            d_v:int = 4,
            embed_dim:int = 3
    ):
        super().__init__()

        self.num_heads = num_heads
        self.d_q = d_q
        self.d_k = d_k
        self.d_v = d_v

        self.heads = nn.ModuleList([SelfAttention(d_q, d_k, d_v, embed_dim) for _ in range(num_heads)])

    def forward(self, X):
        return torch.cat([head(X) for head in self.heads], dim=-1)

class MaxPool3d(nn.Module):
    def __init__(
            self,
            kernel_size:tuple[int, int, int],
            stride:tuple[int, int, int]
    ):
        super(MaxPool3d, self).__init__()

        self.maxPool3d = torch.nn.MaxPool3d(kernel_size=kernel_size, stride=stride)
    
    def __call__(self, input_tensor:torch.Tensor):
        device = input_tensor.device.type
        if device == 'mps':
            input_tensor = input_tensor.to('cpu')
        output = self.maxPool3d(input_tensor)
        if device == 'mps':
            output = output.to(device)

        return output

class Conv3DBlock(nn.Module):
    def __init__(
            self,
            in_channels:int,
            out_channels:int,
            kernel_size_conv:tuple[int, int, int],
            kernel_size_pool:tuple[int, int, int],
            stride:tuple[int, int, int],
            padding_conv:int = 0,
            p_dropout:float = 0.5
    ):
        super(Conv3DBlock, self).__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size_conv, padding=padding_conv)
        self.pool = MaxPool3d(kernel_size=kernel_size_pool, stride=stride)
        self.dropout = nn.Dropout3d(p_dropout)
        self.batchnorm = nn.BatchNorm3d(out_channels)
        self.relu = nn.LeakyReLU()
    
    def forward(self, X):
        Y = self.conv(X)
        Y = self.pool(Y)
        Y = self.batchnorm(Y)
        Y = self.dropout(Y)
        Y = self.relu(Y)

        return Y

class MSCONV3Ds(nn.Module):
    def __init__(
            self,
            use_depth_channel:bool,
            sequence_length = 30,
            num_actions:int = 10,
            p_dropout:float = 0.5
        ):
        super().__init__()

        self.sequence_length = sequence_length
        self.num_actions = num_actions
        self.use_depth_channel = use_depth_channel

        self.conv1 = Conv3DBlock(
            in_channels = sequence_length,
            out_channels = 64,
            kernel_size_conv = (2, 7, 7),
            kernel_size_pool = (1, 7, 7),
            stride=(1, 5, 5),
            p_dropout = p_dropout
        )
        self.conv2 = Conv3DBlock(
            in_channels = 64,
            out_channels = 128,
            kernel_size_conv = (2, 5, 5),
            kernel_size_pool = (1, 5, 5),
            stride=(1, 3, 3),
            p_dropout = p_dropout
        )
        self.conv3 = Conv3DBlock(
            in_channels = 192,
            out_channels = 384,
            kernel_size_conv = (2, 5, 5) if self.use_depth_channel else (1, 5, 5),
            kernel_size_pool = (1, 3, 3),
            stride=(1, 2, 2),
            p_dropout = p_dropout
        )
        self.conv4 = Conv3DBlock(
            in_channels = 576,
            out_channels = 1152,
            kernel_size_conv = (1, 3, 3),
            kernel_size_pool = (1, 2, 2),
            stride=(1, 2, 2),
            p_dropout = p_dropout
        )

        self.downsample13 = MaxPool3d(kernel_size=(2,7,7), stride=(1,3,3))
        self.downsample14 = MaxPool3d(kernel_size=(2,9,9), stride=(2,8,8))
        if self.use_depth_channel:
            self.downsample24 = MaxPool3d(kernel_size=(2,7,7), stride=(2,2,2))
        else:
            self.downsample24 = MaxPool3d(kernel_size=(1,7,7), stride=(1,2,2))

        self.downsample1e = MaxPool3d(kernel_size=(2,28,28), stride=(2,21,21))
        self.downsample2e = MaxPool3d(kernel_size=(2,9,9) if self.use_depth_channel else (1,9,9), stride=(1,6,6))
        self.downsample3e = MaxPool3d(kernel_size=(1,5,5), stride=(1,2,2))

        self.dropout3d = nn.Dropout3d(p=p_dropout)

        self.flatten = nn.Flatten(start_dim = 1)

        self.readout = nn.Linear(15552, num_actions)

        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p = p_dropout)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, X):
        Y = X

        Y1 = self.conv1(Y)
        Y2 = self.conv2(Y1)

        Y13 = self.downsample13(Y1)
        Y14 = self.downsample14(Y1)
        Y24 = self.downsample24(Y2)

        Y2_cat = torch.cat([Y2, Y13], dim=1)
        Y3 = self.conv3(Y2_cat)
        Y3_cat = torch.cat([Y3, Y14, Y24], dim=1)

        Y4 = self.conv4(Y3_cat)
        
        Y1e = self.downsample1e(Y1)
        Y2e = self.downsample2e(Y2)
        Y3e = self.downsample3e(Y3)

        Y4_cat = torch.cat([Y4, Y1e, Y2e, Y3e], dim=1)

        Y = self.flatten(Y4_cat)
        
        Y = self.readout(Y)

        Y = self.softmax(Y)
        
        return Y
    
class MSCONV3Ds_atten(nn.Module):
    def __init__(
            self,
            use_depth_channel:bool,
            sequence_length = 30,
            num_actions:int = 10
        ):
        super().__init__()

        self.sequence_length = sequence_length
        self.num_actions = num_actions
        self.use_depth_channel = use_depth_channel

        self.conv1 = Conv3DBlock(
            in_channels = sequence_length,
            out_channels = 64,
            kernel_size_conv = (2, 7, 7),
            kernel_size_pool = (1, 7, 7),
            stride=(1, 5, 5),
            p_dropout = 0.2
        )
        self.conv2 = Conv3DBlock(
            in_channels = 64,
            out_channels = 128,
            kernel_size_conv = (2, 5, 5),
            kernel_size_pool = (1, 5, 5),
            stride=(1, 3, 3),
            p_dropout = 0.2
        )
        self.conv3 = Conv3DBlock(
            in_channels = 192,
            out_channels = 384,
            kernel_size_conv = (2, 5, 5) if self.use_depth_channel else (1, 5, 5),
            kernel_size_pool = (1, 3, 3),
            stride=(1, 2, 2),
            p_dropout = 0.2
        )
        self.conv4 = Conv3DBlock(
            in_channels = 576,
            out_channels = 1152,
            kernel_size_conv = (1, 3, 3),
            kernel_size_pool = (1, 2, 2),
            stride=(1, 2, 2),
            p_dropout = 0.2
        )

        self.downsample13 = MaxPool3d(kernel_size=(2,7,7), stride=(1,3,3))
        self.downsample14 = MaxPool3d(kernel_size=(2,9,9), stride=(2,8,8))
        if self.use_depth_channel:
            self.downsample24 = MaxPool3d(kernel_size=(2,7,7), stride=(2,2,2))
        else:
            self.downsample24 = MaxPool3d(kernel_size=(1,7,7), stride=(1,2,2))

        self.downsample1e = MaxPool3d(kernel_size=(2,28,28), stride=(2,21,21))
        self.downsample2e = MaxPool3d(kernel_size=(2,9,9) if self.use_depth_channel else (1,9,9), stride=(1,6,6))
        self.downsample3e = MaxPool3d(kernel_size=(1,5,5), stride=(1,2,2))

        self.dropout3d = nn.Dropout3d(p=0.2)

        self.layernorm1 = nn.LayerNorm(9)
        self.layernorm2 = nn.LayerNorm((1728, 25))
        self.attention = MultiHeadSelfAttention(num_heads=4, d_q=2, d_k=2, d_v=4, embed_dim=9)

        self.flatten = nn.Flatten(start_dim = 1)

        self.readout = nn.Linear(43200, num_actions)

        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p = 0.2)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, X):
        Y = X

        Y1 = self.conv1(Y)
        Y2 = self.conv2(Y1)

        Y13 = self.downsample13(Y1)
        Y14 = self.downsample14(Y1)
        Y24 = self.downsample24(Y2)

        Y2_cat = torch.cat([Y2, Y13], dim=1)
        Y3 = self.conv3(Y2_cat)
        Y3_cat = torch.cat([Y3, Y14, Y24], dim=1)

        Y4 = self.conv4(Y3_cat)
        
        Y1e = self.downsample1e(Y1)
        Y2e = self.downsample2e(Y2)
        Y3e = self.downsample3e(Y3)

        Y4_cat = torch.cat([Y4, Y1e, Y2e, Y3e], dim=1)

        Y = torch.reshape(Y4_cat, (Y4_cat.shape[0], Y4_cat.shape[1], Y4_cat.shape[2]*Y4_cat.shape[3]*Y4_cat.shape[4]))
        Y = self.layernorm1(Y)
        Ya = self.attention(Y)
        Y = torch.cat([Y, Ya], dim=2)
        Y = self.layernorm2(Y)
        Y = self.flatten(Y)
        
        Y = self.readout(Y)

        Y = self.softmax(Y)
        
        return Y