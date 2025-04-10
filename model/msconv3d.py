import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
class MSCONV3Dm(nn.Module):
    class Encoder(nn.Module):
        def __init__(
                self,
                sequence_length,
                p_dropout,
                use_depth_channel
        ):
            super().__init__()
            self.use_depth_channel = use_depth_channel

            self.conv1 = Conv3DBlock(
                in_channels = sequence_length,
                out_channels = 64,
                kernel_size_conv = (2, 7, 7),
                kernel_size_pool = (1, 5, 5),
                stride=None,#(1, 5, 5),
                p_dropout = p_dropout
            )
            self.conv2 = Conv3DBlock(
                in_channels = 64,
                out_channels = 128,
                kernel_size_conv = (2, 5, 5),
                kernel_size_pool = (1, 3, 3),
                stride=None,#(1, 4, 4),
                p_dropout = p_dropout
            )
            self.conv3 = Conv3DBlock(
                in_channels = 192,
                out_channels = 384,
                kernel_size_conv = (2, 3, 3) if self.use_depth_channel else (1, 3, 3),
                kernel_size_pool = (1, 2, 2),
                stride=None,#(1, 3, 3),
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
            self.conv5 = Conv3DBlock(
                in_channels = 1728,
                out_channels = 3456,
                kernel_size_conv = (1, 1, 1),
                kernel_size_pool = (1, 3, 3),
                stride=(1, 2, 2),
                p_dropout = p_dropout
            )

            self.downsample13 = MaxPool3d(kernel_size=(2,5,7), stride=(1,3,3))
            self.downsample14 = MaxPool3d(kernel_size=(2,15,12), stride=(2,6,6))
            self.downsample15 = MaxPool3d(kernel_size=(2,20,18), stride=(2,15,15))
            if self.use_depth_channel:
                self.downsample24 = MaxPool3d(kernel_size=(2,5,4), stride=(2,2,2))
            else:
                self.downsample24 = MaxPool3d(kernel_size=(1,5,4), stride=(1,2,2))

            self.downsample25 = MaxPool3d(kernel_size=(1,7,7), stride=(1,5,5))
            self.downsample35 = MaxPool3d(kernel_size=(1,4,4), stride=(1,2,2))

            self.downsample1e = MaxPool3d(kernel_size=(2,50,50), stride=(2,40,40))
            self.downsample2e = MaxPool3d(kernel_size=(2,15,15) if self.use_depth_channel else (1,15,15), stride=(1,12,12))
            self.downsample3e = MaxPool3d(kernel_size=(1,10,10), stride=(1,4,4))
            self.downsample4e = MaxPool3d(kernel_size=(1,2,2), stride=(1,3,3))

            self.dropout3d = nn.Dropout3d(p=p_dropout)

            self.flatten = nn.Flatten(start_dim = 1)
        
        def forward(self, X):
            assert X.shape[3] == 375, f"Expected input shape [B, C, T, H, W] with H=375, but got {X.shape[3]}"
            assert X.shape[4] == 512, f"Expected input shape [B, C, T, H, W] with W=500, but got {X.shape[4]}"
            assert X.shape[2] == 3 or X.shape[2] == 4, f"Expected input shape [B, C, T, H, W] with C=3 or C=4, but got {X.shape[1]}"
            Y = X

            Y1 = self.conv1(Y)
            Y2 = self.conv2(Y1)

            Y13 = self.downsample13(Y1)
            Y14 = self.downsample14(Y1)
            Y15 = self.downsample15(Y1)
            Y24 = self.downsample24(Y2)
            Y25 = self.downsample25(Y2)

            Y2_cat = torch.cat([Y2, Y13], dim=1)
            Y3 = self.conv3(Y2_cat)
            Y3_cat = torch.cat([Y3, Y14, Y24], dim=1)

            Y35 = self.downsample35(Y3)

            Y4 = self.conv4(Y3_cat)

            Y4_cat = torch.cat([Y4, Y15, Y25, Y35], dim=1)
            Y5 = self.conv5(Y4_cat)

            Y1e = self.downsample1e(Y1)
            Y2e = self.downsample2e(Y2)
            Y3e = self.downsample3e(Y3)
            Y4e = self.downsample4e(Y4)

            Y = torch.cat([Y5, Y1e, Y2e, Y3e, Y4e], dim=1)

            #Y = self.flatten(Y5_cat)

            return Y
        
    class Classfifier(nn.Module):
        def __init__(
                self,
                dim_in:int,
                dim_out:int,
                p_dropout:float
            ):
            super().__init__()

            self.flatten = nn.Flatten(start_dim = 1)
            self.linear = nn.Linear(dim_in * 2, dim_out)
            self.relu = nn.LeakyReLU()
            self.dropout = nn.Dropout(p=p_dropout)
            self.softmax = nn.Softmax(dim=1)

        def forward(self, X):
            Y = X
            Y = self.flatten(Y)
            Y = self.linear(Y)
            Y = self.relu(Y)
            Y = self.dropout(Y)
            Y = self.softmax(Y)

            return Y
        
    class AttentionClassifier(nn.Module):
        def __init__(self, in_channels, num_classes):
            super().__init__()
            
            # Query, Key, Value: einfache 1x1x1 Convs
            self.query_conv = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
            self.key_conv   = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
            self.value_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)

            self.softmax = nn.Softmax(dim=-1)

            # Final classifier head
            self.fc = nn.Sequential(
                nn.Linear(in_channels, num_classes),
                nn.Softmax(dim=1)
            )

        def forward(self, x):  # x: [B, C, T, H, W]
            B, C, T, H, W = x.shape
            N = T * H * W

            # Flatten spatial-temporal dims
            x_flat = x.view(B, C, -1)  # [B, C, N]

            Q = self.query_conv(x).view(B, -1, N).permute(0, 2, 1)  # [B, N, C']
            K = self.key_conv(x).view(B, -1, N)                     # [B, C', N]
            V = self.value_conv(x).view(B, C, N)                    # [B, C, N]

            # Attention: [B, N, N]
            attn = self.softmax(torch.bmm(Q, K))  # Q @ K → scaled optional

            # Weighted sum: [B, C, N]
            out = torch.bmm(V, attn.permute(0, 2, 1))

            # Average over N (optional: could use weighted sum again)
            out = out.mean(dim=-1)  # [B, C]

            out = self.fc(out)
            return out

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

        self.encoder = MSCONV3Dm.Encoder(
            sequence_length=sequence_length,
            p_dropout=p_dropout,
            use_depth_channel=use_depth_channel
        )

        #self.classifier = MSCONV3Dm.AttentionClassifier(5184, num_actions)
        self.classifier = MSCONV3Dm.Classfifier(
            dim_in=5184,
            dim_out=num_actions,
            p_dropout=p_dropout
        )

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, X):
        Y = self.encoder(X)
        Y = self.classifier(Y)
        
        return Y
    