import torch
import torch.nn as nn


class Upsample(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        # self.tranConv = nn.ConvTranspose3d(1, 1, 2, stride=2)

    def forward(self, x: torch.Tensor):
        return self.upsample(x)


class Downsample(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = nn.MaxPool3d(2)

    def forward(self, x: torch.Tensor):
        return self.maxpool(x)


class Conv3d(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        norm_groups: int = 1,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(
                input_channels,
                hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.GroupNorm(norm_groups, hidden_channels),
            nn.Conv3d(
                hidden_channels,
                hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.GroupNorm(norm_groups, hidden_channels),
            nn.ReLU(),
            nn.Conv3d(
                hidden_channels,
                hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.GroupNorm(norm_groups, hidden_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class Conv3dResidual(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        norm_groups: int = 1,
    ):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv3d(
                input_channels,
                hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.GroupNorm(norm_groups, hidden_channels),
        )

        self.block2 = nn.Sequential(
            nn.Conv3d(
                input_channels,
                hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.GroupNorm(norm_groups, hidden_channels),
            nn.ReLU(),
            nn.Conv3d(
                hidden_channels,
                hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.GroupNorm(norm_groups, hidden_channels),
        )

        self.activation = nn.ReLU()

    def forward(self, x):
        a = self.block1(x)
        return self.activation(a + self.block2(a) + x)


class Unet3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_conv = nn.Conv3d(4, 16, kernel_size=3, padding=1)
        self.layer_1_down = Conv3d(16, 32)
        self.layer_2_down = Conv3d(32, 64)
        self.layer_3_down = Conv3d(64, 128)
        self.layer_4_down = Conv3d(128, 256)
        self.bridge = Conv3d(256, 256)
        self.layer_4_up = Conv3d(256, 128)
        self.layer_3_up = Conv3d(128, 64)
        self.layer_2_up = Conv3d(64, 32)
        self.layer_1_up = Conv3d(32, 16)
        self.output = nn.Sequential(
            nn.Conv3d(16, 4, kernel_size=1),
            nn.Softmax(dim=1),
        )

        self.downsample = Downsample()
        self.upsample = Upsample()
        self.dropout_5 = nn.Dropout3d(0.5)
        self.dropout_3 = nn.Dropout3d(0.3)
        self.dropout_1 = nn.Dropout3d(0.1)

    # v1
    # def forward(self, x):
    #     x = self.input_conv(x)
    #     l1 = self.layer_1_down(x)
    #     l2 = self.layer_2_down(self.downsample(l1))
    #     l3 = self.layer_3_down(self.downsample(l2))
    #     l4 = self.layer_4_down(self.downsample(l3))
    #     x = self.bridge(self.downsample(l4))
    #     x = self.layer_4_up(self.upsample(x) + l4)
    #     x = self.layer_3_up(self.upsample(x) + l3)
    #     x = self.layer_2_up(self.upsample(x) + l2)
    #     x = self.layer_1_up(self.upsample(x) + l1)
    #     x = self.output(x)
    #     return x

    # v2
    def forward(self, x):
        x = self.input_conv(x)
        l1 = self.layer_1_down(x)
        l2 = self.layer_2_down(self.dropout_1(self.downsample(l1)))
        l3 = self.layer_3_down(self.dropout_3(self.downsample(l2)))
        l4 = self.layer_4_down(self.dropout_5(self.downsample(l3)))
        x = self.bridge(self.downsample(l4))
        x = self.layer_4_up(self.upsample(x) + l4)
        x = self.layer_3_up(self.upsample(x) + l3)
        x = self.layer_2_up(self.upsample(x) + l2)
        x = self.layer_1_up(self.upsample(x) + l1)
        x = self.output(x)
        return x


if __name__ == "__main__":
    model = Unet3D()
    print(model)
    # x = torch.randn(1, 4, 240, 240, 155)
    x = torch.randn(1, 4, 128, 128, 128)
    print(model(x).shape)

    from torchinfo import summary

    batch_size = 1
    summary(model, input_size=(batch_size, 4, 128, 128, 128))
