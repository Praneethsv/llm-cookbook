import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        pooling="max",
        batch_norm=True,
    ) -> None:
        super(CNNBlock, self).__init__()
        self.pooling_layer = nn.MaxPool2d(2) if pooling == "max" else nn.AvgPool2d(2)
        layers = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.ReLU(),
            self.pooling_layer,
        ]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))

        self.cnn_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.cnn_block(x)
