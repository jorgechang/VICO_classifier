"""CNN Models inspired by
'A Convnet for the 2020s' [https://arxiv.org/abs/2201.03545]

MIT License

Copyright (c) Meta Platforms, Inc. and affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""

# Third party modules
import torch
import torch.nn as nn


def vgg_block_double(in_ch, out_ch, kernel_size=3, padding=1):
    """VGG double block"""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
        nn.GELU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=padding),
        nn.GELU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )


class LayerNorm(nn.Module):
    """Layer Normalization function

    See for more details [https://arxiv.org/pdf/1607.06450.pdf]"""

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        """Performs Layer Normalization forward pass"""
        if self.data_format == "channels_last":
            return nn.F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class customVGG(nn.Module):
    """Custom Convolutional Neural Network"""
    def __init__(self, in_ch, num_classes):
        super().__init__()

        # Init stem layer as in 'A convnet for the 2020s'
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 16, kernel_size=4, stride=4),
            LayerNorm(16, eps=1e-6, data_format="channels_first"),
        )

        # Define VGG block
        self.conv_block1 = vgg_block_double(16, 32)
        self.conv_block2 = vgg_block_double(32, 32)

        # Define layer to extract embedding
        self.embedding_block = nn.Linear(192, 32)

        # Define classification head
        self.fc_layers = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        """Performs forward pass"""
        x = self.stem(x)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x.view(x.size(0), -1)
        x = self.embedding_block(x)
        x = self.fc_layers(x)
        return x
