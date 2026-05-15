import torch.nn as nn
from .FATM import FATM_net


class FATSSR_net(nn.Module):
    def __init__(self, in_channels, out_channels, NNN):
        super().__init__()
        self.conv_e1 = nn.Conv2d(in_channels, NNN, 3, 1, 1)
        self.conv_e2 = nn.Conv2d(NNN, NNN, 3, 1, 1)
        self.prelu_e = nn.PReLU()

        self.Spe1_FATM = FATM_net(NNN, NNN, 1)
        self.Spe2_FATM = FATM_net(NNN, NNN, 1)

        self.conv_d1 = nn.Conv2d(NNN, out_channels, 3, 1, 1)
        self.conv_d2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.prelu_d = nn.PReLU()

    def forward(self, x):
        x = self.prelu_e(self.conv_e2(self.conv_e1(x)))  # Simplified encoding block
        Spe2 = self.Spe2_FATM(self.Spe1_FATM(x))  # Directly apply FATM layers
        Output_HSI = self.prelu_d(self.conv_d2(self.conv_d1(x + Spe2)))  # Skip connection and decoding

        return Output_HSI
