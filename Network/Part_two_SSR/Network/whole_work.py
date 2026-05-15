import torch.nn as nn
from .FATSSR import FATSSR_net
import torch.nn.functional as F
import torch


class W_SpectralFormer(nn.Module):
    def __init__(self):
        super(W_SpectralFormer, self).__init__()

        self.SSR = FATSSR_net(in_channels=9, out_channels=14, NNN=64)
        self.srf = SRF()
        model = PSF(15)
        file_path = r'L:/SSR/ALL/experiments_all/PSF/epoch_best.pth'
        psf = torch.load(file_path)['blur_down_conv.psf.weight']
        state_dict = {'psf.weight': psf}
        model.load_state_dict(state_dict)
        for param in model.parameters():
            param.requires_grad = False
        self.PSF = model

    def forward(self, MS1):
        m1 = self.SSR(MS1)
        m1_deg_SRF = self.srf(m1)
        m1_deg_psf = self.PSF(m1)
        m1_deg_psf = F.interpolate(m1_deg_psf, size=(128, 128))
        return m1, m1_deg_SRF, m1_deg_psf


class DualBranchPSF(nn.Module):
    def __init__(self, scale):
        super(DualBranchPSF, self).__init__()

        self.b1 = nn.Conv2d(1, 1, scale, scale, 0, bias=False)
        self.b2 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1, groups=1, bias=False),
            nn.PReLU()
        )
    def zero_mean_normalize(self, x, eps=1e-8):
        mean = x.mean(dim=[2, 3], keepdim=True)
        std = x.std(dim=[2, 3], keepdim=True) + eps
        normalized = (x - mean) / std
        return normalized

    def forward(self, x):
        batch, channels, height, width = x.size()

        branch2_outputs = []
        for i in range(channels):
            single_channel = x[:, i:i + 1, :, :]
            normalized_channel = self.zero_mean_normalize(single_channel)
            branch2 = self.b2(single_channel)
            branch2_outputs.append(branch2)

        branch1 = F.interpolate(torch.cat([self.b1(x[:, i, :, :].view(batch, 1, height, width)) for i in range(channels)], 1), size=(128, 128))  # [batch, channels, height, width]
        branch2 = torch.cat(branch2_outputs, dim=1)  # [batch, channels, height, width]
        output = branch1+branch2

        return output


class PSF(nn.Module):
    def __init__(self, scale):
        super(PSF, self).__init__()
        self.psf = nn.Conv2d(1, 1, scale, scale, 0, bias=False)  # in_channels, out_channels, kernel_size, stride, padding

    def forward(self, x):
        batch, channel, height, weight = list(x.size())
        return torch.cat([self.psf(x[:, i, :, :].view(batch, 1, height, weight)) for i in range(channel)], 1)


class SRF(nn.Module):
    def __init__(self):
        super(SRF, self).__init__()

        self.srf=torch.tensor([[    1,    	0,	   0,	  0,	 0,	 0,  0,	 0,           0],
                               [    0,0.474615225, 0,	  0,	 0,	 0,	 0,	 0,           0],
                               [    0,0.525384775, 0,	  0,	 0,	 0,	 0,	 0,           0],
                               [    0,      0,     1,     0,	 0,	 0,	 0,  0,           0],
                               [    0,      0,     0,0.000707491,0,	 0,	 0,	 0,           0],
                               [    0,	    0,     0,0.405019249,0,	 0,	 0,	 0,           0],
                               [    0,      0,     0,0.479489752,0,  0,	 0,	 0,           0],
                               [    0,	    0,	   0,0.114783508,0,	 0,	 0,	 0,           0],
                               [    0,	    0,	   0,     0,     1,	 0,	 0,	 0,           0],
                               [    0,	    0,	   0,	  0,	 0,	 1,	 0,	 0,           0],
                               [    0,	    0,	   0,	  0,	 0,	 0,	 1,	 0,           0],
                               [    0,	    0,	   0,	  0,	 0,	 0,	 0,	 0.071162967, 0],
                               [    0,	    0,	   0,	  0,	 0,	 0,	 0,	 0.458794981, 0.998910234],
                               [    0,	    0,	   0,	  0,	 0,	 0,	 0,	 0.470042052, 0.001089766]]).transpose(1,0).float()

    def forward(self,x):
        batch, channel_hsi, height, width = x.size()
        channel_msi_sp, channel_hsi_sp = self.srf.size()
        x_reshaped = torch.reshape(x, (batch, channel_hsi, height * width))
        hmsi = torch.bmm(self.srf.expand(batch,-1,-1).to('cuda:0'), x_reshaped).view(batch, channel_msi_sp, height, width)
        return hmsi



