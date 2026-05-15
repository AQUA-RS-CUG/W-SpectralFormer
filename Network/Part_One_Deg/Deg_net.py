import torch.nn as nn
import torch.nn.functional as F
import torch


class BlindNet(nn.Module):
    def __init__(self, ms_bands, ratio):
        super().__init__()
        self.ms_bands = ms_bands
        self.ratio = ratio
        self.psf = PSF(ratio)
        self.DualBranchPSF = DualBranchPSF(ratio)
        self.srf = SRF()

    def forward(self, Hr_HOLCI, Hr_LMS1):
        Lr_LOLCI = self.srf(Hr_HOLCI)
        Lr_LOLCI = torch.clamp(Lr_LOLCI, 0.0, 1.0)
        Lr_LMS1 = self.DualBranchPSF(Hr_LMS1)
        Lr_LMS1 = torch.clamp(Lr_LMS1, 0.0, 1.0)
        return Lr_LOLCI, Lr_LMS1


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
        self.psf = nn.Conv2d(1, 1, scale, scale, 0, bias=False)

    def forward(self, x):
        batch, channel, height, weight = list(x.size())
        return torch.cat([self.psf(x[:, i, :, :].view(batch, 1, height, weight)) for i in range(channel)], 1)
