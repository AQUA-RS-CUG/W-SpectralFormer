import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
from torch.nn import init as init


class PreNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


# --------fllow:Spatial--------
class ChannelAttention(nn.Module):
    def __init__(self, dim):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(dim, dim, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(dim, dim, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class DConv_FEM (nn.Module):
    def __init__(self, dim):
        super(DConv_FEM, self).__init__()
        self.branch1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, dilation=1)
        self.branch2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=3, dilation=3)
        self.branch3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=5, dilation=5)
        self.merge_conv = nn.Conv2d(3 * dim, dim, kernel_size=1)
        self.CAM = ChannelAttention(dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_concat = torch.cat([x1, x2, x3], dim=1)
        x_merged = self.merge_conv(x_concat)
        x_cam = self.CAM(x_merged)
        x_dw2 = self.sigmoid(x_cam)
        x_out = x_dw2 * x_merged
        x_out = x_out + x
        return x_out
# --------below:Spatial--------


# --------Spectral 1D FFT--------
class SpectralFrequencySelection(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.GELU, bias=False):
        super(SpectralFrequencySelection, self).__init__()
        self.act_fft = act_method()
        hid_dim = dim * dw

        self.complex_weight1_real = nn.Parameter(torch.Tensor(dim, hid_dim))
        self.complex_weight1_imag = nn.Parameter(torch.Tensor(dim, hid_dim))
        self.complex_weight2_real = nn.Parameter(torch.Tensor(hid_dim, dim))
        self.complex_weight2_imag = nn.Parameter(torch.Tensor(hid_dim, dim))

        init.kaiming_uniform_(self.complex_weight1_real, a=math.sqrt(5))
        init.kaiming_uniform_(self.complex_weight1_imag, a=math.sqrt(5))
        init.kaiming_uniform_(self.complex_weight2_real, a=math.sqrt(5))
        init.kaiming_uniform_(self.complex_weight2_imag, a=math.sqrt(5))

        if bias:
            self.b1_real = nn.Parameter(torch.zeros((1, 1, hid_dim)), requires_grad=True)
            self.b1_imag = nn.Parameter(torch.zeros((1, 1, hid_dim)), requires_grad=True)
            self.b2_real = nn.Parameter(torch.zeros((1, 1, dim)), requires_grad=True)
            self.b2_imag = nn.Parameter(torch.zeros((1, 1, dim)), requires_grad=True)

        self.bias = bias
        self.norm = norm

    def forward(self, x):
        """
        Input: [B, C, H, W]
        C：1D FFT
        Output: [B, C, H, W]
        """
        B, C, H, W = x.shape

        y = torch.fft.rfft2(x, norm=self.norm)
        weight1 = torch.complex(self.complex_weight1_real, self.complex_weight1_imag)  # [C, hid_dim]
        weight2 = torch.complex(self.complex_weight2_real, self.complex_weight2_imag)  # [hid_dim, C]
        y = rearrange(y, 'b c h w -> b h w c')
        y = y @ weight1

        if self.bias:
            b1 = torch.complex(self.b1_real, self.b1_imag)
            y = y + b1.unsqueeze(0)
        y_cat = torch.cat([y.real, y.imag], dim=-1)
        y_cat = self.act_fft(y_cat)
        y_real, y_imag = torch.chunk(y_cat, 2, dim=-1)
        y = torch.complex(y_real, y_imag)
        y = y @ weight2
        if self.bias:
            b2 = torch.complex(self.b2_real, self.b2_imag)
            y = y + b2.unsqueeze(0)
        y = rearrange(y, 'b h w c -> b c h w')
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y


class FFT_SAM(nn.Module):
    def __init__(self, dim, dim_head, heads):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.conv0 = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=False)
        self.fft_branch0 = SpectralFrequencySelection(dim=dim)

    def forward(self, x1):
        b, c, h, w = x1.shape
        q = self.conv0(x1)
        k = self.conv0(x1)
        v = self.conv1(x1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        Ffft = self.fft_branch0(x1)
        Ffft = rearrange(Ffft, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        attn_head = torch.matmul(Ffft, Ffft.transpose(-2, -1))
        q = torch.matmul(attn_head, q)
        k = torch.matmul(attn_head, k)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out
# --------below:Spectarl--------


# ---------fllow:fusion---------
class Pixel_token(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score_nets = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),  # 深度卷积
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        Pixel_att = self.score_nets(x)
        return Pixel_att


class Channel_token(nn.Module):
    def __init__(self, dim):
        super(Channel_token, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        Channel_att = self.attention(x)
        return Channel_att


class LG_FAM (nn.Module):
    def __init__(self, dim):
        super(LG_FAM, self).__init__()
        self.dim = dim
        self.gamma = nn.Parameter(torch.randn(1))
        self.Channel_token = Channel_token(dim)
        self.Pixel_token = Pixel_token(dim=dim)

    def forward(self, x, y):
        iden_x = x
        iden_y = y
        x_attention = self.Pixel_token(x)
        x_attention = torch.where(x_attention < 1e-5, torch.zeros_like(x_attention), x_attention)
        out_a = torch.mul(x, x_attention)
        y_attention = self.Channel_token(y)
        y_attention = torch.where(y_attention < 1e-5, torch.zeros_like(y_attention), y_attention)
        out_b = torch.mul(y, y_attention)
        out_a = torch.mul(out_a, y_attention)+iden_x
        out_b = torch.mul(out_b, x_attention)+iden_y
        gamma = torch.sigmoid(self.gamma)
        out = gamma * out_a + (1 - gamma) * out_b
        return out
# ---------below:fusion---------


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        out = self.net(x)
        return out


class FATM_net(nn.Module):
    def __init__(self, dim, dim_head, heads):
        super().__init__()
        self.norm = PreNorm(dim)
        self.conv1a = nn.Conv2d(dim, dim, 1, bias=False)
        self.DConv_FEM = DConv_FEM(dim)
        self.FFT_SAM = FFT_SAM(dim, dim_head, heads=heads)
        self.LG_FAM = LG_FAM(dim)
        self.norm2 = PreNorm(dim)
        self.e_att = FeedForward(dim)

    def forward(self, x):
        identity = x
        x = self.norm(x)
        out_a = self.DConv_FEM(self.conv1a(x))
        out_b = self.FFT_SAM(x)
        out = self.LG_FAM(out_a, out_b) + identity
        identity = out
        out = self.norm2(out)
        out = self.e_att(out) + identity
        return out

