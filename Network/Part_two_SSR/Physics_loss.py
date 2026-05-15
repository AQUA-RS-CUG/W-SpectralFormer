import torch


class GetAA(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        Rrs = x / 3.14159256
        rrs = Rrs / (0.52 + 1.7 * Rrs)
        c1, c3, c6 = rrs[:, 0:1], rrs[:, 2:3], rrs[:, 5:6]
        u6 = (-0.084 + (0.007056 + 0.68 * c6) ** 0.5) / 0.34
        bbp_0 = (u6 * 2.7680) / (1 - u6) - 0.0005
        e = torch.exp(-0.9 * c1 / c3)
        g = 3.99 - 3.59 * e
        bbp_values = [bbp_0 * (740 / wavelength) ** g + offset
            for wavelength, offset in zip([443, 493, 560, 665, 704, 740, 783, 833, 865],
                                          [0.0049, 0.0031, 0.0018, 0.0009, 0.0007, 0.0005, 0.0004, 0.0003, 0.0003])]
        return torch.cat(bbp_values, dim=1)


class LossQAA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = GetAA()

    def forward(self, x, y):
        neg_mask = torch.where(x <= 0)
        x[neg_mask] = 0.005
        y[neg_mask] = 0.005
        b1, b2 = self.model(x), self.model(y)
        b1[neg_mask] = float('nan')
        b2[neg_mask] = float('nan')
        loss = torch.nanmean(torch.abs(b2 - b1)) if not torch.all(torch.isnan(b2 - b1)) else 0.0
        return loss


class Loss_SAM(torch.nn.Module):
    def __init__(self):
        super(Loss_SAM, self).__init__()
        self.eps = 1e-10

    def forward(self, im_fake, im_true):
        sum1 = torch.sum(im_true * im_fake, 1)
        sum2 = torch.sum(im_true * im_true, 1)
        sum3 = torch.sum(im_fake * im_fake, 1)
        t = (sum2 * sum3) ** 0.5
        numlocal = torch.gt(t, 0)
        num = torch.sum(numlocal)
        t = sum1 / (t + self.eps)
        angle = torch.acos(t.clip(-1, 1))
        sumangle = torch.where(torch.isnan(angle), torch.full_like(angle, 0), angle).sum()
        if num == 0:
            averangle = sumangle
        else:
            averangle = sumangle / num
        SAM = averangle * 180 / 3.14159256
        return SAM
