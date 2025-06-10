import torch
import torch.nn.functional as F
import torch.nn as nn
import math


def create_window(window_size, channel):
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def calculate_ssim(img1, img2, window, window_size, channel, mask=None):
    if mask is not None:
        img1 = img1 * mask
        img2 = img2 * mask

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2
    
    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if mask is not None:
        mask = mask.unsqueeze(0).unsqueeze(0)
        mask = mask.repeat(img1.size(0), img1.size(1), 1, 1)
        mask_window = F.conv2d(mask, window, padding=window_size//2, groups=channel)
        ssim_map = ssim_map * (mask_window > 0).float()
        return ssim_map.sum() / (mask_window > 0).float().sum()
    
    return ssim_map.mean()


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, mask=None, multi_channel=False):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.channel = 1
        self.window = create_window(window_size, 1)
        self.mask = mask
        self.multi_channel = multi_channel

    def forward(self, img1, img2):
        (_, c, _, _) = img1.size()

        if not self.multi_channel:
            if c != self.channel:
                self.window = create_window(self.window_size, c)
                self.channel = c
            if img1.is_cuda and not self.window.is_cuda:
                self.window = self.window.cuda(img1.get_device())

            return -calculate_ssim(img1, img2, self.window, self.window_size, c, self.mask)

        else:
            ssim_vals = []
            for ch in range(c):
                img1_ch = img1[:, ch:ch+1, :, :]
                img2_ch = img2[:, ch:ch+1, :, :]

                if 1 != self.channel:
                    self.window = create_window(self.window_size, 1)
                    self.channel = 1
                if img1.is_cuda and not self.window.is_cuda:
                    self.window = self.window.cuda(img1.get_device())

                ssim_c = calculate_ssim(img1_ch, img2_ch, self.window, self.window_size, 1, self.mask)
                ssim_vals.append(ssim_c)

            mean_ssim = torch.stack(ssim_vals).mean()
            return -mean_ssim
