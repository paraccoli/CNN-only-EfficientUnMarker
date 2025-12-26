import torch
import torch.nn.functional as F
import numpy as np


def compute_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)
    mse = F.mse_loss(img1, img2, reduction='mean')
    if mse < 1e-10:
        return 100.0  # Perfect match
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()


def compute_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> float:
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    if img1.size(1) == 3:
        weights = torch.tensor([0.299, 0.587, 0.114], device=img1.device).view(1, 3, 1, 1)
        img1 = (img1 * weights).sum(dim=1, keepdim=True)
        img2 = (img2 * weights).sum(dim=1, keepdim=True)
    sigma = 1.5
    gauss = torch.Tensor([
        np.exp(-(x - window_size//2)**2 / (2 * sigma**2))
        for x in range(window_size)
    ])
    gauss = gauss / gauss.sum()
    _1D_window = gauss.unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(img1.size(1), 1, window_size, window_size).contiguous()
    window = window.to(img1.device)
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=img1.size(1))
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=img2.size(1))
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=img1.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=img2.size(1)) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=img1.size(1)) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()


def compute_psnr_masked(img1: torch.Tensor, img2: torch.Tensor, mask: torch.Tensor) -> float:
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)
    if mask.dim() == 3:
        mask = mask.unsqueeze(0)
    mask = (mask > 0.5).float()
    mask_expanded = mask.expand_as(img1)
    diff = (img1 - img2) ** 2
    masked_diff = diff * mask_expanded
    num_pixels = mask_expanded.sum()
    if num_pixels < 1:
        return 100.0
    mse = masked_diff.sum() / num_pixels
    if mse < 1e-10:
        return 100.0
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()


def compute_ssim_masked(img1: torch.Tensor, img2: torch.Tensor, mask: torch.Tensor) -> float:
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)
    if mask.dim() == 3:
        mask = mask.unsqueeze(0)
    mask = (mask > 0.5).float()
    full_ssim_map = _ssim_map(img1, img2)
    mask_2d = mask.squeeze(1)  # (B, H, W)
    if mask_2d.sum() < 1:
        return 1.0
    masked_ssim = (full_ssim_map * mask_2d).sum() / mask_2d.sum()
    return masked_ssim.item()


def _ssim_map(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    if img1.size(1) == 3:
        weights = torch.tensor([0.299, 0.587, 0.114], device=img1.device).view(1, 3, 1, 1)
        img1 = (img1 * weights).sum(dim=1, keepdim=True)
        img2 = (img2 * weights).sum(dim=1, keepdim=True)
    sigma = 1.5
    gauss = torch.Tensor([
        np.exp(-(x - window_size//2)**2 / (2 * sigma**2))
        for x in range(window_size)
    ])
    gauss = gauss / gauss.sum()
    _1D_window = gauss.unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.to(img1.device)
    mu1 = F.conv2d(img1, window, padding=window_size//2)
    mu2 = F.conv2d(img2, window, padding=window_size//2)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.squeeze(1)  # (B, H, W)
