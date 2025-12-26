import torch
import torch.nn.functional as F
from typing import Tuple, Dict
import numpy as np


def dct_transform(img: torch.Tensor) -> torch.Tensor:
    fft_coeffs = torch.fft.rfft2(img, norm='ortho')
    return fft_coeffs


def dct_magnitude(img: torch.Tensor) -> torch.Tensor:
    fft_coeffs = dct_transform(img)
    magnitude = torch.abs(fft_coeffs)
    return magnitude


def frequency_statistics(img: torch.Tensor, num_bands: int = 8) -> Dict[str, torch.Tensor]:
    B, C, H_img, W_img = img.shape
    magnitude = dct_magnitude(img)  # (B, C, H, W//2+1)
    B, C, H, W_freq = magnitude.shape
    freq_y = torch.fft.fftfreq(H, device=img.device).reshape(-1, 1)
    freq_x = torch.fft.rfftfreq(W_img, device=img.device).reshape(1, -1)
    radial_freq = torch.sqrt(freq_y**2 + freq_x**2)
    radial_freq = radial_freq / radial_freq.max()
    band_edges = torch.linspace(0, 1, num_bands + 1, device=img.device)
    band_energies = torch.zeros(B, num_bands, device=img.device)
    band_variances = torch.zeros(B, num_bands, device=img.device)
    for i in range(num_bands):
        mask = (radial_freq >= band_edges[i]) & (radial_freq < band_edges[i + 1])
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        band_mag = magnitude * mask
        band_energies[:, i] = band_mag.sum(dim=(1, 2, 3))
        band_mean = band_mag.sum(dim=(2, 3), keepdim=True) / mask.sum()
        band_var = ((band_mag - band_mean) ** 2 * mask).sum(dim=(1, 2, 3))
        band_variances[:, i] = band_var / (mask.sum() + 1e-8)
    total_energy = magnitude.pow(2).sum(dim=(1, 2, 3))
    return {
        'band_energies': band_energies,
        'band_variances': band_variances,
        'total_energy': total_energy,
        'magnitude_spectrum': magnitude
    }


def visualize_frequency_spectrum(img: torch.Tensor) -> torch.Tensor:
    magnitude = dct_magnitude(img)  # (1, C, H, W//2+1)
    mag_avg = magnitude.mean(dim=1, keepdim=True)  # (1, 1, H, W//2+1)
    mag_log = torch.log1p(mag_avg)
    mag_norm = (mag_log - mag_log.min()) / (mag_log.max() - mag_log.min() + 1e-8)
    return mag_norm


def multi_scale_resize(img: torch.Tensor, sizes: list) -> list:
    resized = []
    for size in sizes:
        if isinstance(size, int):
            size = (size, size)
        resized_img = F.interpolate(
            img,
            size=size,
            mode='bicubic',
            align_corners=False
        )
        resized.append(resized_img)
    return resized


def gaussian_blur(img: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0) -> torch.Tensor:
    coords = torch.arange(kernel_size, dtype=torch.float32, device=img.device)
    coords -= kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = g.unsqueeze(0) * g.unsqueeze(1)
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1,1,K,K)
    C = img.shape[1]
    kernel = kernel.repeat(C, 1, 1, 1)
    padding = kernel_size // 2
    blurred = F.conv2d(img, kernel, padding=padding, groups=C)
    return blurred


def frequency_smoothness_loss(img: torch.Tensor, weight: float = 0.1) -> torch.Tensor:
    magnitude = dct_magnitude(img)
    H, W = magnitude.shape[-2:]
    freq_y = torch.arange(H, device=img.device).reshape(-1, 1).float()
    freq_x = torch.arange(W, device=img.device).reshape(1, -1).float()
    radial_dist = torch.sqrt(freq_y**2 + freq_x**2)
    radial_dist = radial_dist / radial_dist.max()
    freq_weight = radial_dist.pow(2)
    weighted_mag = magnitude * freq_weight.unsqueeze(0).unsqueeze(0)
    loss = weighted_mag.mean() * weight
    return loss


def inverse_dct_transform(dct_coeffs: torch.Tensor) -> torch.Tensor:
    img = torch.fft.irfft2(dct_coeffs, norm='ortho')
    return img


def adaptive_low_pass_filter(
    img: torch.Tensor, 
    cutoff_freq: float = 0.5, 
    order: int = 2
) -> torch.Tensor:
    dct_coeffs = dct_transform(img)
    B, C, H, W_orig = img.shape
    freq_y = torch.fft.fftfreq(H, device=img.device).reshape(-1, 1)
    freq_x = torch.fft.rfftfreq(W_orig, device=img.device).reshape(1, -1)
    radial_freq = torch.sqrt(freq_y**2 + freq_x**2)
    radial_freq = radial_freq / radial_freq.max()
    filter_mask = 1 / (1 + (radial_freq / cutoff_freq) ** (2 * order))
    filter_mask = filter_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W_orig//2+1)
    filtered_coeffs = dct_coeffs * filter_mask
    filtered_img = torch.fft.irfft2(filtered_coeffs, s=(H, W_orig), norm='ortho')
    return filtered_img
