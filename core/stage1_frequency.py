import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

try:
    from timm import create_model
except ImportError:
    raise ImportError("Please install timm: pip install timm>=0.6.12")
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from utils.image_processing import frequency_statistics


class FrequencyFeatureExtractor(nn.Module):
    def __init__(self, num_bands: int = 8):
        super().__init__()
        self.num_bands = num_bands
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
    def forward(self, img: torch.Tensor) -> Dict[str, torch.Tensor]:
        freq_stats = frequency_statistics(img, num_bands=self.num_bands)
        magnitude = freq_stats['magnitude_spectrum']  # (B, C, H, W//2+1)
        mag_avg = magnitude.mean(dim=1, keepdim=True)  # (B, 1, H, W//2+1)
        H = mag_avg.shape[2]
        mag_square = F.interpolate(
            mag_avg,
            size=(H, H),
            mode='bilinear',
            align_corners=False
        )
        spatial_features = self.spatial_encoder(mag_square)  # (B, 64, 1, 1)
        spatial_features = spatial_features.flatten(1)  # (B, 64)
        return {
            'band_energies': freq_stats['band_energies'],
            'band_variances': freq_stats['band_variances'],
            'total_energy': freq_stats['total_energy'],
            'spatial_features': spatial_features
        }


class FrequencyBandPredictor(nn.Module):
    def __init__(
        self,
        num_bands: int = 8,
        pretrained: bool = True
    ):
        super().__init__()
        self.num_bands = num_bands
        self.freq_extractor = FrequencyFeatureExtractor(num_bands=num_bands)
        self.image_encoder = create_model(
            'mobilenetv3_small_100',
            pretrained=pretrained,
            num_classes=0,
            global_pool=''
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            img_features = self.image_encoder(dummy_input)
            img_features = F.adaptive_avg_pool2d(img_features, 1)
            img_feature_dim = img_features.shape[1]
        freq_feature_dim = num_bands * 2 + 1 + 64
        self.fusion = nn.Sequential(
            nn.Linear(img_feature_dim + freq_feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_bands)
        )
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        B = img.shape[0]
        freq_features = self.freq_extractor(img)
        freq_vec = torch.cat([
            freq_features['band_energies'],
            freq_features['band_variances'],
            freq_features['total_energy'].unsqueeze(1),
            freq_features['spatial_features']
        ], dim=1)  # (B, freq_feature_dim)
        img_features = self.image_encoder(img)  # (B, C, H', W')
        img_features = F.adaptive_avg_pool2d(img_features, 1)  # (B, C, 1, 1)
        img_features = img_features.flatten(1)  # (B, C)
        combined = torch.cat([img_features, freq_vec], dim=1)  # (B, total_dim)
        band_logits = self.fusion(combined)  # (B, num_bands)
        band_mask = torch.sigmoid(band_logits)  # (B, num_bands)
        return band_mask


class Stage1FrequencyAnalysis:
    def __init__(
        self,
        device: str = 'cuda',
        num_bands: int = 8,
        pretrained: bool = True
    ):
        self.device = torch.device(device)
        self.num_bands = num_bands
        self.predictor = FrequencyBandPredictor(
            num_bands=num_bands,
            pretrained=pretrained
        ).to(self.device).eval()
    def analyze(
        self,
        img: torch.Tensor,
        return_stats: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        img = img.to(self.device)
        with torch.no_grad():
            band_mask = self.predictor(img)
            if return_stats:
                freq_stats = frequency_statistics(img, num_bands=self.num_bands)
                stats = {
                    'band_mask': band_mask,
                    'band_energies': freq_stats['band_energies'],
                    'band_variances': freq_stats['band_variances'],
                    'total_energy': freq_stats['total_energy']
                }
                return band_mask, stats
            return band_mask, {}
    def get_attack_priority(self, band_mask: torch.Tensor) -> torch.Tensor:
        priority_indices = torch.argsort(band_mask, dim=1, descending=True)
        return priority_indices
    def apply_band_filter(
        self,
        img: torch.Tensor,
        band_mask: torch.Tensor,
        threshold: float = 0.5
    ) -> torch.Tensor:
        from utils.image_processing import dct_transform, inverse_dct_transform
        dct_coeffs = dct_transform(img)  # (B, 3, H, W//2+1)
        B, C, H, W = dct_coeffs.shape
        freq_y = torch.fft.fftfreq(H, device=img.device).reshape(-1, 1)
        freq_x = torch.fft.rfftfreq(H, device=img.device).reshape(1, -1)
        radial_freq = torch.sqrt(freq_y**2 + freq_x**2)
        radial_freq = radial_freq / radial_freq.max()
        band_edges = torch.linspace(0, 1, self.num_bands + 1, device=img.device)
        freq_band_mask = torch.zeros_like(radial_freq)
        for i in range(self.num_bands):
            keep_band = (band_mask[:, i] > threshold).float()  # (B,)
            spatial_mask = (
                (radial_freq >= band_edges[i]) & 
                (radial_freq < band_edges[i + 1])
            ).float()
            for b in range(B):
                freq_band_mask += spatial_mask * keep_band[b]
        freq_band_mask = freq_band_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        filtered_coeffs = dct_coeffs * freq_band_mask
        filtered_img = inverse_dct_transform(filtered_coeffs)
        return filtered_img


def create_frequency_analyzer(
    device: str = 'cuda',
    num_bands: int = 8,
    pretrained: bool = True
) -> Stage1FrequencyAnalysis:
    return Stage1FrequencyAnalysis(
        device=device,
        num_bands=num_bands,
        pretrained=pretrained
    )
