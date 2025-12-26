import torch
import torch.nn as nn
from typing import Tuple, Optional

try:
    from timm import create_model
except ImportError:
    raise ImportError("Please install timm: pip install timm>=0.6.12")


class QuickDetector(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone = create_model(
            'mobilenetv3_small_100',
            pretrained=pretrained,
            num_classes=1
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.backbone(x)
        confidence = torch.sigmoid(logits).squeeze(1)
        return confidence


class DeepDetector(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone = create_model(
            'densenet121',
            pretrained=pretrained,
            num_classes=1
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.backbone(x)
        confidence = torch.sigmoid(logits).squeeze(1)
        return confidence


class DualBranchDetector(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        uncertainty_low: float = 0.3,
        uncertainty_high: float = 0.7,
        quick_weight: float = 0.6,
        deep_weight: float = 0.4
    ):
        super().__init__()
        self.quick_detector = QuickDetector(pretrained=pretrained)
        self.deep_detector = DeepDetector(pretrained=pretrained)
        self.uncertainty_low = uncertainty_low
        self.uncertainty_high = uncertainty_high
        self.quick_weight = quick_weight
        self.deep_weight = deep_weight
    def forward(
        self,
        x: torch.Tensor,
        force_deep: bool = False
    ) -> Tuple[torch.Tensor, dict]:
        quick_conf = self.quick_detector(x)
        uncertain_mask = (
            (quick_conf >= self.uncertainty_low) &
            (quick_conf <= self.uncertainty_high)
        ) | force_deep
        if uncertain_mask.any() or force_deep:
            deep_conf = self.deep_detector(x)
            final_conf = (
                self.quick_weight * quick_conf +
                self.deep_weight * deep_conf
            )
            info = {
                'used_deep': True,
                'quick_conf': quick_conf,
                'deep_conf': deep_conf,
                'uncertain_count': uncertain_mask.sum().item()
            }
        else:
            final_conf = quick_conf
            info = {
                'used_deep': False,
                'quick_conf': quick_conf,
                'deep_conf': None,
                'uncertain_count': 0
            }
        return final_conf, info
    def eval_quick_only(self, x: torch.Tensor) -> torch.Tensor:
        return self.quick_detector(x)
    def eval_deep_only(self, x: torch.Tensor) -> torch.Tensor:
        return self.deep_detector(x)


class WatermarkDetector:
    def __init__(
        self,
        device: str = 'cuda',
        mode: str = 'adaptive',
        pretrained: bool = True
    ):
        self.device = torch.device(device)
        self.mode = mode
        if mode == 'adaptive':
            self.model = DualBranchDetector(pretrained=pretrained)
        elif mode == 'quick':
            self.model = QuickDetector(pretrained=pretrained)
        elif mode == 'deep':
            self.model = DeepDetector(pretrained=pretrained)
        else:
            raise ValueError(f"Invalid mode: {mode}")
        self.model.to(self.device).eval()
    def detect(
        self,
        img: torch.Tensor,
        return_info: bool = False
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        img = img.to(self.device)
        with torch.no_grad():
            if self.mode == 'adaptive':
                confidence, info = self.model(img)
                if return_info:
                    return confidence, info
                return confidence, None
            else:
                confidence = self.model(img)
                return confidence, None
    def is_watermarked(
        self,
        img: torch.Tensor,
        threshold: float = 0.5
    ) -> torch.Tensor:
        confidence, _ = self.detect(img)
        return confidence > threshold
    def score_differentiable(
        self,
        img: torch.Tensor,
        branch: str = 'quick'
    ) -> torch.Tensor:
        img = img.to(self.device)
        if self.mode == 'adaptive':
            if branch == 'deep':
                return self.model.deep_detector(img)
            return self.model.quick_detector(img)
        else:
            return self.model(img)


def create_detector(
    mode: str = 'adaptive',
    device: str = 'cuda',
    pretrained: bool = True
) -> WatermarkDetector:
    return WatermarkDetector(
        device=device,
        mode=mode,
        pretrained=pretrained
    )
