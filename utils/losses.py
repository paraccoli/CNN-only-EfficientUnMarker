import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class MaskedMSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    def forward(self, pred, target, mask):
        mask = mask.expand_as(pred)
        squared_error = (pred - target) ** 2
        masked_error = squared_error * mask
        if self.reduction == 'mean':
            valid_pixels = mask.sum() + 1e-8
            loss = masked_error.sum() / valid_pixels
        elif self.reduction == 'sum':
            loss = masked_error.sum()
        else:
            loss = masked_error
        
        return loss


class VGGPerceptualLoss(nn.Module):
    def __init__(self, layer_weights=None, device='cuda'):
        super().__init__()
        if layer_weights is None:
            self.layer_weights = {
                'relu1_2': 0.125,
                'relu2_2': 0.25,
                'relu3_3': 0.375,
                'relu4_3': 0.25
            }
        else:
            self.layer_weights = layer_weights
        vgg16 = models.vgg16(pretrained=True).features
        vgg16.eval()
        for param in vgg16.parameters():
            param.requires_grad = False
        self.layer_name_mapping = {
            '3': 'relu1_2',   # After conv1_2
            '8': 'relu2_2',   # After conv2_2
            '15': 'relu3_3',  # After conv3_3
            '22': 'relu4_3'   # After conv4_3
        }
        self.feature_extractor = vgg16
        self.feature_extractor = self.feature_extractor.to(device)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.device = device
    def normalize(self, x):
        """Normalize input to VGG range."""
        return (x - self.mean.to(x.device)) / self.std.to(x.device)
    def extract_features(self, x):
        x = self.normalize(x)
        features = {}
        for name, module in self.feature_extractor._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                layer_name = self.layer_name_mapping[name]
                features[layer_name] = x
        return features
    def forward(self, pred, target):
        pred_features = self.extract_features(pred)
        with torch.no_grad():
            target_features = self.extract_features(target)
        loss = 0.0
        for layer_name, weight in self.layer_weights.items():
            pred_feat = pred_features[layer_name]
            target_feat = target_features[layer_name]
            layer_loss = F.mse_loss(pred_feat, target_feat)
            loss += weight * layer_loss
        return loss


class InpaintingLoss(nn.Module):
    def __init__(self, lambda_reg=1.0, lambda_bg=0.1, lambda_perc=0.1, 
                 lambda_detector=0.0, detector=None, device='cuda'):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.lambda_bg = lambda_bg
        self.lambda_perc = lambda_perc
        self.lambda_detector = lambda_detector
        self.detector = detector
        self.mse_loss = MaskedMSELoss()
        self.perceptual_loss = VGGPerceptualLoss(device=device)
    def forward(self, pred, target, mask):
        loss_pixel_reg = self.mse_loss(pred, target, mask)
        loss_pixel_bg = self.mse_loss(pred, target, 1 - mask)
        loss_perc = self.perceptual_loss(pred, target)
        loss_det = torch.tensor(0.0, device=pred.device)
        if self.lambda_detector > 0 and self.detector is not None:
            det_score = self.detector.score_differentiable(pred, branch='quick')  # (B,)
            loss_det = det_score.mean()
        total_loss = (
            self.lambda_reg * loss_pixel_reg +
            self.lambda_bg * loss_pixel_bg +
            self.lambda_perc * loss_perc +
            self.lambda_detector * loss_det
        )
        loss_dict = {
            'total': total_loss.item(),
            'pixel_reg': loss_pixel_reg.item(),
            'pixel_bg': loss_pixel_bg.item(),
            'perceptual': loss_perc.item(),
            'detector': loss_det.item() if isinstance(loss_det, torch.Tensor) else 0.0
        }
        return total_loss, loss_dict


if __name__ == '__main__':
    B, C, H, W = 2, 3, 256, 256
    pred = torch.randn(B, C, H, W).abs()  # [0, ~1]
    target = torch.randn(B, C, H, W).abs()
    mask = torch.randint(0, 2, (B, 1, H, W)).float()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pred = pred.to(device)
    target = target.to(device)
    mask = mask.to(device)
    print("1. MaskedMSELoss")
    mse_loss = MaskedMSELoss()
    loss_region = mse_loss(pred, target, mask)
    loss_bg = mse_loss(pred, target, 1 - mask)
    print(f"   Region loss: {loss_region.item():.4f}")
    print(f"   Background loss: {loss_bg.item():.4f}")
    print("\n2. VGGPerceptualLoss")
    vgg_loss = VGGPerceptualLoss(device=device)
    loss_perc = vgg_loss(pred, target)
    print(f"   Perceptual loss: {loss_perc.item():.4f}")
    print("\n3. InpaintingLoss (combined)")
    combined_loss = InpaintingLoss(
        lambda_reg=1.0,
        lambda_bg=0.1,
        lambda_perc=0.1,
        device=device
    )
    total_loss, loss_dict = combined_loss(pred, target, mask)
    print(f"   Total loss: {loss_dict['total']:.4f}")
    print(f"   - Pixel (region): {loss_dict['pixel_reg']:.4f}")
    print(f"   - Pixel (bg): {loss_dict['pixel_bg']:.4f}")
    print(f"   - Perceptual: {loss_dict['perceptual']:.4f}")
