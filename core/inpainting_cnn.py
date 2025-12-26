import torch
import torch.nn as nn
import torch.nn.functional as F


class PartialConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.mask_conv = nn.Conv2d(1, 1, kernel_size, stride, padding, bias=False)
        self.mask_conv.weight.data.fill_(1.0)
        self.mask_conv.weight.requires_grad = False
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
    def forward(self, x, mask):
        with torch.no_grad():
            mask_updated = self.mask_conv(mask)
            kernel_size = self.conv.kernel_size[0] * self.conv.kernel_size[1]
            mask_ratio = mask_updated / kernel_size
            mask_updated = (mask_ratio > 0).float()
        x_masked = x * mask
        output = self.conv(x_masked)
        output = torch.where(
            mask_ratio > 0,
            output / (mask_ratio + 1e-8),
            torch.zeros_like(output)
        )
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)
        return output, mask_updated


class PartialConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.pconv1 = PartialConv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.pconv2 = PartialConv2d(out_channels, out_channels, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x, mask):
        out, mask_out = self.pconv1(x, mask)
        out = self.bn1(out)
        out = self.relu(out)
        out, mask_out = self.pconv2(out, mask_out)
        out = self.bn2(out)
        out = self.relu(out)
        return out, mask_out


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.upsample(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out


class PartialConvUNet(nn.Module):
    def __init__(self, input_channels=4, output_channels=3, base_channels=64, resolution=256):
        super().__init__()
        self.resolution = resolution
        self.enc1 = PartialConvBlock(input_channels, base_channels, stride=2)      # → H/2
        self.enc2 = PartialConvBlock(base_channels, base_channels * 2, stride=2)   # → H/4
        self.enc3 = PartialConvBlock(base_channels * 2, base_channels * 4, stride=2) # → H/8
        self.enc4 = PartialConvBlock(base_channels * 4, base_channels * 8, stride=2) # → H/16
        self.enc5 = PartialConvBlock(base_channels * 8, base_channels * 8, stride=2) # → H/32
        self.dec5 = UpsampleBlock(base_channels * 8, base_channels * 8)
        self.dec4 = UpsampleBlock(base_channels * 16, base_channels * 4)  # 512+512 from skip
        self.dec3 = UpsampleBlock(base_channels * 8, base_channels * 2)   # 256+256 from skip
        self.dec2 = UpsampleBlock(base_channels * 4, base_channels)        # 128+128 from skip
        self.dec1 = UpsampleBlock(base_channels * 2, base_channels)        # 64+64 from skip
        self.final_conv = nn.Conv2d(base_channels, output_channels, kernel_size=3, padding=1)
        self.final_act = nn.Tanh()
    def forward(self, x, mask):
        x_in = torch.cat([x, mask], dim=1)  # (B, 4, H, W)
        e1, m1 = self.enc1(x_in, mask)      # (B, 64, H/2, W/2)
        e2, m2 = self.enc2(e1, m1)          # (B, 128, H/4, W/4)
        e3, m3 = self.enc3(e2, m2)          # (B, 256, H/8, W/8)
        e4, m4 = self.enc4(e3, m3)          # (B, 512, H/16, W/16)
        e5, m5 = self.enc5(e4, m4)          # (B, 512, H/32, W/32)
        d5 = self.dec5(e5)                          # (B, 512, H/16, W/16)
        d4 = self.dec4(torch.cat([d5, e4], dim=1))  # (B, 256, H/8, W/8)
        d3 = self.dec3(torch.cat([d4, e3], dim=1))  # (B, 128, H/4, W/4)
        d2 = self.dec2(torch.cat([d3, e2], dim=1))  # (B, 64, H/2, W/2)
        d1 = self.dec1(torch.cat([d2, e1], dim=1))  # (B, 64, H, W)
        out = self.final_conv(d1)           # (B, 3, H, W)
        out = self.final_act(out)           # Tanh: [-1, 1]
        out = (out + 1) / 2                 # Scale to [0, 1]
        return out
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_inpainting_model(resolution=256, device='cuda'):
    model = PartialConvUNet(
        input_channels=4,
        output_channels=3,
        base_channels=64,
        resolution=resolution
    )
    model = model.to(device)
    param_count = model.count_parameters()
    print(f"Created PartialConvUNet ({resolution}x{resolution})")
    print(f"Total parameters: {param_count:,} ({param_count / 1e6:.2f}M)")
    return model


if __name__ == '__main__':
    model_256 = create_inpainting_model(resolution=256, device='cpu')
    x = torch.randn(1, 3, 256, 256)
    mask = torch.randint(0, 2, (1, 1, 256, 256)).float()
    with torch.no_grad():
        output = model_256(x, mask)
    print(f"\nInput shape: {x.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
