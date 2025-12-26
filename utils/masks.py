from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch


def load_yolo_label(label_path: Path, img_width: int, img_height: int) -> Optional[np.ndarray]:
    if not label_path.exists():
        return None
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    with open(label_path, 'r') as f:
        lines = f.readlines()
    if not lines:
        return None
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        x_c = float(parts[1])
        y_c = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])
        x_center_px = int(x_c * img_width)
        y_center_px = int(y_c * img_height)
        box_w_px = int(w * img_width)
        box_h_px = int(h * img_height)
        x1 = max(0, x_center_px - box_w_px // 2)
        y1 = max(0, y_center_px - box_h_px // 2)
        x2 = min(img_width, x_center_px + box_w_px // 2)
        y2 = min(img_height, y_center_px + box_h_px // 2)
        mask[y1:y2, x1:x2] = 1
    return mask


def create_watermark_mask(
    image_path: Path,
    dataset_root: Path,
    label_subdir: str = "labels",
    split: str = "val"
) -> Optional[torch.Tensor]:
    image_stem = image_path.stem
    label_path = dataset_root / label_subdir / split / f"{image_stem}.txt"
    from PIL import Image
    img_pil = Image.open(image_path).convert("RGB")
    img_width, img_height = img_pil.size
    mask_np = load_yolo_label(label_path, img_width, img_height)
    if mask_np is None:
        return None
    mask_tensor = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0)
    return mask_tensor


def dilate_mask(mask: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    import torch.nn.functional as F
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device)
    padding = kernel_size // 2
    dilated = F.conv2d(mask, kernel, padding=padding)
    dilated = (dilated > 0).float()
    return dilated
