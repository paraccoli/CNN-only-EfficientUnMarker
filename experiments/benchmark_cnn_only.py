import argparse
import csv
import os
import time
from pathlib import Path
from typing import Optional, List, Tuple

import sys
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
from PIL import Image
from core.efficient_unmarker import EfficientUnMarker
try:
    from utils.metrics import compute_psnr as project_compute_psnr
except Exception:
    project_compute_psnr = None

def load_image(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    # HWC -> CHW
    chw = np.transpose(arr, (0, 1, 2)) if arr.ndim == 3 else arr
    chw = np.transpose(chw, (2, 0, 1))
    tensor = torch.from_numpy(chw).unsqueeze(0)  # (1,3,H,W)
    return tensor

def compute_psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    if project_compute_psnr is not None:
        try:
            return float(project_compute_psnr(a.cpu(), b.cpu()))
        except Exception:
            pass
    a = a.clamp(0, 1)
    b = b.clamp(0, 1)
    mse = torch.mean((a - b) ** 2).item()
    if mse <= 1e-12:
        return 99.0
    return 10.0 * np.log10(1.0 / mse)

def match_clean(path: Path, clean_root: Optional[Path], wm_suffix: Optional[str]) -> Optional[Path]:
    if clean_root is None:
        return None
    candidate = clean_root / path.name
    if candidate.exists():
        return candidate
    if wm_suffix:
        stem = path.stem
        if stem.endswith(wm_suffix):
            clean_stem = stem[: -len(wm_suffix)]
            candidate2 = clean_root / f"{clean_stem}{path.suffix}"
            if candidate2.exists():
                return candidate2
    return None

def list_images(root: Path, exts: Tuple[str, ...], wm_suffix: Optional[str]) -> List[Path]:
    files = []
    for ext in exts:
        files.extend(list(root.rglob(f"*{ext}")))
    if wm_suffix:
        files = [p for p in files if p.stem.endswith(wm_suffix)]
    return sorted(files)

def reset_cuda_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def get_cuda_peak() -> Optional[int]:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated()
    return None

def main():
    parser = argparse.ArgumentParser(description="Phase A: CNN-only benchmark (quick detector)")
    parser.add_argument("--config", type=str, default="configs/fast_cnn_only.yaml", help="Path to YAML config")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory of watermarked images")
    parser.add_argument("--clean_dir", type=str, default=None, help="Directory of clean reference images (optional)")
    parser.add_argument("--output", type=str, default="results/benchmarks/phaseA_cnn_only.csv", help="Output CSV path")
    parser.add_argument("--exts", type=str, default=".png,.jpg,.jpeg", help="Comma-separated extensions")
    parser.add_argument("--wm_suffix", type=str, default=None, help="Suffix indicating watermarked files (e.g., 'v2')")
    parser.add_argument("--det_threshold", type=float, default=None, help="Override detection success threshold")
    args = parser.parse_args()
    input_root = Path(args.input_dir)
    clean_root = Path(args.clean_dir) if args.clean_dir else None
    out_csv = Path(args.output)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    exts = tuple([e.strip() for e in args.exts.split(",") if e.strip()])
    imgs = list_images(input_root, exts, wm_suffix=args.wm_suffix)
    if not imgs:
        print(f"No images found under {input_root} with exts {exts}")
        return
    unmarker = EfficientUnMarker(config_path=args.config)
    if args.det_threshold is not None:
        unmarker.config["target_detection"] = float(args.det_threshold)
        print(f"Overriding target_detection to {unmarker.config['target_detection']}")
    total = len(imgs)
    successes = 0
    psnr_list: List[float] = []
    times: List[float] = []
    vram_peaks: List[Optional[int]] = []
    rows = []
    print(f"Running CNN-only benchmark on {total} images...")
    for idx, img_path in enumerate(imgs, 1):
        try:
            wm_tensor = load_image(img_path)
            clean_path = match_clean(img_path, clean_root, wm_suffix=args.wm_suffix)
            clean_tensor = load_image(clean_path) if clean_path else None
            reset_cuda_peak()
            start = time.time()
            attacked, metrics = unmarker.attack(wm_tensor, mask=None, clean_img=clean_tensor)
            elapsed = time.time() - start
            peak = get_cuda_peak()
            success = bool(metrics.get("success", False))
            detection_score = float(metrics.get("detection_score", np.nan))
            psnr = None
            if clean_tensor is not None:
                psnr = compute_psnr(attacked.cpu(), clean_tensor.cpu())
                psnr_list.append(psnr)
            times.append(elapsed)
            vram_peaks.append(peak)
            if success:
                successes += 1
            rows.append({
                "file": str(img_path.relative_to(input_root)),
                "success": success,
                "detection_score": detection_score,
                "time_sec": elapsed,
                "psnr": psnr if psnr is not None else "",
                "vram_peak_bytes": peak if peak is not None else ""
            })
            rate = successes / idx
            avg_time = sum(times) / len(times)
            print(f"[{idx}/{total}] success={success} score={detection_score:.4f} time={elapsed:.2f}s avg_time={avg_time:.2f}s rate={rate*100:.1f}%")
        except Exception as e:
            print(f"[{idx}/{total}] ERROR on {img_path}: {e}")
            rows.append({
                "file": str(img_path.relative_to(input_root)),
                "success": False,
                "detection_score": "",
                "time_sec": "",
                "psnr": "",
                "vram_peak_bytes": "",
                "error": str(e)
            })
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "file", "success", "detection_score", "time_sec", "psnr", "vram_peak_bytes", "error"
        ])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    avg_time = sum(times) / len(times) if times else float("nan")
    avg_psnr = sum(psnr_list) / len(psnr_list) if psnr_list else float("nan")
    success_rate = successes / total if total else 0.0
    print("\nBenchmark Summary (Phase A: CNN-only)")
    print(f"  Images: {total}")
    print(f"  Success rate: {success_rate*100:.1f}% (threshold={unmarker.config['target_detection']})")
    print(f"  Avg runtime: {avg_time:.2f}s/image (target < 10s)")
    if torch.cuda.is_available():
        peaks = [p for p in vram_peaks if p is not None]
        if peaks:
            print(f"  Max VRAM peak: {max(peaks)/ (1024**3):.2f} GB (target < 8GB)")
    if not np.isnan(avg_psnr):
        print(f"  Avg PSNR: {avg_psnr:.2f} dB")
    else:
        print("  PSNR: (clean references not provided)")
    print(f"\nCSV saved: {out_csv}")

if __name__ == "__main__":
    main()
