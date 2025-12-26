import torch
import torch.nn.functional as F
from pathlib import Path
import yaml
import time
from typing import Dict, Tuple, Optional

import sys
sys.path.append(str(Path(__file__).parent.parent))

from core.detection import WatermarkDetector
from core.early_stopping import EarlyStopping
from core.stage1_frequency import Stage1FrequencyAnalysis
from utils.image_processing import frequency_smoothness_loss
from utils.metrics import compute_psnr


class EfficientUnMarker:
    def __init__(self, config_path: str = 'configs/default.yaml'):
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_file) as f:
            self.config = yaml.safe_load(f)
        if self.config['device'] == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available, using CPU")
            self.device = torch.device('cpu')
        elif self.config['device'] == 'cuda' and torch.cuda.is_available():
            try:
                cap = torch.cuda.get_device_capability()
                if cap[0] >= 12:  # sm_120+ (RTX 5070 Ti)
                    self.device = torch.device('cuda')
                else:
                    self.device = torch.device('cuda')
            except RuntimeError as e:
                if 'no kernel image' in str(e):
                    print(f"{torch.cuda.get_device_name(0)} (sm_{cap[0]}{cap[1]}) detected")
                    print("PyTorch build missing sm_120 kernels (requires CUDA 12.8+)")
                    print("Install with: pip3 install torch --index-url https://download.pytorch.org/whl/cu128")
                    self.device = torch.device('cpu')
                else:
                    raise
        else:
            self.device = torch.device(self.config['device'])
        print("Loading Phase 2 components...")
        if self.config.get('stage1', {}).get('enabled', False):
            print("  [Stage 1] Frequency band predictor (MobileNetV3)")
            self.freq_analyzer = Stage1FrequencyAnalysis(
                device=str(self.device),
                num_bands=8,
                pretrained=True
            )
        else:
            self.freq_analyzer = None
        detector_mode = self.config.get('detector', {}).get('mode', 'adaptive')
        print(f"  [Detector] Mode: {detector_mode} (Quick: MobileNetV3" +
              (")" if detector_mode == 'quick' else " + Deep: DenseNet)"))
        self.detector = WatermarkDetector(
            device=str(self.device),
            mode=detector_mode,
            pretrained=True
        )
        stopping_cfg = self.config.get('stopping', {})
        self.early_stopper = EarlyStopping(
            check_interval=5,
            detection_threshold=self.config['target_detection'],
            variance_threshold=1e-4,
            patience=5,
            min_iterations=10,
            psnr_min=stopping_cfg.get('psnr_min'),
            psnr_patience=stopping_cfg.get('psnr_patience', 2)
        )
        print(f"EfficientUnMarker (Phase 2) initialized on {self.device}")
        print(f"Max iterations: {self.config['max_iterations']}")
        print(f"Stage 1 enabled: {self.config.get('stage1', {}).get('enabled', False)}")
        print(f"Stage 3 enabled: {self.config.get('stage3', {}).get('enabled', False)}")
    def attack(
        self,
        img: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        clean_img: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        start_time = time.time()
        img = img.to(self.device)
        if clean_img is None:
            clean_img = img.clone().detach()
        else:
            clean_img = clean_img.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)
        h, w = img.shape[-2:]
        print(f"Attacking image {img.shape}...")
        metrics = {
            'stage1_time': 0,
            'stage2_time': 0,
            'stage3_time': 0,
            'total_iterations': 0,
            'early_stopped': False
        }
        band_mask = None
        if self.freq_analyzer is not None:
            stage1_start = time.time()
            print("[Stage 1] Analyzing frequency bands...")
            band_mask, freq_stats = self.freq_analyzer.analyze(img, return_stats=True)
            metrics['stage1_time'] = time.time() - stage1_start
            print(f"[Stage 1] Completed in {metrics['stage1_time']*1000:.1f}ms")
        stage2_start = time.time()
        attacked_img = self._multi_scale_attack(img, band_mask, mask=mask, clean_img=clean_img)
        metrics['stage2_time'] = time.time() - stage2_start
        metrics['total_iterations'] = self.early_stopper.total_checks * self.early_stopper.check_interval
        metrics['early_stopped'] = (
            self.early_stopper.stopped_by_detection or
            self.early_stopper.stopped_by_convergence
        )
        if self.config.get('stage3', {}).get('enabled', False):
            stage3_start = time.time()
            print("[Stage 3] Quality refinement...")
            attacked_img = self._quality_refinement(attacked_img)
            metrics['stage3_time'] = time.time() - stage3_start
            print(f"[Stage 3] Completed in {metrics['stage3_time']:.2f}s")
        elapsed_time = time.time() - start_time
        with torch.no_grad():
            detection_score, detector_info = self.detector.detect(attacked_img, return_info=True)
            detection_score = detection_score.item()
        success = detection_score < self.config['target_detection']
        print(f"Attack completed in {elapsed_time:.2f}s")
        print(f"Detection score: {detection_score:.4f} (threshold: {self.config['target_detection']})")
        print(f"Success: {success}")
        self.early_stopper.reset()
        metrics.update({
            'success': success,
            'detection_score': detection_score,
            'time': elapsed_time,
            'detector_info': detector_info
        })
        return attacked_img, metrics
    def _multi_scale_attack(
        self,
        img: torch.Tensor,
        band_mask: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        clean_img: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        h, w = img.shape[-2:]
        low_res_size = self.config['stage2']['low_res_size']
        low_iters = self.config['stage2']['low_iters']
        high_iters = self.config['stage2']['high_iters']
        print(f"[Stage 2] Low-res ({low_res_size}x{low_res_size}): up to {low_iters} iterations")
        low_res = F.interpolate(img, size=low_res_size, mode='bilinear', align_corners=False)
        low_mask = None
        low_clean = None
        if mask is not None:
            low_mask = F.interpolate(mask, size=low_res_size, mode='nearest')
        if clean_img is not None:
            low_clean = F.interpolate(clean_img, size=low_res_size, mode='bilinear', align_corners=False)
        low_opt = self._optimize_with_early_stopping(
            low_res, max_iters=low_iters, mask=low_mask, clean_img=low_clean
        )
        print(f"[Stage 2] Upscaling to {h}x{w}")
        upscaled = F.interpolate(low_opt, size=(h, w), mode='bicubic', align_corners=False)
        print(f"[Stage 2] High-res finetune: up to {high_iters} iterations")
        final = self._optimize_with_early_stopping(
            upscaled, max_iters=high_iters, mask=mask, clean_img=clean_img
        )
        return final
    def _optimize_with_early_stopping(
        self, 
        img: torch.Tensor, 
        max_iters: int,
        mask: Optional[torch.Tensor] = None,
        clean_img: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        img_opt = img.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([img_opt], lr=self.config.get('learning_rate', 0.01))
        loss_cfg = self.config.get('loss', {})
        w_det = float(loss_cfg.get('detector_weight', 1.0))
        w_freq = float(loss_cfg.get('freq_weight', 0.0))
        w_smooth = float(loss_cfg.get('smooth_weight', 0.0))
        w_rec = float(loss_cfg.get('rec_region_weight', 0.0))
        w_bg = float(loss_cfg.get('w_bg', 0.0))
        w_psnr_penalty = float(loss_cfg.get('psnr_penalty', 0.0))
        psnr_penalty_floor = loss_cfg.get('psnr_penalty_floor', None)
        if psnr_penalty_floor is None:
            psnr_penalty_floor = self.config.get('stopping', {}).get('psnr_min', 0.0)
        psnr_penalty_floor = float(psnr_penalty_floor or 0.0)
        psnr_guard = self.early_stopper.psnr_min is not None and clean_img is not None
        for i in range(max_iters):
            optimizer.zero_grad()
            psnr_est = None
            if (psnr_guard or w_psnr_penalty > 0.0) and clean_img is not None:
                mse = F.mse_loss(img_opt, clean_img)
                psnr_est = 10.0 * torch.log10(1.0 / (mse + 1e-8))
            det_conf_q = self.detector.score_differentiable(img_opt, branch='quick')
            det_conf_d = self.detector.score_differentiable(img_opt, branch='deep')
            det_conf = torch.maximum(det_conf_q, det_conf_d)
            det_loss = det_conf.mean()
            if w_freq > 0.0:
                fft = torch.fft.rfft2(img_opt)
                magnitude = torch.abs(fft)
                freq_loss = -magnitude.mean()
            else:
                freq_loss = det_loss.detach() * 0.0
            if w_smooth > 0.0:
                smooth_loss = frequency_smoothness_loss(
                    img_opt, weight=self.config.get('quality', {}).get('freq_smoothness_weight', 0.1)
                )
            else:
                smooth_loss = det_loss.detach() * 0.0
            if w_rec > 0.0 and clean_img is not None and mask is not None:
                rec_loss = ((img_opt - clean_img) ** 2 * mask.expand_as(img_opt)).mean()
            else:
                rec_loss = det_loss.detach() * 0.0
            if w_bg > 0.0 and clean_img is not None and mask is not None:
                bg_mask = 1.0 - mask.expand_as(img_opt)
                bg_loss = ((img_opt - clean_img) ** 2 * bg_mask).mean()
            else:
                bg_loss = det_loss.detach() * 0.0
            psnr_penalty_term = det_loss.detach() * 0.0
            if w_psnr_penalty > 0.0 and psnr_est is not None:
                deficit = torch.clamp(psnr_penalty_floor - psnr_est, min=0.0)
                psnr_penalty_term = w_psnr_penalty * deficit
            loss = (
                w_det * det_loss
                + w_freq * freq_loss
                + w_smooth * smooth_loss
                + w_rec * rec_loss
                + w_bg * bg_loss
                + psnr_penalty_term
            )
            loss.backward()
            if mask is not None and img_opt.grad is not None:
                grad_mask = mask.expand_as(img_opt)
                img_opt.grad.mul_(grad_mask)
            optimizer.step()
            with torch.no_grad():
                img_opt.clamp_(0, 1)
            psnr_current = None
            if psnr_guard and psnr_est is not None:
                psnr_current = float(psnr_est.detach().cpu().item())
            elif self.early_stopper.psnr_min is not None and clean_img is not None:
                psnr_current = compute_psnr(img_opt.detach().cpu(), clean_img.detach().cpu())
            if (i + 1) % self.early_stopper.check_interval == 0:
                with torch.no_grad():
                    detection_score, _ = self.detector.detect(img_opt.detach())
                    detection_score = detection_score.item()
                if self.early_stopper.should_stop(i, loss.item(), detection_score, psnr=psnr_current):
                    stop_reason = []
                    if self.early_stopper.stopped_by_detection:
                        stop_reason.append("detection")
                    if self.early_stopper.stopped_by_convergence:
                        stop_reason.append("convergence")
                    if self.early_stopper.stopped_by_quality:
                        stop_reason.append("quality")
                    print(f"  Early stopped at iter {i+1}/{max_iters} ({', '.join(stop_reason)})")
                    break
        return img_opt.detach()
    def _quality_refinement(self, img: torch.Tensor) -> torch.Tensor:
        smoothness_weight = self.config.get('quality', {}).get('freq_smoothness_weight', 0.1)
        from utils.image_processing import adaptive_low_pass_filter
        refined = adaptive_low_pass_filter(
            img,
            cutoff_freq=0.7,
            order=2
        )
        refined = 0.9 * refined + 0.1 * img
        return refined


def main():
    unmarker = EfficientUnMarker(config_path='configs/fast_cnn_only.yaml')
    print("\nGenerating dummy watermarked image (512x512)")
    dummy_img = torch.rand(1, 3, 512, 512)
    print("\nExecuting attack...")
    result, metrics = unmarker.attack(dummy_img)
    print("\nVERIFICATION")
    assert result.shape == dummy_img.shape, "Shape mismatch!"
    assert 0 <= result.min() <= result.max() <= 1, "Invalid range!"
    is_cuda = torch.cuda.is_available()
    time_target = 30 if is_cuda else 60
    print(f"\nPerformance Breakdown:")
    print(f"  Stage 1 (Frequency): {metrics.get('stage1_time', 0)*1000:.1f}ms")
    print(f"  Stage 2 (Attack): {metrics['stage2_time']:.2f}s")
    print(f"  Stage 3 (Refinement): {metrics.get('stage3_time', 0):.2f}s")
    print(f"  Total: {metrics['time']:.2f}s (target: < {time_target}s)")
    print(f"\nOptimization:")
    print(f"  Total iterations: {metrics['total_iterations']}")
    print(f"  Early stopped: {metrics['early_stopped']}")
    if metrics.get('detector_info'):
        print(f"\nDetector Info:")
        print(f"  Used deep branch: {metrics['detector_info'].get('used_deep', False)}")
        if metrics['detector_info'].get('quick_conf') is not None:
            print(f"  Quick confidence: {metrics['detector_info']['quick_conf'].item():.4f}")
    print(f"\nResults:")
    print(f"  Shape: {result.shape}")
    print(f"  Range: [{result.min():.3f}, {result.max():.3f}]")
    print(f"  Detection score: {metrics['detection_score']:.4f}")
    print(f"  Success: {metrics['success']}")
    if is_cuda:
        print(f"\nDevice: {torch.cuda.get_device_name(0)}")
        print("Phase 2 Integration Test PASSED on CUDA")
    else:
        print("\nDevice: CPU")
        print("Phase 2 Integration Test PASSED on CPU")
        print("Install CUDA 12.8+ PyTorch:")
        print("pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128")
    print("\nNext: Phase 3 - Benchmark & Ablation Studies")


if __name__ == '__main__':
    main()
