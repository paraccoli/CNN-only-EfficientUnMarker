import torch
from typing import Optional, List
from collections import deque


class EarlyStopping:
    def __init__(
        self,
        check_interval: int = 5,
        detection_threshold: float = 0.5,
        variance_threshold: float = 1e-4,
        patience: int = 5,
        min_iterations: int = 10,
        psnr_min: float = None,
        psnr_patience: int = 2
    ):
        self.check_interval = check_interval
        self.detection_threshold = detection_threshold
        self.variance_threshold = variance_threshold
        self.patience = patience
        self.min_iterations = min_iterations
        self.psnr_min = psnr_min
        self.psnr_patience = psnr_patience
        self.loss_history = deque(maxlen=patience)
        self.detection_history = deque(maxlen=patience)
        self.patience_counter = 0
        self.best_detection_score = float('inf')
        self.psnr_counter = 0
        self.total_checks = 0
        self.stopped_by_detection = False
        self.stopped_by_convergence = False
        self.stopped_by_quality = False
    def should_stop(
        self,
        iteration: int,
        loss: Optional[float] = None,
        detection_score: Optional[float] = None,
        psnr: Optional[float] = None
    ) -> bool:
        if iteration < self.min_iterations:
            return False
        if (iteration + 1) % self.check_interval != 0:
            return False
        self.total_checks += 1
        if loss is not None:
            self.loss_history.append(loss)
        if detection_score is not None:
            self.detection_history.append(detection_score)
            if detection_score < self.best_detection_score:
                self.best_detection_score = detection_score
                self.patience_counter = 0
        if detection_score is not None and detection_score < self.detection_threshold:
            self.stopped_by_detection = True
            return True
        if self.psnr_min is not None and psnr is not None:
            if psnr < self.psnr_min:
                self.psnr_counter += 1
            else:
                self.psnr_counter = 0
            if self.psnr_counter >= self.psnr_patience:
                self.stopped_by_quality = True
                return True
        else:
            self.psnr_counter = 0
        if len(self.loss_history) >= self.patience:
            loss_var = torch.tensor(list(self.loss_history)).var().item()
            if loss_var < self.variance_threshold:
                self.stopped_by_convergence = True
                return True
        if detection_score is not None:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                return True
        return False
    def reset(self):
        self.loss_history.clear()
        self.detection_history.clear()
        self.patience_counter = 0
        self.psnr_counter = 0
        self.best_detection_score = float('inf')
        self.total_checks = 0
        self.stopped_by_detection = False
        self.stopped_by_convergence = False
        self.stopped_by_quality = False
    def get_stats(self) -> dict:
        return {
            'total_checks': self.total_checks,
            'stopped_by_detection': self.stopped_by_detection,
            'stopped_by_convergence': self.stopped_by_convergence,
            'stopped_by_quality': self.stopped_by_quality,
            'best_detection_score': self.best_detection_score,
            'final_loss_variance': (
                torch.tensor(list(self.loss_history)).var().item()
                if len(self.loss_history) > 1 else float('inf')
            )
        }


class AdaptiveThreshold:
    def __init__(
        self,
        initial_threshold: float = 0.5,
        adjustment_rate: float = 0.05,
        window_size: int = 10
    ):
        self.current_threshold = initial_threshold
        self.adjustment_rate = adjustment_rate
        self.history = deque(maxlen=window_size)
    def update(self, detection_score: float) -> float:
        self.history.append(detection_score)
        if len(self.history) < 3:
            return self.current_threshold
        recent_scores = list(self.history)[-5:]
        if all(s > self.current_threshold + 0.1 for s in recent_scores):
            self.current_threshold += self.adjustment_rate
            self.current_threshold = min(0.7, self.current_threshold)
        elif all(s < self.current_threshold - 0.1 for s in recent_scores):
            self.current_threshold -= self.adjustment_rate
            self.current_threshold = max(0.3, self.current_threshold)
        return self.current_threshold
    def get_threshold(self) -> float:
        return self.current_threshold
    def reset(self):
        self.history.clear()


class ProgressTracker:
    def __init__(self, log_interval: int = 10):
        self.log_interval = log_interval
        self.iteration = 0
        self.losses = []
        self.detection_scores = []
    def update(
        self,
        loss: Optional[float] = None,
        detection_score: Optional[float] = None
    ):
        self.iteration += 1
        if loss is not None:
            self.losses.append(loss)
        if detection_score is not None:
            self.detection_scores.append(detection_score)
        if self.iteration % self.log_interval == 0:
            msg = f"Iter {self.iteration}"
            if self.losses:
                msg += f" | Loss: {self.losses[-1]:.4f}"
            if self.detection_scores:
                msg += f" | Detection: {self.detection_scores[-1]:.4f}"
            print(msg)
    def get_summary(self) -> dict:
        summary = {
            'total_iterations': self.iteration,
            'final_loss': self.losses[-1] if self.losses else None,
            'final_detection': self.detection_scores[-1] if self.detection_scores else None,
        }
        if len(self.losses) > 1:
            summary['loss_improvement'] = self.losses[0] - self.losses[-1]
        if len(self.detection_scores) > 1:
            summary['detection_improvement'] = self.detection_scores[0] - self.detection_scores[-1]
        return summary
    def reset(self):
        self.iteration = 0
        self.losses = []
        self.detection_scores = []