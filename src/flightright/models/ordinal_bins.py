# src/flightright/models/ordinal_bins.py

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

@dataclass
class DepDelayBinsOrdinalModel:
    thresholds: List[int]
    bin_labels: List[str]
    calibrators: Dict[int, object]
    bin_weights_minutes: Optional[List[float]] = None

    def predict_ge_proba(self, X) -> Dict[int, np.ndarray]:
        out = {}
        for thr in self.thresholds:
            cal = self.calibrators[thr]
            out[thr] = cal.predict_proba(X)[:, 1].astype(float)
        return out

    def predict_bin_proba(self, X) -> np.ndarray:
        from flightright.models.utils import ge_to_bins
        p_ge = self.predict_ge_proba(X)
        return ge_to_bins(
            p_ge[self.thresholds[0]],
            p_ge[self.thresholds[1]],
            p_ge[self.thresholds[2]],
            p_ge[self.thresholds[3]],
        )

    def predict_bin_label(self, X) -> List[str]:
        P = self.predict_bin_proba(X)
        idx = np.argmax(P, axis=1)
        return [self.bin_labels[i] for i in idx]

    def predict_expected_delay_minutes(self, X) -> np.ndarray:
        P = self.predict_bin_proba(X)
        if self.bin_weights_minutes is None:
            w = np.array([7.5, 22.5, 37.5, 52.5, 75.0], dtype=float)
        else:
            w = np.array(self.bin_weights_minutes, dtype=float)
        return (P * w[None, :]).sum(axis=1)