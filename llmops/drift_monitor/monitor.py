"""Statistical drift detection for LLM score and feature distributions."""

from __future__ import annotations

import statistics

from shared.logger import get_logger

logger = get_logger(__name__)

_PSI_THRESHOLD = 0.2  # Population Stability Index — conventional alert boundary


def _psi(expected: list[float], actual: list[float], bins: int = 10) -> float:
    """Compute the Population Stability Index between two score distributions."""
    import numpy as np

    expected_arr = np.array(expected)
    actual_arr = np.array(actual)
    breakpoints = np.linspace(0, 1, bins + 1)
    expected_pct = np.histogram(expected_arr, breakpoints)[0] / len(expected_arr)
    actual_pct = np.histogram(actual_arr, breakpoints)[0] / len(actual_arr)
    # Avoid division by zero
    expected_pct = np.where(expected_pct == 0, 1e-6, expected_pct)
    actual_pct = np.where(actual_pct == 0, 1e-6, actual_pct)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


class DriftMonitor:
    """Detect score drift between a reference window and a current window."""

    def __init__(self, psi_threshold: float = _PSI_THRESHOLD) -> None:
        self._threshold = psi_threshold

    def check(
        self,
        reference_scores: list[float],
        current_scores: list[float],
    ) -> dict[str, object]:
        psi = _psi(reference_scores, current_scores)
        drifted = psi > self._threshold
        result: dict[str, object] = {
            "psi": round(psi, 4),
            "threshold": self._threshold,
            "drifted": drifted,
            "ref_mean": round(statistics.mean(reference_scores), 4),
            "cur_mean": round(statistics.mean(current_scores), 4),
        }
        logger.info("drift_check", **result)
        return result
