"""Statistical drift detection for propensity score and feature distributions."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

import numpy as np

from shared.logger import get_logger

logger = get_logger(__name__)

_PSI_THRESHOLD_WARNING = 0.10
_PSI_THRESHOLD_CRITICAL = 0.20


def _psi(expected: list[float], actual: list[float], bins: int = 10) -> float:
    """Compute Population Stability Index between two score distributions."""
    expected_arr = np.array(expected)
    actual_arr = np.array(actual)
    breakpoints = np.linspace(0, 1, bins + 1)
    expected_pct = np.histogram(expected_arr, breakpoints)[0] / len(expected_arr)
    actual_pct = np.histogram(actual_arr, breakpoints)[0] / len(actual_arr)
    expected_pct = np.where(expected_pct == 0, 1e-6, expected_pct)
    actual_pct = np.where(actual_pct == 0, 1e-6, actual_pct)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


class DriftMonitor:
    """Detect score drift between a reference window and a current window."""

    def __init__(self, psi_threshold: float = _PSI_THRESHOLD_CRITICAL) -> None:
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


@dataclass
class DriftReport:
    psi: float
    status: str  # "ok" | "warning" | "critical"
    baseline_date: str
    current_date: str
    n_baseline: int
    n_current: int
    feature_breakdown: dict[str, float] = field(default_factory=dict)


class PropensityDriftMonitor:
    """PSI-based drift monitor for propensity score distributions."""

    def __init__(
        self,
        warning_threshold: float = _PSI_THRESHOLD_WARNING,
        critical_threshold: float = _PSI_THRESHOLD_CRITICAL,
    ) -> None:
        self._warning = warning_threshold
        self._critical = critical_threshold

    def compute_psi(
        self,
        baseline_scores: list[float] | np.ndarray,
        current_scores: list[float] | np.ndarray,
        bins: int = 10,
    ) -> float:
        """Return PSI between baseline and current score distributions."""
        baseline_arr = np.asarray(baseline_scores, dtype=float)
        current_arr = np.asarray(current_scores, dtype=float)
        breakpoints = np.linspace(0.0, 1.0, bins + 1)

        baseline_pct = np.histogram(baseline_arr, bins=breakpoints)[0] / len(baseline_arr)
        current_pct = np.histogram(current_arr, bins=breakpoints)[0] / len(current_arr)

        baseline_pct = np.clip(baseline_pct, 1e-6, None)
        current_pct = np.clip(current_pct, 1e-6, None)

        return float(np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct)))

    def check_drift(
        self,
        baseline_scores: list[float] | np.ndarray,
        current_scores: list[float] | np.ndarray,
        baseline_date: str = "",
        current_date: str = "",
        feature_scores: dict[str, tuple[list[float], list[float]]] | None = None,
    ) -> DriftReport:
        """Compute PSI and return a DriftReport with status classification.

        Args:
            baseline_scores: Reference distribution (e.g., last 30-day window).
            current_scores: Current distribution to compare against baseline.
            baseline_date: ISO date string for the baseline window.
            current_date: ISO date string for the current window.
            feature_scores: Optional per-feature (baseline, current) pairs for breakdown.
        """
        psi = self.compute_psi(baseline_scores, current_scores)

        if psi > self._critical:
            status = "critical"
        elif psi > self._warning:
            status = "warning"
        else:
            status = "ok"

        feature_breakdown: dict[str, float] = {}
        if feature_scores:
            feature_breakdown = {
                name: round(self.compute_psi(base, curr), 4)
                for name, (base, curr) in feature_scores.items()
            }

        report = DriftReport(
            psi=round(psi, 4),
            status=status,
            baseline_date=baseline_date,
            current_date=current_date,
            n_baseline=len(np.asarray(baseline_scores)),
            n_current=len(np.asarray(current_scores)),
            feature_breakdown=feature_breakdown,
        )
        logger.info(
            "propensity_drift_check",
            psi=report.psi,
            status=report.status,
            n_baseline=report.n_baseline,
            n_current=report.n_current,
        )
        return report
