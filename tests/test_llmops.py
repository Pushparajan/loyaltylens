"""Tests for llmops: PSI drift detection, eval harness, prompt registry CLI."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from click.testing import CliRunner

from llmops.drift_monitor.monitor import PropensityDriftMonitor, _psi
from llmops.eval_harness.evaluator import OfferCopyEvaluator, OfferCopyInput
from llmops.eval_harness.run_eval import main as eval_main
from llmops.prompt_registry.cli import cli


# ── PSI drift detection ────────────────────────────────────────────────────────


class TestPSI:
    def test_identical_distributions_near_zero(self) -> None:
        rng = np.random.default_rng(0)
        scores = rng.uniform(0, 1, 1000).tolist()
        psi = _psi(scores, scores)
        assert psi < 0.01

    def test_shifted_distribution_detects_drift(self) -> None:
        rng = np.random.default_rng(1)
        baseline = rng.normal(0.5, 0.1, 2000).clip(0, 1).tolist()
        current = rng.normal(0.8, 0.1, 500).clip(0, 1).tolist()
        psi = _psi(baseline, current)
        assert psi > 0.20, f"Expected PSI > 0.20 for shifted distributions, got {psi:.4f}"

    def test_psi_non_negative(self) -> None:
        rng = np.random.default_rng(2)
        a = rng.uniform(0, 1, 500).tolist()
        b = rng.uniform(0.2, 0.8, 500).tolist()
        assert _psi(a, b) >= 0.0


class TestPropensityDriftMonitor:
    def test_ok_status_for_stable_distribution(self) -> None:
        rng = np.random.default_rng(10)
        baseline = rng.normal(0.55, 0.15, 3000).clip(0, 1)
        current = rng.normal(0.55, 0.15, 1000).clip(0, 1)
        monitor = PropensityDriftMonitor()
        report = monitor.check_drift(baseline, current, "2026-01-01", "2026-04-17")
        assert report.status == "ok"
        assert report.psi < 0.10

    def test_critical_status_for_large_shift(self) -> None:
        rng = np.random.default_rng(20)
        baseline = rng.normal(0.4, 0.1, 3000).clip(0, 1)
        current = rng.normal(0.8, 0.1, 1000).clip(0, 1)
        monitor = PropensityDriftMonitor()
        report = monitor.check_drift(baseline, current)
        assert report.status == "critical"
        assert report.psi > 0.20

    def test_report_fields_populated(self) -> None:
        scores = [0.5] * 100
        monitor = PropensityDriftMonitor()
        report = monitor.check_drift(
            scores, scores, baseline_date="2026-01-01", current_date="2026-04-17"
        )
        assert report.baseline_date == "2026-01-01"
        assert report.current_date == "2026-04-17"
        assert report.n_baseline == 100
        assert report.n_current == 100

    def test_compute_psi_symmetric_for_identical(self) -> None:
        scores = np.linspace(0, 1, 200).tolist()
        monitor = PropensityDriftMonitor()
        psi = monitor.compute_psi(scores, scores)
        assert psi < 0.01

    def test_feature_breakdown_included(self) -> None:
        rng = np.random.default_rng(30)
        base = rng.uniform(0, 1, 500).tolist()
        curr = rng.uniform(0, 1, 200).tolist()
        monitor = PropensityDriftMonitor()
        report = monitor.check_drift(
            base, curr,
            feature_scores={"recency_days": (base, curr), "frequency_30d": (base, curr)},
        )
        assert "recency_days" in report.feature_breakdown
        assert "frequency_30d" in report.feature_breakdown


# ── OfferCopyEvaluator ─────────────────────────────────────────────────────────


class TestOfferCopyEvaluator:
    _COPY = OfferCopyInput(
        headline="Double Star Day",
        body="Earn 2x stars on every purchase this Saturday. No code needed.",
        cta="Start earning →",
    )
    _REFERENCE = "Earn double loyalty stars this Saturday. No code needed. Rewards applied automatically."

    def test_bleu_identical_text_is_one(self) -> None:
        ev = OfferCopyEvaluator()
        score = ev.bleu_score("hello world offer", "hello world offer")
        assert score == pytest.approx(1.0, abs=0.01)

    def test_bleu_disjoint_text_is_low(self) -> None:
        ev = OfferCopyEvaluator()
        score = ev.bleu_score("abc def ghi", "xyz uvw rst")
        assert score < 0.05

    def test_rouge_l_identical_is_one(self) -> None:
        ev = OfferCopyEvaluator()
        assert ev.rouge_l("hello world", "hello world") == pytest.approx(1.0, abs=0.01)

    def test_rouge_l_in_range(self) -> None:
        ev = OfferCopyEvaluator()
        score = ev.rouge_l(self._COPY.body, self._REFERENCE)
        assert 0.0 <= score <= 1.0

    def test_llm_judge_returns_defaults_without_backend(self) -> None:
        ev = OfferCopyEvaluator(judge_backend=None)
        scores = ev.llm_judge(self._COPY)
        assert set(scores) == {"coherence", "brand_alignment", "cta_strength"}
        assert all(0.0 <= v <= 1.0 for v in scores.values())

    def test_aggregate_in_range(self) -> None:
        ev = OfferCopyEvaluator()
        agg = ev.aggregate_score(self._COPY, self._REFERENCE)
        assert 0.0 <= agg <= 1.0

    def test_evaluate_returns_eval_result(self) -> None:
        ev = OfferCopyEvaluator()
        result = ev.evaluate(self._COPY, self._REFERENCE)
        assert result.aggregate == pytest.approx(
            0.2 * result.bleu
            + 0.2 * result.rouge_l
            + 0.6 * ((result.coherence + result.brand_alignment + result.cta_strength) / 3),
            abs=1e-3,
        )

    def test_llm_judge_fallback_on_backend_error(self) -> None:
        class _BadBackend:
            def generate(self, messages: list[dict[str, str]]) -> str:
                raise RuntimeError("network error")

        ev = OfferCopyEvaluator(judge_backend=_BadBackend())
        scores = ev.llm_judge(self._COPY)
        assert scores == {"coherence": 0.75, "brand_alignment": 0.75, "cta_strength": 0.75}


# ── Eval harness main() ────────────────────────────────────────────────────────


class TestEvalHarnessMain:
    def test_passes_at_default_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("llmops.eval_harness.run_eval._RESULTS_DIR", Path(tmpdir)):
                code = eval_main(["--threshold", "0.75", "--n", "5"])
        assert code == 0

    def test_fails_at_impossibly_high_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("llmops.eval_harness.run_eval._RESULTS_DIR", Path(tmpdir)):
                code = eval_main(["--threshold", "0.999", "--n", "5"])
        assert code == 1

    def test_writes_result_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = Path(tmpdir)
            with patch("llmops.eval_harness.run_eval._RESULTS_DIR", results_path):
                eval_main(["--threshold", "0.75", "--n", "3"])
            files = list(results_path.glob("eval_*.json"))
            assert len(files) == 1
            data = json.loads(files[0].read_text())
        assert "mean_aggregate_score" in data
        assert data["n"] == 3


# ── Prompt registry CLI ───────────────────────────────────────────────────────


class TestPromptRegistryCLI:
    """Tests run against temporary copies of active.json / history.json."""

    def _make_registry(self, tmp_path: Path) -> tuple[Path, Path]:
        active = tmp_path / "active.json"
        history = tmp_path / "history.json"
        active.write_text(json.dumps({"version": "v1"}))
        history.write_text(
            json.dumps(
                [
                    {"version": "v1", "activated_at": "2026-04-01T09:00:00Z", "previous": None},
                ]
            )
        )
        return active, history

    def test_list_shows_versions(self, tmp_path: Path) -> None:
        active, history = self._make_registry(tmp_path)
        runner = CliRunner()
        with (
            patch("llmops.prompt_registry.cli._ACTIVE_FILE", active),
            patch("llmops.prompt_registry.cli._HISTORY_FILE", history),
            patch(
                "llmops.prompt_registry.cli._discover_versions",
                return_value=["v1", "v2"],
            ),
        ):
            result = runner.invoke(cli, ["prompt", "list"])
        assert result.exit_code == 0
        assert "v1" in result.output
        assert "active" in result.output

    def test_activate_updates_active_json(self, tmp_path: Path) -> None:
        active, history = self._make_registry(tmp_path)
        runner = CliRunner()
        with (
            patch("llmops.prompt_registry.cli._ACTIVE_FILE", active),
            patch("llmops.prompt_registry.cli._HISTORY_FILE", history),
            patch(
                "llmops.prompt_registry.cli._prompt_path",
                side_effect=lambda v: tmp_path / f"system_{v}.yaml",
            ),
        ):
            # Create a fake v2 YAML so path.exists() returns True
            (tmp_path / "system_v2.yaml").write_text("version: 2\nsystem: test\nuser_template: test\n")
            result = runner.invoke(cli, ["prompt", "activate", "v2"])
        assert result.exit_code == 0, result.output
        assert json.loads(active.read_text())["version"] == "v2"

    def test_rollback_reverts_to_previous(self, tmp_path: Path) -> None:
        active, history = self._make_registry(tmp_path)
        # Set up: v2 is active, previous was v1
        active.write_text(json.dumps({"version": "v2"}))
        history.write_text(
            json.dumps(
                [
                    {"version": "v1", "activated_at": "2026-04-01T09:00:00Z", "previous": None},
                    {"version": "v2", "activated_at": "2026-04-10T14:00:00Z", "previous": "v1"},
                ]
            )
        )
        runner = CliRunner()
        with (
            patch("llmops.prompt_registry.cli._ACTIVE_FILE", active),
            patch("llmops.prompt_registry.cli._HISTORY_FILE", history),
            patch(
                "llmops.prompt_registry.cli._prompt_path",
                side_effect=lambda v: tmp_path / f"system_{v}.yaml",
            ),
        ):
            (tmp_path / "system_v1.yaml").write_text("version: 1\nsystem: test\nuser_template: test\n")
            result = runner.invoke(cli, ["prompt", "rollback"])
        assert result.exit_code == 0, result.output
        assert json.loads(active.read_text())["version"] == "v1"

    def test_diff_shows_changes(self, tmp_path: Path) -> None:
        v1_yaml = "version: 1\nsystem: |\n  You are a copywriter.\nuser_template: |\n  Write an offer.\n"
        v2_yaml = "version: 2\nsystem: |\n  You are a concise copywriter.\nuser_template: |\n  Write an offer.\n"
        (tmp_path / "system_v1.yaml").write_text(v1_yaml)
        (tmp_path / "system_v2.yaml").write_text(v2_yaml)
        runner = CliRunner()
        with patch("llmops.prompt_registry.cli._prompt_path", side_effect=lambda v: tmp_path / f"system_{v}.yaml"):
            result = runner.invoke(cli, ["prompt", "diff", "v1", "v2"])
        assert result.exit_code == 0, result.output
        assert "concise" in result.output
