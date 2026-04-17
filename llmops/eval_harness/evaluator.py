"""OfferCopyEvaluator: BLEU, ROUGE-L, and LLM-as-judge scoring for offer copy."""

from __future__ import annotations

import json
from dataclasses import dataclass

from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer

from shared.logger import get_logger

logger = get_logger(__name__)

_JUDGE_PROMPT = """\
You are evaluating loyalty offer copy. Score the following on three dimensions from 1 to 5.

Headline: {headline}
Body: {body}
CTA: {cta}

Scoring rubric:
- coherence (1-5): Natural flow, grammatically correct. 5=flawless, 1=confusing.
- brand_alignment (1-5): Warm, community-focused brand voice. 5=on-brand, 1=off-brand or pushy.
- cta_strength (1-5): Clear, appropriate urgency. 5=compelling, 1=weak or missing.

Respond ONLY with valid JSON: {{"coherence": N, "brand_alignment": N, "cta_strength": N}}"""

_DEFAULT_JUDGE_SCORES: dict[str, float] = {
    "coherence": 0.75,
    "brand_alignment": 0.75,
    "cta_strength": 0.75,
}


@dataclass
class EvalResult:
    bleu: float
    rouge_l: float
    coherence: float
    brand_alignment: float
    cta_strength: float
    aggregate: float


@dataclass
class OfferCopyInput:
    """Minimal offer copy fields needed for evaluation (avoids importing llm_generator)."""

    headline: str
    body: str
    cta: str


class OfferCopyEvaluator:
    """Score offer copy using lexical metrics and an optional LLM-as-judge."""

    def __init__(self, judge_backend: object | None = None) -> None:
        self._judge = judge_backend
        self._bleu = BLEU(effective_order=True)
        self._rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def bleu_score(self, generated: str, reference: str) -> float:
        """BLEU score in [0, 1] (sacrebleu sentence BLEU / 100)."""
        return self._bleu.sentence_score(generated, [reference]).score / 100.0

    def rouge_l(self, generated: str, reference: str) -> float:
        """ROUGE-L F-measure in [0, 1]."""
        return float(self._rouge.score(generated, reference)["rougeL"].fmeasure)

    def llm_judge(self, copy: OfferCopyInput) -> dict[str, float]:
        """Call LLM judge; fall back to neutral defaults if backend unavailable."""
        if self._judge is None:
            return dict(_DEFAULT_JUDGE_SCORES)

        prompt = _JUDGE_PROMPT.format(
            headline=copy.headline, body=copy.body, cta=copy.cta
        )
        try:
            messages = [{"role": "user", "content": prompt}]
            raw: str = self._judge.generate(messages)  # type: ignore[union-attr]
            scores = json.loads(raw)
            return {k: float(scores[k]) / 5.0 for k in ("coherence", "brand_alignment", "cta_strength")}
        except Exception as exc:
            logger.warning("llm_judge_fallback", error=str(exc))
            return dict(_DEFAULT_JUDGE_SCORES)

    def aggregate_score(self, copy: OfferCopyInput, reference: str) -> float:
        """0.2 × BLEU + 0.2 × ROUGE-L + 0.6 × mean(LLM-judge dimensions)."""
        b = self.bleu_score(copy.body, reference)
        r = self.rouge_l(copy.body, reference)
        judge = self.llm_judge(copy)
        judge_avg = sum(judge.values()) / len(judge)
        return round(0.2 * b + 0.2 * r + 0.6 * judge_avg, 4)

    def evaluate(self, copy: OfferCopyInput, reference: str) -> EvalResult:
        """Return a full EvalResult with all component scores."""
        b = self.bleu_score(copy.body, reference)
        r = self.rouge_l(copy.body, reference)
        judge = self.llm_judge(copy)
        judge_avg = sum(judge.values()) / len(judge)
        agg = round(0.2 * b + 0.2 * r + 0.6 * judge_avg, 4)
        return EvalResult(
            bleu=round(b, 4),
            rouge_l=round(r, 4),
            coherence=judge["coherence"],
            brand_alignment=judge["brand_alignment"],
            cta_strength=judge["cta_strength"],
            aggregate=agg,
        )
