"""Build LLM prompts from customer context, features, and retrieved documents."""

from __future__ import annotations

from shared.schemas import CustomerProfile, FeatureVector


class PromptBuilder:
    """Assemble system + user messages for the offer-generation LLM call."""

    SYSTEM_PROMPT = (
        "You are a personalisation engine for a loyalty programme. "
        "Using the customer profile and relevant context provided, generate "
        "a concise, friendly, and highly personalised retention offer. "
        "Respond with JSON: {\"subject\": \"...\", \"body\": \"...\", \"offer_code\": \"...\"}"
    )

    def build(
        self,
        customer: CustomerProfile,
        features: FeatureVector | None,
        context_docs: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Return an OpenAI-compatible messages list."""
        context_text = "\n\n".join(d["text"] for d in context_docs)
        feature_summary = ""
        if features:
            pairs = zip(features.feature_names, features.values)
            feature_summary = ", ".join(f"{n}={v:.3f}" for n, v in pairs)

        user_content = (
            f"Customer tier: {customer.tier}\n"
            f"Total spend: ${customer.total_spend:.2f}\n"
            f"Visit count: {customer.visit_count}\n"
            f"Churn score: {customer.churn_score or 'unknown'}\n"
            f"Feature snapshot: {feature_summary or 'none'}\n\n"
            f"Relevant loyalty programme context:\n{context_text or 'No context available.'}\n\n"
            "Generate a personalised retention offer."
        )
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
