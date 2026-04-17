import type { FeedbackPayload, OfferCopy, StatsResponse } from "./types";

const GENERATOR_URL = "http://localhost:8004";
const FEEDBACK_URL = "/feedback"; // proxied via Vite to http://localhost:8005

const CATEGORIES = ["retention", "reactivation", "upsell", "birthday", "referral"];

const MOCK_OFFERS: OfferCopy[] = [
  {
    offer_id: "O001",
    customer_id: "C001",
    headline: "Your exclusive Gold reward is waiting",
    body: "As a valued Gold member, enjoy 20% off your next visit. Use code GOLD20 at checkout.",
    cta: "Redeem your reward →",
    tone: "warm",
    category: "retention",
    prompt_version: 1,
    model_version: "mock",
  },
  {
    offer_id: "O002",
    customer_id: "C002",
    headline: "We miss you — here's a treat",
    body: "It's been a while! Come back and enjoy a complimentary item with your next purchase.",
    cta: "Claim your treat →",
    tone: "friendly",
    category: "reactivation",
    prompt_version: 1,
    model_version: "mock",
  },
  {
    offer_id: "O003",
    customer_id: "C003",
    headline: "Double points this weekend only",
    body: "Earn 2× loyalty points on every purchase Saturday and Sunday. No code needed.",
    cta: "Start earning →",
    tone: "energetic",
    category: "upsell",
    prompt_version: 2,
    model_version: "mock",
  },
  {
    offer_id: "O004",
    customer_id: "C004",
    headline: "Happy birthday! A gift from us",
    body: "Celebrate your special day with a complimentary drink on us. Valid for 7 days.",
    cta: "Claim your birthday gift →",
    tone: "celebratory",
    category: "birthday",
    prompt_version: 2,
    model_version: "mock",
  },
  {
    offer_id: "O005",
    customer_id: "C005",
    headline: "Refer a friend, earn 500 Stars",
    body: "Every friend who joins using your code earns you 500 bonus Stars. Stars never expire.",
    cta: "Share your code →",
    tone: "warm",
    category: "referral",
    prompt_version: 1,
    model_version: "mock",
  },
];

let _mockIndex = 0;

export async function fetchOffer(): Promise<OfferCopy> {
  try {
    const offerId = `O${String(Math.floor(Math.random() * 5) + 1).padStart(3, "0")}`;
    const customerId = `C${String(Math.floor(Math.random() * 100) + 1).padStart(3, "0")}`;
    const res = await fetch(`${GENERATOR_URL}/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ customer_id: customerId, offer_id: offerId, prompt_version: 1 }),
      signal: AbortSignal.timeout(3000),
    });
    if (!res.ok) throw new Error(`Generator returned ${res.status}`);
    const data = await res.json();
    return {
      offer_id: offerId,
      customer_id: customerId,
      headline: data.headline ?? "",
      body: data.body ?? "",
      cta: data.cta ?? "",
      tone: data.tone ?? "",
      category: CATEGORIES[Math.floor(Math.random() * CATEGORIES.length)],
      prompt_version: data.prompt_version ?? 1,
      model_version: data.model_version ?? "unknown",
    };
  } catch {
    // Fall back to mock offers when the generator isn't running
    const offer = MOCK_OFFERS[_mockIndex % MOCK_OFFERS.length];
    _mockIndex++;
    return offer;
  }
}

export async function submitFeedback(payload: FeedbackPayload): Promise<void> {
  const res = await fetch(`${FEEDBACK_URL}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`Feedback API returned ${res.status}`);
}

export async function fetchStats(): Promise<StatsResponse> {
  const res = await fetch(`${FEEDBACK_URL}/stats`);
  if (!res.ok) throw new Error(`Stats API returned ${res.status}`);
  return res.json() as Promise<StatsResponse>;
}
