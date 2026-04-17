export interface OfferCopy {
  offer_id: string;
  customer_id: string;
  headline: string;
  body: string;
  cta: string;
  tone: string;
  category: string;
  prompt_version: number;
  model_version: string;
}

export interface FeedbackPayload {
  offer_id: string;
  customer_id: string;
  generated_copy: {
    headline: string;
    body: string;
    cta: string;
    tone: string;
  };
  rating: number;
  thumbs: "up" | "down";
  prompt_version: string;
  model_version: string;
}

export interface StatsResponse {
  record_count: number;
  avg_rating: number;
  thumbs_up_rate: number;
  thumbs_down_rate: number;
  by_prompt_version: Record<string, number>;
  by_model_version: Record<string, number>;
}
