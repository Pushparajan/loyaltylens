import { useCallback, useEffect, useState } from "react";
import { fetchOffer, fetchStats, submitFeedback } from "./api";
import type { OfferCopy, StatsResponse } from "./types";

// ── Star rating ───────────────────────────────────────────────────────────────

function StarRating({
  value,
  onChange,
}: {
  value: number;
  onChange: (n: number) => void;
}) {
  const [hovered, setHovered] = useState(0);
  return (
    <div className="flex gap-1">
      {[1, 2, 3, 4, 5].map((n) => (
        <button
          key={n}
          type="button"
          className={`text-2xl transition-colors ${
            n <= (hovered || value) ? "text-brand-gold" : "text-gray-300"
          }`}
          onMouseEnter={() => setHovered(n)}
          onMouseLeave={() => setHovered(0)}
          onClick={() => onChange(n)}
          aria-label={`${n} star${n > 1 ? "s" : ""}`}
        >
          ★
        </button>
      ))}
    </div>
  );
}

// ── Bar chart (SVG, no library) ───────────────────────────────────────────────

function BarChart({ data }: { data: Record<string, number> }) {
  const entries = Object.entries(data).sort((a, b) => a[0].localeCompare(b[0]));
  if (entries.length === 0)
    return <p className="text-sm text-gray-400 italic">No data yet.</p>;

  const maxVal = Math.max(...entries.map(([, v]) => v), 5);
  const barH = 120;
  const barW = 48;
  const gap = 24;
  const svgW = entries.length * (barW + gap) + gap;

  return (
    <svg
      width={svgW}
      height={barH + 40}
      role="img"
      aria-label="Average rating by prompt version"
    >
      {entries.map(([label, val], i) => {
        const x = gap + i * (barW + gap);
        const h = Math.round((val / maxVal) * barH);
        const y = barH - h;
        return (
          <g key={label}>
            <rect x={x} y={y} width={barW} height={h} rx={4} fill="#00704A" />
            <text
              x={x + barW / 2}
              y={y - 4}
              textAnchor="middle"
              fontSize={11}
              fill="#374151"
            >
              {val.toFixed(2)}
            </text>
            <text
              x={x + barW / 2}
              y={barH + 18}
              textAnchor="middle"
              fontSize={11}
              fill="#6B7280"
            >
              {label}
            </text>
          </g>
        );
      })}
      {/* y-axis reference line */}
      <line x1={0} y1={barH} x2={svgW} y2={barH} stroke="#E5E7EB" strokeWidth={1} />
    </svg>
  );
}

// ── Review tab ────────────────────────────────────────────────────────────────

function ReviewTab() {
  const [offer, setOffer] = useState<OfferCopy | null>(null);
  const [loading, setLoading] = useState(false);
  const [rating, setRating] = useState(0);
  const [thumbs, setThumbs] = useState<"up" | "down" | null>(null);
  const [submitted, setSubmitted] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadOffer = useCallback(async () => {
    setLoading(true);
    setError(null);
    setRating(0);
    setThumbs(null);
    setSubmitted(false);
    try {
      setOffer(await fetchOffer());
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadOffer();
  }, [loadOffer]);

  const handleSubmit = async () => {
    if (!offer || thumbs === null || rating === 0) return;
    setSubmitting(true);
    setError(null);
    try {
      await submitFeedback({
        offer_id: offer.offer_id,
        customer_id: offer.customer_id,
        generated_copy: {
          headline: offer.headline,
          body: offer.body,
          cta: offer.cta,
          tone: offer.tone,
        },
        rating,
        thumbs,
        prompt_version: String(offer.prompt_version),
        model_version: offer.model_version,
      });
      setSubmitted(true);
    } catch (e) {
      setError(String(e));
    } finally {
      setSubmitting(false);
    }
  };

  if (loading)
    return (
      <div className="flex items-center justify-center h-48 text-gray-400">
        Loading offer…
      </div>
    );

  if (!offer) return null;

  return (
    <div className="max-w-xl mx-auto space-y-6">
      {/* offer card */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6 space-y-3">
        <span className="inline-block text-xs font-medium uppercase tracking-wide text-brand-green bg-brand-cream px-2 py-0.5 rounded">
          {offer.category}
        </span>
        <h2 className="text-xl font-semibold text-gray-900">{offer.headline}</h2>
        <p className="text-gray-600 leading-relaxed">{offer.body}</p>
        <p className="text-brand-green font-medium">{offer.cta}</p>
        <p className="text-xs text-gray-400 pt-1">
          prompt v{offer.prompt_version} · model {offer.model_version}
        </p>
      </div>

      {submitted ? (
        <div className="bg-green-50 border border-green-200 rounded-xl p-4 text-center space-y-3">
          <p className="text-green-700 font-medium">Feedback submitted ✓</p>
          <button
            onClick={() => void loadOffer()}
            className="bg-brand-green text-white text-sm px-5 py-2 rounded-lg hover:bg-green-700 transition-colors"
          >
            Next offer →
          </button>
        </div>
      ) : (
        <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6 space-y-5">
          {/* thumbs */}
          <div>
            <p className="text-sm font-medium text-gray-700 mb-2">
              Would you send this offer?
            </p>
            <div className="flex gap-3">
              {(["up", "down"] as const).map((t) => (
                <button
                  key={t}
                  type="button"
                  onClick={() => setThumbs(t)}
                  className={`flex items-center gap-1.5 px-4 py-2 rounded-lg border text-sm font-medium transition-all ${
                    thumbs === t
                      ? t === "up"
                        ? "bg-green-600 border-green-600 text-white"
                        : "bg-red-500 border-red-500 text-white"
                      : "border-gray-200 text-gray-600 hover:border-gray-400"
                  }`}
                >
                  {t === "up" ? "👍 Yes" : "👎 No"}
                </button>
              ))}
            </div>
          </div>

          {/* star rating */}
          <div>
            <p className="text-sm font-medium text-gray-700 mb-2">
              Rate the copy quality
            </p>
            <StarRating value={rating} onChange={setRating} />
          </div>

          {error && <p className="text-red-500 text-sm">{error}</p>}

          <button
            onClick={() => void handleSubmit()}
            disabled={thumbs === null || rating === 0 || submitting}
            className="w-full bg-brand-green text-white font-medium py-2.5 rounded-lg hover:bg-green-700 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {submitting ? "Submitting…" : "Submit feedback"}
          </button>
        </div>
      )}
    </div>
  );
}

// ── Dashboard tab ─────────────────────────────────────────────────────────────

function DashboardTab() {
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchStats()
      .then(setStats)
      .catch((e: unknown) => setError(String(e)))
      .finally(() => setLoading(false));
  }, []);

  if (loading)
    return (
      <div className="flex items-center justify-center h-48 text-gray-400">
        Loading stats…
      </div>
    );
  if (error)
    return <p className="text-red-500 text-sm text-center">{error}</p>;
  if (!stats || stats.record_count === 0)
    return (
      <p className="text-gray-400 text-sm text-center italic">
        No feedback collected yet. Review some offers first.
      </p>
    );

  return (
    <div className="max-w-2xl mx-auto space-y-8">
      {/* summary metrics */}
      <div className="grid grid-cols-3 gap-4">
        {[
          { label: "Records", value: String(stats.record_count) },
          { label: "Avg Rating", value: stats.avg_rating.toFixed(2) + " / 5" },
          {
            label: "👍 Rate",
            value: (stats.thumbs_up_rate * 100).toFixed(0) + "%",
          },
        ].map(({ label, value }) => (
          <div
            key={label}
            className="bg-white rounded-xl border border-gray-100 shadow-sm p-4 text-center"
          >
            <p className="text-2xl font-bold text-gray-900">{value}</p>
            <p className="text-xs text-gray-500 mt-1">{label}</p>
          </div>
        ))}
      </div>

      {/* bar chart */}
      <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-6">
        <h3 className="text-sm font-semibold text-gray-700 mb-4">
          Avg Rating by Prompt Version
        </h3>
        <BarChart data={stats.by_prompt_version} />
      </div>

      {/* model version table */}
      {Object.keys(stats.by_model_version).length > 0 && (
        <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-6">
          <h3 className="text-sm font-semibold text-gray-700 mb-3">
            Avg Rating by Model Version
          </h3>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-gray-400 text-xs border-b">
                <th className="pb-2">Model</th>
                <th className="pb-2 text-right">Avg Rating</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(stats.by_model_version).map(([model, avg]) => (
                <tr key={model} className="border-b last:border-0">
                  <td className="py-2 text-gray-700">{model}</td>
                  <td className="py-2 text-right font-medium text-gray-900">
                    {avg.toFixed(2)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

// ── Root app ──────────────────────────────────────────────────────────────────

type Tab = "review" | "dashboard";

export default function App() {
  const [tab, setTab] = useState<Tab>("review");

  return (
    <div className="min-h-screen bg-gray-50">
      {/* header */}
      <header className="bg-brand-green text-white px-6 py-4 shadow-sm">
        <div className="max-w-2xl mx-auto flex items-center justify-between">
          <div>
            <h1 className="text-lg font-bold tracking-tight">LoyaltyLens</h1>
            <p className="text-green-200 text-xs">Offer Feedback Tool</p>
          </div>
          <nav className="flex gap-1 bg-green-700/40 rounded-lg p-1">
            {(["review", "dashboard"] as Tab[]).map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className={`px-4 py-1.5 rounded-md text-sm font-medium transition-colors capitalize ${
                  tab === t ? "bg-white text-brand-green" : "text-green-100 hover:text-white"
                }`}
              >
                {t}
              </button>
            ))}
          </nav>
        </div>
      </header>

      <main className="max-w-2xl mx-auto px-4 py-8">
        {tab === "review" ? <ReviewTab /> : <DashboardTab />}
      </main>
    </div>
  );
}
