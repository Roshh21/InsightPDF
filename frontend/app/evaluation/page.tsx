"use client";

import { useEffect, useState } from "react";
import { BarChart3 } from "lucide-react";
import { api, ApiError } from "@/lib/api";
import type { EvaluationSummary } from "@/lib/types";
import { EmptyState, PageHeader, Spinner } from "@/components/ui";
import { formatLatency, formatPct } from "@/lib/format";

export default function EvaluationPage() {
  const [summary, setSummary] = useState<EvaluationSummary | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .getEvaluationSummary()
      .then(setSummary)
      .catch((e) => setError(e instanceof ApiError ? e.message : "Failed to load evaluation summary."));
  }, []);

  return (
    <div className="mx-auto max-w-4xl px-8 py-10">
      <PageHeader
        eyebrow="Observability"
        title="Evaluation"
        description="Computed from actual retrieval scores, citation checks, and recorded runs — never fabricated."
      />

      {error && <div className="mb-4 rounded-[5px] border border-danger/30 bg-danger-soft px-3 py-2 text-[13px] text-danger">{error}</div>}

      {!summary ? (
        <Spinner size={16} />
      ) : !summary.has_data ? (
        <EmptyState
          icon={BarChart3}
          title="No data yet"
          description="Run a chat turn or a tool to start populating real evaluation metrics."
        />
      ) : (
        <>
          <div className="mb-6 grid grid-cols-2 gap-3 sm:grid-cols-3">
            <MetricCard label="Retrieval Relevance" value={formatPct(summary.retrieval_relevance_avg)} hint="Avg. similarity score of retrieved evidence" />
            <MetricCard label="Answer Faithfulness" value={formatPct(summary.answer_faithfulness_avg)} hint="Embedding-similarity proxy vs. retrieved evidence" />
            <MetricCard label="Citation Accuracy" value={formatPct(summary.citation_accuracy_avg)} hint="Share of citations verified against source text" />
            <MetricCard label="Average Latency" value={formatLatency(summary.avg_latency_ms)} hint="End-to-end per run" />
            <MetricCard label="Recorded Runs" value={String(summary.total_runs)} hint="Chat turns + tool executions evaluated" />
            <MetricCard label="Failed Runs" value={String(summary.failed_runs)} hint="Provider calls that ultimately failed" />
          </div>

          <div className="card p-4">
            <div className="mb-3 font-serif text-[14.5px] font-semibold text-ink">Primary vs. fallback provider usage</div>
            <div className="mb-2 flex h-3 w-full overflow-hidden rounded-full bg-paper-dim">
              <div className="h-full bg-cobalt" style={{ width: `${summary.primary_request_pct ?? 0}%` }} />
              <div className="h-full bg-amber" style={{ width: `${summary.fallback_request_pct ?? 0}%` }} />
            </div>
            <div className="flex flex-wrap items-center gap-4 text-[12.5px] text-ink-soft">
              <span className="flex items-center gap-1.5">
                <span className="h-2 w-2 rounded-full bg-cobalt" /> Primary — {summary.primary_requests} requests (
                {formatPct((summary.primary_request_pct ?? 0) / 100)})
              </span>
              <span className="flex items-center gap-1.5">
                <span className="h-2 w-2 rounded-full bg-amber" /> Fallback — {summary.fallback_requests} requests (
                {formatPct((summary.fallback_request_pct ?? 0) / 100)})
              </span>
            </div>
            {Object.keys(summary.provider_breakdown).length > 0 && (
              <div className="mt-3 flex flex-wrap gap-1.5 border-t border-line pt-3">
                {Object.entries(summary.provider_breakdown).map(([provider, count]) => (
                  <span key={provider} className="rounded-full border border-line bg-paper-soft px-2 py-0.5 text-[11px] capitalize text-ink-soft">
                    {provider}: {count}
                  </span>
                ))}
              </div>
            )}
            <p className="mt-3 text-[12px] text-ink-faint">{summary.total_tool_executions} tool execution(s) recorded overall.</p>
          </div>
        </>
      )}
    </div>
  );
}

function MetricCard({ label, value, hint }: { label: string; value: string; hint: string }) {
  return (
    <div className="card p-4">
      <div className="mb-1 font-mono text-[10px] uppercase tracking-wide text-ink-faint">{label}</div>
      <div className="font-serif text-[24px] font-semibold text-ink">{value}</div>
      <p className="mt-1 text-[11px] text-ink-faint">{hint}</p>
    </div>
  );
}
