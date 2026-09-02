"use client";

import { useCallback, useEffect, useState } from "react";
import { Cloud, RefreshCw, Timer } from "lucide-react";
import { api, ApiError } from "@/lib/api";
import type { ModelStatus, ProviderStatus } from "@/lib/types";
import { PageHeader, Spinner } from "@/components/ui";

const SLOT_LABEL: Record<string, string> = {
  primary: "Primary",
  secondary: "Secondary",
  tertiary: "Tertiary",
};

export default function ModelsPage() {
  const [status, setStatus] = useState<ModelStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const load = useCallback((refresh = false) => {
    if (refresh) setRefreshing(true);
    api
      .getModelStatus(refresh)
      .then(setStatus)
      .catch((e) => setError(e instanceof ApiError ? e.message : "Failed to load model status."))
      .finally(() => setRefreshing(false));
  }, []);

  useEffect(() => {
    load();
    const interval = setInterval(() => load(), 20000);
    return () => clearInterval(interval);
  }, [load]);

  return (
    <div className="mx-auto max-w-4xl px-4 py-8 sm:px-6 md:px-8 md:py-10">
      <PageHeader
        eyebrow="Free-Provider LLM Gateway"
        title="Model Status"
        description="What the agent actually verified just now, for each configured free provider slot — not an assumption from configuration alone."
        actions={
          <button className="btn-secondary" onClick={() => load(true)} disabled={refreshing}>
            <RefreshCw size={14} className={refreshing ? "animate-spin" : ""} /> Recheck
          </button>
        }
      />

      {error && <div className="mb-4 rounded-[5px] border border-danger/30 bg-danger-soft px-3 py-2 text-[13px] text-danger">{error}</div>}

      {!status ? (
        <Spinner size={16} />
      ) : (
        <>
          <div className="mb-6 grid grid-cols-1 gap-4 sm:grid-cols-3">
            {status.providers.map((p) => (
              <ProviderCard key={p.slot} provider={p} isActive={status.active_provider === p.provider} />
            ))}
          </div>

          <div className="card p-4">
            <div className="mb-3 font-serif text-[14.5px] font-semibold text-ink">Active configuration</div>
            <dl className="grid grid-cols-2 gap-y-2 text-[13px] sm:grid-cols-3">
              <Row label="Active provider" value={status.active_provider || "None available"} />
              <Row label="Vector store" value={status.vector_store} />
              <Row label="Embedding model" value={status.embedding_model} mono />
            </dl>
            {!status.active_provider && (
              <p className="mt-3 rounded-[5px] bg-danger-soft px-3 py-2 text-[12.5px] text-danger">
                No free LLM provider is currently reachable. Add a free API key (Groq, Gemini, or OpenRouter) in the
                backend .env, or wait out any active rate-limit cooldowns above.
              </p>
            )}
          </div>

          <p className="mt-4 text-[12px] text-ink-faint">
            InsightPDF uses only free hosted providers by default, with automatic failover between them — no paid
            API is required. See Settings for which provider slots are currently configured.
          </p>
        </>
      )}
    </div>
  );
}

function Row({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div>
      <dt className="mb-0.5 font-mono text-[10px] uppercase tracking-wide text-ink-faint">{label}</dt>
      <dd className={mono ? "font-mono text-[12px] text-ink" : "text-ink"}>{value}</dd>
    </div>
  );
}

function ProviderCard({ provider, isActive }: { provider: ProviderStatus; isActive: boolean }) {
  return (
    <div className={`card p-4 ${isActive ? "border-cobalt/40 shadow-pop" : ""}`}>
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Cloud size={16} className="text-ink-faint" />
          <div>
            <div className="font-serif text-[14.5px] font-semibold capitalize text-ink">{provider.provider}</div>
            <div className="font-mono text-[10.5px] uppercase tracking-wide text-ink-faint">{SLOT_LABEL[provider.slot]}</div>
          </div>
        </div>
        {isActive && <span className="badge border-cobalt/30 bg-cobalt-soft text-cobalt-dark">Active</span>}
      </div>
      <div className="flex flex-col gap-1.5 text-[12.5px]">
        <Dot ok={provider.configured} label="Configured" />
        <Dot ok={provider.available} label={provider.cooling_down ? "Rate-limited" : "Available"} warn={provider.cooling_down} />
      </div>
      {provider.cooling_down && (
        <div className="mt-2 flex items-center gap-1.5 rounded-[4px] bg-amber-soft px-2 py-1 text-[11px] text-amber-dark">
          <Timer size={11} />
          Cooling down ~{Math.ceil(provider.cooldown_remaining_s)}s before retrying
        </div>
      )}
      <div className="mt-3 flex flex-col gap-1 border-t border-line pt-3 font-mono text-[11px] text-ink-soft">
        <div>fast: {provider.model_fast}</div>
        <div>strong: {provider.model_strong}</div>
      </div>
    </div>
  );
}

function Dot({ ok, label, warn }: { ok: boolean; label: string; warn?: boolean }) {
  return (
    <span className="flex items-center gap-1.5 text-ink-soft">
      <span className={`h-1.5 w-1.5 rounded-full ${ok ? "bg-success" : warn ? "bg-amber" : "bg-danger"}`} />
      {label}
    </span>
  );
}
