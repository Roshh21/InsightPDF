"use client";

import { useEffect, useState } from "react";
import { api, ApiError } from "@/lib/api";
import type { PublicConfig } from "@/lib/types";
import { PageHeader, Spinner } from "@/components/ui";

export default function SettingsPage() {
  const [config, setConfig] = useState<PublicConfig | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .getPublicConfig()
      .then(setConfig)
      .catch((e) => setError(e instanceof ApiError ? e.message : "Failed to load configuration."));
  }, []);

  return (
    <div className="mx-auto max-w-3xl px-8 py-10">
      <PageHeader
        eyebrow="Configuration"
        title="Settings"
        description="Read-only. Secrets and API keys are never sent to the browser — configure them in the backend's .env file."
      />

      {error && <div className="mb-4 rounded-[5px] border border-danger/30 bg-danger-soft px-3 py-2 text-[13px] text-danger">{error}</div>}

      {!config ? (
        <Spinner size={16} />
      ) : (
        <div className="flex flex-col gap-4">
          <Section title="Environment">
            <Row label="App" value={config.app_name} />
            <Row label="Environment" value={config.environment} />
          </Section>
          <Section title="Model gateway (free-provider-first)">
            {config.llm_slots.map((s) => (
              <Row
                key={s.slot}
                label={`${s.slot.charAt(0).toUpperCase()}${s.slot.slice(1)} — ${s.provider}`}
                value={s.configured ? "Configured" : "Not configured"}
              />
            ))}
            <Row label="Web research" value={config.web_research_configured ? "Configured" : "Not configured (optional)"} />
            {!config.any_llm_configured && (
              <p className="mt-1 rounded-[4px] bg-danger-soft px-2 py-1.5 text-[12px] text-danger">
                No free LLM provider is configured yet — add at least one of GROQ_API_KEY, GEMINI_API_KEY, or
                OPENROUTER_API_KEY to the backend .env. See the README for how to get a free key from each.
              </p>
            )}
          </Section>
          <Section title="Retrieval">
            <Row label="Vector store" value={config.vector_store} />
            <Row label="Embedding provider" value={config.embedding_provider} />
            <Row label="Embedding model" value={config.embedding_model} mono />
          </Section>
          <Section title="Uploads">
            <Row label="Max file size" value={`${config.max_upload_mb} MB`} />
          </Section>
        </div>
      )}
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="card p-4">
      <div className="mb-3 font-serif text-[14.5px] font-semibold text-ink">{title}</div>
      <dl className="flex flex-col gap-2.5">{children}</dl>
    </div>
  );
}

function Row({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="flex items-center justify-between text-[13px]">
      <dt className="text-ink-soft">{label}</dt>
      <dd className={mono ? "font-mono text-[12px] text-ink" : "font-medium text-ink"}>{value}</dd>
    </div>
  );
}
