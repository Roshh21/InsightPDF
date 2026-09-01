"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { Globe2, Search } from "lucide-react";
import { api, ApiError } from "@/lib/api";
import type { Document, ToolExecuteResponse } from "@/lib/types";
import { EmptyState, PageHeader, Spinner } from "@/components/ui";
import ToolResultView from "@/components/ToolResultView";

export default function ResearchPage() {
  const params = useParams();
  const workspaceId = params.workspaceId as string;

  const [documents, setDocuments] = useState<Document[] | null>(null);
  const [webConfigured, setWebConfigured] = useState<boolean | null>(null);
  const [selectedId, setSelectedId] = useState<string>("");
  const [query, setQuery] = useState("What has changed in this field since this paper was published?");
  const [result, setResult] = useState<ToolExecuteResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .listDocuments(workspaceId)
      .then(setDocuments)
      .catch((e) => setError(e instanceof ApiError ? e.message : "Failed to load documents."));
    api.getPublicConfig().then((c) => setWebConfigured(c.web_research_configured)).catch(() => setWebConfigured(null));
  }, [workspaceId]);

  const eligible = (documents || []).filter((d) => d.status === "READY" && d.capabilities?.includes("web_research"));

  async function run() {
    if (!selectedId || !query.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await api.executeTool(workspaceId, "web_research", [selectedId], { query: query.trim() });
      setResult(res);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Web research failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mx-auto max-w-4xl px-8 py-10">
      <PageHeader
        eyebrow="Research paper mode"
        title="Web Research"
        description="Compare an uploaded paper against current external context. Citations into the paper and web sources are kept separate."
      />

      {webConfigured === false && (
        <div className="mb-5 rounded-[5px] border border-amber/30 bg-amber-soft px-3 py-2 text-[12.5px] text-amber-dark">
          Web research isn&apos;t configured on this server yet — set <code className="font-mono">TAVILY_API_KEY</code> in the backend
          .env to enable it.
        </div>
      )}

      {!documents ? (
        <Spinner size={16} />
      ) : eligible.length === 0 ? (
        <EmptyState
          icon={Globe2}
          title="No eligible research papers"
          description="Upload and process a research paper to use web research."
        />
      ) : (
        <>
          <div className="card mb-5 flex flex-col gap-3 p-4">
            <div>
              <label className="mb-1 block text-[12px] font-medium text-ink-soft">Paper</label>
              <select className="field" value={selectedId} onChange={(e) => setSelectedId(e.target.value)}>
                <option value="">Select a paper…</option>
                {eligible.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.profile?.title || d.original_filename}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="mb-1 block text-[12px] font-medium text-ink-soft">Research question</label>
              <textarea className="field" rows={2} value={query} onChange={(e) => setQuery(e.target.value)} />
            </div>
            <button className="btn-primary self-start" disabled={!selectedId || loading} onClick={run}>
              {loading ? <Spinner size={14} /> : <Search size={14} />}
              Research
            </button>
            {error && <div className="text-[12.5px] text-danger">{error}</div>}
          </div>

          {result && (
            <div className="card p-5 animate-fade-in">
              <div className="mb-3 font-serif text-[16px] font-semibold text-ink">{result.title}</div>
              <ToolResultView result={result} />
            </div>
          )}
        </>
      )}
    </div>
  );
}
