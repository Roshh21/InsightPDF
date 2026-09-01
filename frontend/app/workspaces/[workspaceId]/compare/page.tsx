"use client";

import { useEffect, useMemo, useState } from "react";
import { useParams } from "next/navigation";
import { CheckSquare, Square, SquareStack } from "lucide-react";
import { api, ApiError } from "@/lib/api";
import type { Document, ToolCatalogEntry, ToolExecuteResponse } from "@/lib/types";
import { EmptyState, PageHeader, Spinner } from "@/components/ui";
import ToolPanel from "@/components/ToolPanel";
import ToolResultView from "@/components/ToolResultView";

export default function ComparePage() {
  const params = useParams();
  const workspaceId = params.workspaceId as string;

  const [documents, setDocuments] = useState<Document[] | null>(null);
  const [catalog, setCatalog] = useState<ToolCatalogEntry[] | null>(null);
  const [selected, setSelected] = useState<string[]>([]);
  const [result, setResult] = useState<{ tool: ToolCatalogEntry; result: ToolExecuteResponse } | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([api.listDocuments(workspaceId), api.getToolCatalog()])
      .then(([docs, cat]) => {
        setDocuments(docs);
        setCatalog(cat);
      })
      .catch((e) => setError(e instanceof ApiError ? e.message : "Failed to load workspace."));
  }, [workspaceId]);

  const readyDocs = (documents || []).filter((d) => d.status === "READY");
  const selectedDocs = readyDocs.filter((d) => selected.includes(d.id));

  const compareTools = useMemo(() => {
    if (!catalog || selectedDocs.length < 2) return [];
    return catalog.filter(
      (t) => t.requires_multi_document && selectedDocs.every((d) => d.capabilities?.includes(t.id))
    );
  }, [catalog, selectedDocs]);

  function toggle(id: string) {
    setResult(null);
    setSelected((s) => (s.includes(id) ? s.filter((x) => x !== id) : [...s, id]));
  }

  return (
    <div className="mx-auto flex max-w-6xl gap-8 px-8 py-10">
      <aside className="w-[280px] shrink-0">
        <PageHeader title="Compare" description="Select two or more documents to run a structured, cited comparison." />
        {error && <div className="mb-3 rounded-[5px] border border-danger/30 bg-danger-soft px-3 py-2 text-[12.5px] text-danger">{error}</div>}
        {!documents ? (
          <Spinner size={14} />
        ) : readyDocs.length < 2 ? (
          <p className="text-[12.5px] text-ink-faint">Upload at least two processed documents to compare them.</p>
        ) : (
          <div className="flex flex-col gap-1">
            {readyDocs.map((d) => {
              const checked = selected.includes(d.id);
              return (
                <button
                  key={d.id}
                  onClick={() => toggle(d.id)}
                  className="flex items-start gap-2 rounded-[5px] px-2 py-1.5 text-left text-[12.5px] text-ink hover:bg-paper-dim"
                >
                  {checked ? <CheckSquare size={14} className="mt-0.5 shrink-0 text-cobalt" /> : <Square size={14} className="mt-0.5 shrink-0 text-ink-faint" />}
                  <span className="truncate">{d.profile?.title || d.original_filename}</span>
                </button>
              );
            })}
          </div>
        )}

        {selectedDocs.length >= 2 && (
          <div className="mt-6">
            <div className="mb-2 font-mono text-[10.5px] uppercase tracking-wide text-ink-faint">Comparison tools</div>
            {compareTools.length === 0 ? (
              <p className="text-[12px] text-ink-faint">No comparison tool is common to the selected documents&apos; types.</p>
            ) : (
              <ToolPanel
                tools={compareTools}
                workspaceId={workspaceId}
                documentIds={selectedDocs.map((d) => d.id)}
                onResult={(tool, res) => setResult({ tool, result: res })}
              />
            )}
          </div>
        )}
      </aside>

      <div className="min-w-0 flex-1">
        {selectedDocs.length < 2 ? (
          <EmptyState icon={SquareStack} title="Pick documents to compare" description="Select two or more from the left to see available comparison tools." />
        ) : !result ? (
          <div className="card flex flex-col items-center gap-1 px-6 py-14 text-center text-ink-soft">
            <p className="text-[13px]">Run a comparison tool from the left to see results here.</p>
          </div>
        ) : (
          <div className="card p-5 animate-fade-in">
            <div className="mb-3 font-serif text-[16px] font-semibold text-ink">{result.result.title || result.tool.name}</div>
            <ToolResultView result={result.result} />
          </div>
        )}
      </div>
    </div>
  );
}
