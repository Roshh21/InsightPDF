"use client";

import { useEffect, useMemo, useState } from "react";
import { useParams } from "next/navigation";
import { CheckSquare, Square, SquareStack } from "lucide-react";
import { api, ApiError } from "@/lib/api";
import type {
  Document,
  ToolCatalogEntry,
  ToolExecuteResponse,
} from "@/lib/types";
import { EmptyState, PageHeader, Spinner } from "@/components/ui";
import ToolPanel from "@/components/ToolPanel";
import ToolResultView from "@/components/ToolResultView";

export default function ComparePage() {
  const params = useParams<{ workspaceId?: string }>();
  const workspaceId = params?.workspaceId;

  const [documents, setDocuments] = useState<Document[] | null>(null);
  const [catalog, setCatalog] = useState<ToolCatalogEntry[] | null>(null);
  const [selected, setSelected] = useState<string[]>([]);
  const [result, setResult] = useState<{
    tool: ToolCatalogEntry;
    result: ToolExecuteResponse;
  } | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Wait until Next.js provides the dynamic route parameter.
    if (!workspaceId) {
      setDocuments(null);
      setCatalog(null);
      return;
    }

    let cancelled = false;

    setDocuments(null);
    setCatalog(null);
    setError(null);

    Promise.all([
      api.listDocuments(workspaceId),
      api.getToolCatalog(),
    ])
      .then(([docs, cat]) => {
        if (cancelled) return;

        setDocuments(docs);
        setCatalog(cat);
      })
      .catch((e) => {
        if (cancelled) return;

        setError(
          e instanceof ApiError
            ? e.message
            : "Failed to load workspace."
        );

        setDocuments([]);
        setCatalog([]);
      });

    return () => {
      cancelled = true;
    };
  }, [workspaceId]);

  const readyDocs = useMemo(
    () => (documents ?? []).filter((document) => document.status === "READY"),
    [documents]
  );

  const selectedDocs = useMemo(
    () => readyDocs.filter((document) => selected.includes(document.id)),
    [readyDocs, selected]
  );

  const compareTools = useMemo(() => {
    if (!catalog || selectedDocs.length < 2) {
      return [];
    }

    return catalog.filter(
      (tool) =>
        tool.requires_multi_document &&
        selectedDocs.every((document) =>
          document.capabilities?.includes(tool.id)
        )
    );
  }, [catalog, selectedDocs]);

  function toggle(id: string) {
    setResult(null);

    setSelected((current) =>
      current.includes(id)
        ? current.filter((item) => item !== id)
        : [...current, id]
    );
  }

  // Hooks must run before conditional returns.
  // This handles the short period where the dynamic route parameter
  // has not yet been resolved without passing undefined to child components.
  if (!workspaceId) {
    return (
      <div className="mx-auto max-w-6xl px-4 py-8 sm:px-6 md:px-8 md:py-10">
        <div className="card p-6">
          <p className="text-[12.5px] text-ink-faint">
            Loading workspace...
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="mx-auto flex max-w-6xl flex-col gap-6 px-4 py-8 sm:px-6 md:flex-row md:gap-8 md:px-8 md:py-10">
      <aside className="w-full shrink-0 md:w-[240px] lg:w-[280px]">
        <PageHeader
          title="Compare"
          description="Select two or more documents to run a structured, cited comparison."
        />

        {error && (
          <div className="mb-3 rounded-[5px] border border-danger/30 bg-danger-soft px-3 py-2 text-[12.5px] text-danger">
            {error}
          </div>
        )}

        {!documents ? (
          <Spinner size={14} />
        ) : readyDocs.length < 2 ? (
          <p className="text-[12.5px] text-ink-faint">
            Upload at least two processed documents to compare them.
          </p>
        ) : (
          <div className="flex flex-col gap-1">
            {readyDocs.map((document) => {
              const checked = selected.includes(document.id);

              return (
                <button
                  key={document.id}
                  type="button"
                  onClick={() => toggle(document.id)}
                  className="flex items-start gap-2 rounded-[5px] px-2 py-1.5 text-left text-[12.5px] text-ink hover:bg-paper-dim"
                >
                  {checked ? (
                    <CheckSquare
                      size={14}
                      className="mt-0.5 shrink-0 text-cobalt"
                    />
                  ) : (
                    <Square
                      size={14}
                      className="mt-0.5 shrink-0 text-ink-faint"
                    />
                  )}

                  <span className="truncate">
                    {document.profile?.title ||
                      document.original_filename}
                  </span>
                </button>
              );
            })}
          </div>
        )}

        {selectedDocs.length >= 2 && (
          <div className="mt-6">
            <div className="mb-2 font-mono text-[10.5px] uppercase tracking-wide text-ink-faint">
              Comparison tools
            </div>

            {!catalog ? (
              <Spinner size={14} />
            ) : compareTools.length === 0 ? (
              <p className="text-[12px] text-ink-faint">
                No comparison tool is common to the selected documents&apos;
                types.
              </p>
            ) : (
              <ToolPanel
                tools={compareTools}
                workspaceId={workspaceId}
                documentIds={selectedDocs.map(
                  (document) => document.id
                )}
                onResult={(tool, res) =>
                  setResult({
                    tool,
                    result: res,
                  })
                }
              />
            )}
          </div>
        )}
      </aside>

      <div className="min-w-0 flex-1">
        {selectedDocs.length < 2 ? (
          <EmptyState
            icon={SquareStack}
            title="Pick documents to compare"
            description="Select two or more from the left to see available comparison tools."
          />
        ) : !result ? (
          <div className="card flex flex-col items-center gap-1 px-6 py-14 text-center text-ink-soft">
            <p className="text-[13px]">
              Run a comparison tool from the left to see results here.
            </p>
          </div>
        ) : (
          <div className="card p-5 animate-fade-in">
            <div className="mb-3 font-serif text-[16px] font-semibold text-ink">
              {result.result.title || result.tool.name}
            </div>

            <ToolResultView result={result.result} />
          </div>
        )}
      </div>
    </div>
  );
}
