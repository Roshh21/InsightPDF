"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import {
  ArrowLeft,
  MessageSquare,
  RotateCw,
  Sparkles,
  Tag,
  Trash2,
} from "lucide-react";
import { api, ApiError } from "@/lib/api";
import type {
  Document,
  ToolCatalogEntry,
  ToolExecuteResponse,
} from "@/lib/types";
import { ConfidenceBar, Spinner, StatusBadge } from "@/components/ui";
import { docTypeLabel } from "@/lib/format";
import ToolPanel from "@/components/ToolPanel";
import ToolResultView from "@/components/ToolResultView";
import ChatThread from "@/components/ChatThread";

interface RunRecord {
  key: string;
  tool: ToolCatalogEntry;
  result: ToolExecuteResponse;
}

export default function DocumentPage() {
  const params = useParams<{
    workspaceId?: string;
    documentId?: string;
  }>();

  const router = useRouter();

  const workspaceId = params?.workspaceId;
  const documentId = params?.documentId;

  const [document, setDocument] = useState<Document | null>(null);
  const [catalog, setCatalog] = useState<ToolCatalogEntry[] | null>(null);
  const [runs, setRuns] = useState<RunRecord[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [retrying, setRetrying] = useState(false);

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const clearPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const load = useCallback(async () => {
    if (!documentId) {
      return;
    }

    try {
      setError(null);

      const doc = await api.getDocument(documentId);
      setDocument(doc);
    } catch (e) {
      setError(
        e instanceof ApiError
          ? e.message
          : "Failed to load document."
      );
    }
  }, [documentId]);

  useEffect(() => {
    if (!documentId) {
      setDocument(null);
      return;
    }

    void load();
  }, [documentId, load]);

  useEffect(() => {
    if (!documentId) {
      setCatalog(null);
      return;
    }

    let cancelled = false;

    api
      .getToolCatalog()
      .then((tools) => {
        if (!cancelled) {
          setCatalog(tools);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setCatalog([]);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [documentId]);

  useEffect(() => {
    const active =
      document?.status === "UPLOADED" ||
      document?.status === "PROCESSING";

    clearPolling();

    if (!active) {
      return;
    }

    pollRef.current = setInterval(() => {
      void load();
    }, 2500);

    return clearPolling;
  }, [document?.status, load, clearPolling]);

  useEffect(() => {
    return () => {
      clearPolling();
    };
  }, [clearPolling]);

  function handleResult(
    tool: ToolCatalogEntry,
    result: ToolExecuteResponse
  ) {
    setRuns((current) => [
      {
        key: `${tool.id}-${Date.now()}`,
        tool,
        result,
      },
      ...current,
    ]);
  }

  async function handleRetry() {
    if (!documentId) {
      return;
    }

    setRetrying(true);
    setError(null);

    try {
      await api.retryDocument(documentId);
      await load();
    } catch (e) {
      setError(
        e instanceof ApiError
          ? e.message
          : "Retry failed."
      );
    } finally {
      setRetrying(false);
    }
  }

  async function handleDelete() {
    if (!documentId || !workspaceId) {
      return;
    }

    if (!confirm("Delete this document? This cannot be undone.")) {
      return;
    }

    try {
      await api.deleteDocument(documentId);
      router.push(`/workspaces/${workspaceId}`);
    } catch (e) {
      setError(
        e instanceof ApiError
          ? e.message
          : "Failed to delete document."
      );
    }
  }

  /*
   * Dynamic route parameters may temporarily be unavailable.
   * Render a safe loading state instead of passing undefined
   * into API calls or child components.
   */
  if (!workspaceId || !documentId) {
    return (
      <div className="mx-auto max-w-6xl px-8 py-10">
        <div className="flex items-center gap-2 py-10 text-ink-soft">
          <Spinner size={16} />
          Loading document…
        </div>
      </div>
    );
  }

  if (!document) {
    return (
      <div className="mx-auto max-w-6xl px-8 py-10">
        {error ? (
          <div className="rounded-[5px] border border-danger/30 bg-danger-soft px-3 py-2 text-[13px] text-danger">
            {error}
          </div>
        ) : (
          <div className="flex items-center gap-2 py-10 text-ink-soft">
            <Spinner size={16} />
            Loading document…
          </div>
        )}
      </div>
    );
  }

  const availableTools = (catalog ?? []).filter((tool) =>
    document.capabilities?.includes(tool.id)
  );

  const profile = document.profile;
  const notReady = document.status !== "READY";

  return (
    <div className="flex">
      <div className="min-w-0 flex-1 px-8 py-10">
        <Link
          href={`/workspaces/${workspaceId}`}
          className="mb-4 inline-flex items-center gap-1.5 text-[12.5px] text-ink-soft hover:text-ink"
        >
          <ArrowLeft size={13} />
          Back to workspace
        </Link>

        <div className="mb-6 flex items-start justify-between gap-4">
          <div>
            <div className="mb-1 flex items-center gap-2">
              <span className="font-mono text-[11px] uppercase tracking-wider text-amber-dark">
                {docTypeLabel(document.document_type)}
              </span>

              <StatusBadge status={document.status} />
            </div>

            <h1 className="font-serif text-[24px] font-semibold leading-tight text-ink">
              {profile?.title || document.original_filename}
            </h1>

            {profile?.authors && profile.authors.length > 0 && (
              <p className="mt-1 text-[13px] text-ink-soft">
                {profile.authors.join(", ")}
              </p>
            )}
          </div>
        </div>

        {notReady ? (
          <div className="card flex flex-col items-center gap-2 px-6 py-14 text-center">
            {document.status === "FAILED" ? (
              <>
                <div className="font-serif text-[16px] font-semibold text-danger">
                  Processing failed
                </div>

                <p className="max-w-md text-[13px] text-ink-soft">
                  {document.error_message}
                </p>

                <div className="mt-2 flex items-center gap-2">
                  <button
                    type="button"
                    className="btn-primary"
                    disabled={retrying}
                    onClick={handleRetry}
                  >
                    {retrying ? (
                      <Spinner size={14} />
                    ) : (
                      <RotateCw size={14} />
                    )}
                    Retry
                  </button>

                  <button
                    type="button"
                    className="btn-secondary text-danger hover:bg-danger-soft"
                    onClick={handleDelete}
                  >
                    <Trash2 size={14} />
                    Delete
                  </button>
                </div>
              </>
            ) : (
              <>
                <Spinner size={20} />

                <div className="font-serif text-[16px] font-semibold text-ink">
                  Processing document…
                </div>

                <p className="max-w-md text-[13px] text-ink-soft">
                  Parsing, classifying, and indexing this document. Tools and
                  chat will unlock automatically once it&apos;s ready.
                </p>
              </>
            )}
          </div>
        ) : (
          <>
            <div className="card mb-6 grid grid-cols-2 gap-4 p-4 sm:grid-cols-4">
              <Stat
                label="Confidence"
                value={
                  <ConfidenceBar
                    value={document.classification_confidence}
                  />
                }
              />

              <Stat
                label="Pages"
                value={
                  <span className="font-mono text-[13px] text-ink">
                    {document.page_count ?? "—"}
                  </span>
                }
              />

              <Stat
                label="Sections"
                value={
                  <span className="font-mono text-[13px] text-ink">
                    {profile?.sections.length ?? 0}
                  </span>
                }
              />

              <Stat
                label="Tables"
                value={
                  <span className="font-mono text-[13px] text-ink">
                    {profile?.tables.length ?? 0}
                  </span>
                }
              />
            </div>

            {profile?.summary_hint && (
              <p className="mb-4 text-[13.5px] leading-relaxed text-ink-soft">
                {profile.summary_hint}
              </p>
            )}

            {profile?.topics && profile.topics.length > 0 && (
              <div className="mb-8 flex flex-wrap items-center gap-1.5">
                <Tag size={12} className="text-ink-faint" />

                {profile.topics.map((topic) => (
                  <span
                    key={topic}
                    className="rounded-full border border-line bg-paper-soft px-2 py-0.5 text-[11.5px] text-ink-soft"
                  >
                    {topic}
                  </span>
                ))}
              </div>
            )}

            <div className="mb-8">
              <div className="mb-3 flex items-center gap-2">
                <Sparkles
                  size={14}
                  className="text-amber-dark"
                />

                <div className="font-serif text-[15px] font-semibold text-ink">
                  Results
                </div>
              </div>

              {runs.length === 0 ? (
                <div className="card flex flex-col items-center gap-1 px-6 py-10 text-center text-ink-soft">
                  <p className="text-[13px]">
                    Run a tool from the right to see results here.
                  </p>
                </div>
              ) : (
                <div className="flex flex-col gap-4">
                  {runs.map((run) => (
                    <div
                      key={run.key}
                      className="card p-4 animate-fade-in"
                    >
                      <div className="mb-3 font-serif text-[15px] font-semibold text-ink">
                        {run.result.title || run.tool.name}
                      </div>

                      <ToolResultView result={run.result} />
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div>
              <div className="mb-3 flex items-center gap-2">
                <MessageSquare
                  size={14}
                  className="text-cobalt"
                />

                <div className="font-serif text-[15px] font-semibold text-ink">
                  Ask a follow-up
                </div>
              </div>

              <div className="card p-4">
                <ChatThread
                  workspaceId={workspaceId}
                  documentIds={[documentId]}
                  compact
                  placeholder="e.g. Explain the second section in simpler terms"
                  suggestions={[
                    "Summarize this",
                    "What are the key takeaways?",
                  ]}
                />
              </div>
            </div>
          </>
        )}
      </div>

      {!notReady && (
        <aside className="hidden w-[300px] shrink-0 border-l border-line px-5 py-10 lg:block">
          <div className="mb-3 font-serif text-[14px] font-semibold text-ink">
            Tools
          </div>

          {catalog === null ? (
            <Spinner size={14} />
          ) : (
            <ToolPanel
              tools={availableTools}
              workspaceId={workspaceId}
              documentIds={[documentId]}
              onResult={handleResult}
            />
          )}
        </aside>
      )}
    </div>
  );
}

function Stat({
  label,
  value,
}: {
  label: string;
  value: React.ReactNode;
}) {
  return (
    <div>
      <div className="mb-1 font-mono text-[10px] uppercase tracking-wide text-ink-faint">
        {label}
      </div>
      {value}
    </div>
  );
}
