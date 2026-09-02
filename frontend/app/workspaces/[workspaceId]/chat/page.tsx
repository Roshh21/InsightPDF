"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { CheckSquare, Square } from "lucide-react";
import { api, ApiError } from "@/lib/api";
import type { Document } from "@/lib/types";
import { PageHeader, Spinner } from "@/components/ui";
import ChatThread from "@/components/ChatThread";

export default function WorkspaceChatPage() {
  const params = useParams<{ workspaceId?: string }>();
  const workspaceId = params?.workspaceId;

  const [documents, setDocuments] = useState<Document[] | null>(null);
  const [selected, setSelected] = useState<string[]>([]);
  const [spoilerLevel, setSpoilerLevel] = useState<string>("chapter");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // The route parameter may not be available immediately.
    // Do not make an API request until it exists.
    if (!workspaceId) {
      setDocuments(null);
      return;
    }

    let cancelled = false;

    setDocuments(null);
    setError(null);

    api
      .listDocuments(workspaceId)
      .then((docs) => {
        if (!cancelled) {
          setDocuments(docs);
        }
      })
      .catch((e) => {
        if (!cancelled) {
          setError(
            e instanceof ApiError
              ? e.message
              : "Failed to load documents."
          );
          setDocuments([]);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [workspaceId]);

  const readyDocs = (documents ?? []).filter(
    (document) => document.status === "READY"
  );

  const hasLiterature = readyDocs.some(
    (document) => document.document_type === "literature"
  );

  function toggle(id: string) {
    setSelected((current) =>
      current.includes(id)
        ? current.filter((item) => item !== id)
        : [...current, id]
    );
  }

  // Invalid/missing route parameter.
  // Hooks have already been called, so this conditional return is safe.
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
      <aside className="w-full shrink-0 md:w-[220px] lg:w-[260px]">
        <PageHeader
          title="Chat"
          description="Ask about one document, several, or leave none selected to search the whole workspace."
        />

        {error && (
          <div className="mb-3 rounded-[5px] border border-danger/30 bg-danger-soft px-3 py-2 text-[12.5px] text-danger">
            {error}
          </div>
        )}

        {!documents ? (
          <Spinner size={14} />
        ) : readyDocs.length === 0 ? (
          <p className="text-[12.5px] text-ink-faint">
            No processed documents yet.
          </p>
        ) : (
          <div className="flex flex-col gap-1">
            <div className="mb-1 font-mono text-[10.5px] uppercase tracking-wide text-ink-faint">
              Scope {selected.length === 0 && "(whole workspace)"}
            </div>

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

        {hasLiterature && (
          <div className="mt-5">
            <div className="mb-1.5 font-mono text-[10.5px] uppercase tracking-wide text-ink-faint">
              Spoiler level
            </div>

            <select
              className="field text-[12.5px]"
              value={spoilerLevel}
              onChange={(event) =>
                setSpoilerLevel(event.target.value)
              }
            >
              <option value="none">No spoilers</option>
              <option value="chapter">Up to current chapter</option>
              <option value="full">Full spoilers</option>
            </select>
          </div>
        )}
      </aside>

      <div className="min-w-0 flex-1">
        <div className="card p-4">
          {!documents ? (
            <Spinner size={14} />
          ) : (
            <ChatThread
              workspaceId={workspaceId}
              documentIds={selected}
              spoilerLevel={
                hasLiterature ? spoilerLevel : undefined
              }
              suggestions={[
                "Summarize the whole workspace",
                "What do these documents have in common?",
              ]}
            />
          )}
        </div>
      </div>
    </div>
  );
}

