"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { CheckSquare, Square } from "lucide-react";
import { api, ApiError } from "@/lib/api";
import type { Document } from "@/lib/types";
import { PageHeader, Spinner, StatusBadge } from "@/components/ui";
import ChatThread from "@/components/ChatThread";

export default function WorkspaceChatPage() {
  const params = useParams();
  const workspaceId = params.workspaceId as string;

  const [documents, setDocuments] = useState<Document[] | null>(null);
  const [selected, setSelected] = useState<string[]>([]);
  const [spoilerLevel, setSpoilerLevel] = useState<string>("chapter");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .listDocuments(workspaceId)
      .then(setDocuments)
      .catch((e) => setError(e instanceof ApiError ? e.message : "Failed to load documents."));
  }, [workspaceId]);

  const readyDocs = (documents || []).filter((d) => d.status === "READY");
  const hasLiterature = readyDocs.some((d) => d.document_type === "literature");

  function toggle(id: string) {
    setSelected((s) => (s.includes(id) ? s.filter((x) => x !== id) : [...s, id]));
  }

  return (
    <div className="mx-auto flex max-w-6xl gap-8 px-8 py-10">
      <aside className="w-[260px] shrink-0">
        <PageHeader title="Chat" description="Ask about one document, several, or leave none selected to search the whole workspace." />
        {error && <div className="mb-3 rounded-[5px] border border-danger/30 bg-danger-soft px-3 py-2 text-[12.5px] text-danger">{error}</div>}
        {!documents ? (
          <Spinner size={14} />
        ) : readyDocs.length === 0 ? (
          <p className="text-[12.5px] text-ink-faint">No processed documents yet.</p>
        ) : (
          <div className="flex flex-col gap-1">
            <div className="mb-1 font-mono text-[10.5px] uppercase tracking-wide text-ink-faint">
              Scope {selected.length === 0 && "(whole workspace)"}
            </div>
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

        {hasLiterature && (
          <div className="mt-5">
            <div className="mb-1.5 font-mono text-[10.5px] uppercase tracking-wide text-ink-faint">Spoiler level</div>
            <select className="field text-[12.5px]" value={spoilerLevel} onChange={(e) => setSpoilerLevel(e.target.value)}>
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
              spoilerLevel={hasLiterature ? spoilerLevel : undefined}
              suggestions={["Summarize the whole workspace", "What do these documents have in common?"]}
            />
          )}
        </div>
      </div>
    </div>
  );
}
