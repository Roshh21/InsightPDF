"use client";

import { useState } from "react";
import Link from "next/link";
import { FileText, RotateCw, Trash2 } from "lucide-react";
import type { Document } from "@/lib/types";
import { StatusBadge, Spinner } from "./ui";
import { docTypeLabel, formatDate } from "@/lib/format";

export default function DocumentCard({
  document,
  workspaceId,
  onDelete,
  onRetry,
}: {
  document: Document;
  workspaceId: string;
  onDelete?: (id: string) => void;
  onRetry?: (id: string) => Promise<void> | void;
}) {
  const [retrying, setRetrying] = useState(false);
  const failed = document.status === "FAILED";

  async function handleRetry(e: React.MouseEvent) {
    e.preventDefault();
    if (!onRetry || retrying) return;
    setRetrying(true);
    try {
      await onRetry(document.id);
    } finally {
      setRetrying(false);
    }
  }

  return (
    <div className={`card group relative flex flex-col gap-2.5 p-4 transition-shadow hover:shadow-pop ${failed ? "border-danger/30" : ""}`}>
      <div className="flex items-start justify-between gap-2">
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-[5px] bg-cobalt-soft">
          <FileText size={15} className="text-cobalt" />
        </div>
        <StatusBadge status={document.status} />
      </div>
      <Link href={`/workspaces/${workspaceId}/documents/${document.id}`} className="min-w-0">
        <div className="truncate font-serif text-[14.5px] font-semibold text-ink group-hover:text-cobalt" title={document.original_filename}>
          {document.profile?.title || document.original_filename}
        </div>
      </Link>
      <div className="flex items-center gap-2 font-mono text-[11px] text-ink-faint">
        <span>{docTypeLabel(document.document_type)}</span>
        {document.page_count && (
          <>
            <span>·</span>
            <span>{document.page_count}p</span>
          </>
        )}
        <span>·</span>
        <span>{formatDate(document.created_at)}</span>
        {document.retry_count > 0 && (
          <>
            <span>·</span>
            <span>
              retried {document.retry_count}×
            </span>
          </>
        )}
      </div>

      {failed && document.error_message && (
        <div className="rounded-[4px] bg-danger-soft px-2 py-1.5 text-[11px] leading-relaxed text-danger">{document.error_message}</div>
      )}

      {failed ? (
        // A failed document must never be a dead end -- Retry/Delete are
        // always visible here, not tucked behind hover.
        <div className="flex items-center gap-2 pt-0.5">
          <button
            onClick={handleRetry}
            disabled={retrying}
            className="btn-secondary flex-1 !py-1.5 text-[12px]"
          >
            {retrying ? <Spinner size={12} /> : <RotateCw size={12} />}
            Retry
          </button>
          {onDelete && (
            <button
              onClick={(e) => {
                e.preventDefault();
                onDelete(document.id);
              }}
              className="btn-secondary !py-1.5 text-[12px] text-danger hover:bg-danger-soft"
            >
              <Trash2 size={12} />
              Delete
            </button>
          )}
        </div>
      ) : (
        onDelete && (
          <button
            onClick={(e) => {
              e.preventDefault();
              onDelete(document.id);
            }}
            className="absolute right-3 top-3 hidden rounded-[4px] p-1 text-ink-faint hover:bg-danger-soft hover:text-danger group-hover:block"
            title="Delete document"
          >
            <Trash2 size={13} />
          </button>
        )
      )}
    </div>
  );
}
