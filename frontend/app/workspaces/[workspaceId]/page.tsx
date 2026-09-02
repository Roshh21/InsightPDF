"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { FileText, Trash2, X } from "lucide-react";
import { api, ApiError } from "@/lib/api";
import type { Document, UploadError, WorkspaceDetail } from "@/lib/types";
import { EmptyState, PageHeader, Spinner } from "@/components/ui";
import UploadDropzone from "@/components/UploadDropzone";
import DocumentCard from "@/components/DocumentCard";

const ACTIVE_STATUSES = new Set(["UPLOADED", "PROCESSING"]);

export default function WorkspaceOverviewPage() {
  const params = useParams<{ workspaceId?: string }>();
  const router = useRouter();
  const workspaceId = params?.workspaceId;

  const [workspace, setWorkspace] = useState<WorkspaceDetail | null>(null);
  const [documents, setDocuments] = useState<Document[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploadErrors, setUploadErrors] = useState<UploadError[]>([]);
  const [uploading, setUploading] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const loadAll = useCallback(async () => {
    if (!workspaceId) return;

    try {
      const [ws, docs] = await Promise.all([
        api.getWorkspace(workspaceId),
        api.listDocuments(workspaceId),
      ]);

      setWorkspace(ws);
      setDocuments(docs);
    } catch (e) {
      setError(
        e instanceof ApiError
          ? e.message
          : "Failed to load workspace."
      );
    }
  }, [workspaceId]);

  useEffect(() => {
    if (!workspaceId) return;

    loadAll();
  }, [workspaceId, loadAll]);

  useEffect(() => {
    const hasActive = documents?.some((d) =>
      ACTIVE_STATUSES.has(d.status)
    );

    if (hasActive && !pollRef.current) {
      pollRef.current = setInterval(loadAll, 3000);
    } else if (!hasActive && pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }

    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, [documents, loadAll]);

  async function handleUpload(files: File[]) {
    if (!workspaceId) return;

    setUploading(true);
    setError(null);
    setUploadErrors([]);

    try {
      // Every file is validated and ingested independently on the backend.
      // A rejected/unreadable file is reported by name here without ever
      // blocking or hiding the files that succeeded.
      const result = await api.uploadDocuments(workspaceId, files);

      if (result.errors && result.errors.length > 0) {
        setUploadErrors(result.errors);
      }

      await loadAll();
    } catch (e) {
      setError(
        e instanceof ApiError
          ? e.message
          : "Upload failed."
      );
    } finally {
      setUploading(false);
    }
  }

  async function handleDeleteDocument(id: string) {
    if (!confirm("Delete this document?")) return;

    try {
      await api.deleteDocument(id);
      await loadAll();
    } catch (e) {
      setError(
        e instanceof ApiError
          ? e.message
          : "Failed to delete document."
      );
    }
  }

  async function handleRetryDocument(id: string) {
    try {
      await api.retryDocument(id);
      await loadAll();
    } catch (e) {
      setError(
        e instanceof ApiError
          ? e.message
          : "Failed to retry document."
      );
    }
  }

  async function handleDeleteWorkspace() {
    if (!workspaceId) return;

    if (
      !confirm(
        `Delete workspace "${workspace?.name}" and all its documents? This cannot be undone.`
      )
    ) {
      return;
    }

    try {
      await api.deleteWorkspace(workspaceId);
      router.push("/");
    } catch (e) {
      setError(
        e instanceof ApiError
          ? e.message
          : "Failed to delete workspace."
      );
    }
  }

  if (!workspaceId) {
    return (
      <div className="mx-auto max-w-6xl px-4 py-8 sm:px-6 md:px-8 md:py-10">
        <div className="rounded-[5px] border border-danger/30 bg-danger-soft px-3 py-2 text-[13px] text-danger">
          Invalid workspace URL.
        </div>
      </div>
    );
  }

  if (!workspace || !documents) {
    return (
      <div className="mx-auto max-w-6xl px-4 py-8 sm:px-6 md:px-8 md:py-10">
        {error ? (
          <div className="rounded-[5px] border border-danger/30 bg-danger-soft px-3 py-2 text-[13px] text-danger">
            {error}
          </div>
        ) : (
          <div className="flex items-center gap-2 py-10 text-ink-soft">
            <Spinner size={16} /> Loading workspace…
          </div>
        )}
      </div>
    );
  }

  const statusCounts = workspace.stats.by_status;

  return (
    <div className="mx-auto max-w-6xl px-4 py-8 sm:px-6 md:px-8 md:py-10">
      <PageHeader
        eyebrow={`${workspace.stats.total_documents} document${
          workspace.stats.total_documents === 1 ? "" : "s"
        }`}
        title={workspace.name}
        description={workspace.description || undefined}
        actions={
          <button
            type="button"
            className="btn-secondary text-danger hover:bg-danger-soft"
            onClick={handleDeleteWorkspace}
          >
            <Trash2 size={14} /> Delete workspace
          </button>
        }
      />

      {statusCounts && Object.keys(statusCounts).length > 0 && (
        <div className="mb-6 flex flex-wrap gap-2">
          {Object.entries(statusCounts).map(([status, count]) => (
            <div
              key={status}
              className="rounded-[5px] border border-line bg-paper-soft px-2.5 py-1 font-mono text-[11px] text-ink-soft"
            >
              {count} {status.toLowerCase()}
            </div>
          ))}
        </div>
      )}

      <div className="mb-8">
        <UploadDropzone
          onFiles={handleUpload}
          uploading={uploading}
        />
      </div>

      {uploadErrors.length > 0 && (
        <div className="mb-6 rounded-md border border-danger/30 bg-danger-soft p-3 animate-fade-in">
          <div className="mb-1.5 flex items-center justify-between">
            <span className="text-[12.5px] font-medium text-danger">
              {uploadErrors.length} file
              {uploadErrors.length > 1 ? "s" : ""} couldn&apos;t be
              uploaded
            </span>

            <button
              type="button"
              onClick={() => setUploadErrors([])}
              className="text-danger hover:opacity-70"
            >
              <X size={14} />
            </button>
          </div>

          <ul className="flex flex-col gap-1">
            {uploadErrors.map((e, i) => (
              <li
                key={i}
                className="text-[12px] text-danger"
              >
                <span className="font-mono">{e.filename}</span> —{" "}
                {e.message}
              </li>
            ))}
          </ul>
        </div>
      )}

      {error && (
        <div className="mb-4 rounded-[5px] border border-danger/30 bg-danger-soft px-3 py-2 text-[13px] text-danger">
          {error}
        </div>
      )}

      {documents.length === 0 ? (
        <EmptyState
          icon={FileText}
          title="No documents yet"
          description="Drop one or more PDFs above to get started."
        />
      ) : (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {documents.map((doc) => (
            <DocumentCard
              key={doc.id}
              document={doc}
              workspaceId={workspaceId}
              onDelete={handleDeleteDocument}
              onRetry={handleRetryDocument}
            />
          ))}
        </div>
      )}
    </div>
  );
}
