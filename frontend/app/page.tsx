"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { FolderKanban, Plus, Trash2, X } from "lucide-react";
import { api, ApiError } from "@/lib/api";
import type { Workspace } from "@/lib/types";
import { EmptyState, PageHeader, Spinner } from "@/components/ui";
import { formatDate } from "@/lib/format";

export default function DashboardPage() {
  const [workspaces, setWorkspaces] = useState<Workspace[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showCreate, setShowCreate] = useState(false);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [creating, setCreating] = useState(false);

  function refresh() {
    api
      .listWorkspaces()
      .then(setWorkspaces)
      .catch((e) => setError(e instanceof ApiError ? e.message : "Failed to load workspaces."));
  }

  useEffect(refresh, []);

  async function handleCreate(e: React.FormEvent) {
    e.preventDefault();
    if (!name.trim()) return;
    setCreating(true);
    try {
      await api.createWorkspace(name.trim(), description.trim() || undefined);
      setName("");
      setDescription("");
      setShowCreate(false);
      refresh();
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Failed to create workspace.");
    } finally {
      setCreating(false);
    }
  }

  async function handleDelete(id: string) {
    if (!confirm("Delete this workspace and all its documents? This cannot be undone.")) return;
    await api.deleteWorkspace(id);
    refresh();
  }

  return (
    <div className="mx-auto max-w-5xl px-4 py-8 sm:px-6 md:px-8 md:py-10">
      <PageHeader
        eyebrow="Document Intelligence"
        title="Workspaces"
        description="Each workspace is an isolated collection of documents — upload, classify, and interrogate them together."
        actions={
          <button className="btn-primary" onClick={() => setShowCreate((v) => !v)}>
            <Plus size={15} /> New workspace
          </button>
        }
      />

      {showCreate && (
        <form onSubmit={handleCreate} className="card mb-6 flex flex-col gap-3 p-4 animate-fade-in">
          <div className="flex items-center justify-between">
            <div className="font-serif text-[15px] font-semibold text-ink">Create workspace</div>
            <button type="button" onClick={() => setShowCreate(false)} className="text-ink-faint hover:text-ink">
              <X size={16} />
            </button>
          </div>
          <input
            className="field"
            placeholder="Name — e.g. 'Transformer papers' or 'Q3 board prep'"
            value={name}
            onChange={(e) => setName(e.target.value)}
            autoFocus
          />
          <textarea
            className="field"
            rows={2}
            placeholder="Description (optional)"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
          />
          <button className="btn-primary self-start" disabled={creating || !name.trim()}>
            {creating ? <Spinner size={14} /> : <Plus size={15} />}
            Create
          </button>
        </form>
      )}

      {error && <div className="mb-4 rounded-[5px] border border-danger/30 bg-danger-soft px-3 py-2 text-[13px] text-danger">{error}</div>}

      {!workspaces ? (
        <div className="flex items-center gap-2 py-10 text-ink-soft">
          <Spinner size={16} /> Loading workspaces…
        </div>
      ) : workspaces.length === 0 ? (
        <EmptyState
          icon={FolderKanban}
          title="No workspaces yet"
          description="Create your first workspace to start uploading documents."
          action={
            <button className="btn-primary" onClick={() => setShowCreate(true)}>
              <Plus size={15} /> New workspace
            </button>
          }
        />
      ) : (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {workspaces.map((ws) => (
            <Link
              key={ws.id}
              href={`/workspaces/${ws.id}`}
              className="card group relative flex flex-col gap-2 p-4 transition-shadow hover:shadow-pop"
            >
              <div className="flex h-8 w-8 items-center justify-center rounded-[5px] bg-cobalt-soft">
                <FolderKanban size={15} className="text-cobalt" />
              </div>
              <div className="truncate font-serif text-[15.5px] font-semibold text-ink group-hover:text-cobalt" title={ws.name}>{ws.name}</div>
              {ws.description && <p className="line-clamp-2 text-[12.5px] text-ink-soft">{ws.description}</p>}
              <div className="mt-1 font-mono text-[11px] text-ink-faint">Created {formatDate(ws.created_at)}</div>
              <button
                onClick={(e) => {
                  e.preventDefault();
                  handleDelete(ws.id);
                }}
                className="absolute right-3 top-3 hidden rounded-[4px] p-1 text-ink-faint hover:bg-danger-soft hover:text-danger group-hover:block"
              >
                <Trash2 size={13} />
              </button>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
