"use client";

import { AlertCircle, CheckCircle2, Clock, Loader2 } from "lucide-react";
import type { DocumentStatus } from "@/lib/types";

export function StatusBadge({ status }: { status: DocumentStatus | string }) {
  const map: Record<string, { cls: string; icon: any; label: string }> = {
    UPLOADED: { cls: "border-line text-ink-soft bg-paper-dim", icon: Clock, label: "Uploaded" },
    PROCESSING: { cls: "border-cobalt/30 text-cobalt bg-cobalt-soft", icon: Loader2, label: "Processing" },
    READY: { cls: "border-success/30 text-success bg-success-soft", icon: CheckCircle2, label: "Ready" },
    FAILED: { cls: "border-danger/30 text-danger bg-danger-soft", icon: AlertCircle, label: "Failed" },
  };
  const entry = map[status] || map.UPLOADED;
  const Icon = entry.icon;
  return (
    <span className={`badge ${entry.cls}`}>
      <Icon size={11} className={status === "PROCESSING" ? "animate-spin" : ""} />
      {entry.label}
    </span>
  );
}

export function Spinner({ size = 16, className = "" }: { size?: number; className?: string }) {
  return <Loader2 size={size} className={`animate-spin text-ink-faint ${className}`} />;
}

export function EmptyState({
  icon: Icon,
  title,
  description,
  action,
}: {
  icon: any;
  title: string;
  description?: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center rounded-md border border-dashed border-line px-6 py-14 text-center">
      <div className="mb-3 flex h-10 w-10 items-center justify-center rounded-full bg-paper-dim">
        <Icon size={18} className="text-ink-faint" />
      </div>
      <div className="font-serif text-[16px] font-semibold text-ink">{title}</div>
      {description && <p className="mt-1 max-w-sm text-[13px] text-ink-soft">{description}</p>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}

export function ConfidenceBar({ value }: { value: number | null | undefined }) {
  const pct = Math.round((value ?? 0) * 100);
  return (
    <div className="flex items-center gap-2">
      <div className="h-1.5 w-16 overflow-hidden rounded-full bg-paper-dim">
        <div
          className="h-full rounded-full bg-cobalt"
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="font-mono text-[11px] text-ink-faint">{pct}%</span>
    </div>
  );
}

export function PageHeader({
  eyebrow,
  title,
  description,
  actions,
}: {
  eyebrow?: string;
  title: string;
  description?: string;
  actions?: React.ReactNode;
}) {
  return (
    <div className="mb-6 flex flex-col flex-wrap items-start justify-between gap-4 sm:flex-row sm:items-start">
      <div className="min-w-0">
        {eyebrow && (
          <div className="mb-1 font-mono text-[11px] font-medium uppercase tracking-wider text-amber-dark">
            {eyebrow}
          </div>
        )}
        <h1 className="break-words font-serif text-[24px] font-semibold leading-tight text-ink">{title}</h1>
        {description && <p className="mt-1.5 max-w-2xl text-[13.5px] text-ink-soft">{description}</p>}
      </div>
      {actions && <div className="flex shrink-0 flex-wrap items-center gap-2">{actions}</div>}
    </div>
  );
}
