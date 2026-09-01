"use client";

import { useState } from "react";
import { Check, X } from "lucide-react";
import type { Citation } from "@/lib/types";

export function CitationTab({ citation }: { citation: Citation }) {
  const [open, setOpen] = useState(false);
  const unsupported = citation.supported === false;
  return (
    <span className="relative inline-block">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className={`citation-tab ${unsupported ? "unsupported" : ""}`}
        title={unsupported ? "Could not be verified against retrieved evidence" : "Grounded in retrieved evidence"}
      >
        {citation.supported === true && <Check size={10} strokeWidth={3} />}
        {unsupported && <X size={10} strokeWidth={3} />}
        {citation.document_name}
        <span className="opacity-70">· p.{citation.page}</span>
      </button>
      {open && (
        <span className="absolute left-0 top-full z-20 mt-1.5 block w-72 rounded-md border border-line bg-paper-soft p-3 text-left text-[12.5px] leading-relaxed text-ink shadow-pop animate-fade-in">
          <span className="mb-1 flex items-center justify-between font-mono text-[10.5px] text-ink-faint">
            <span>
              {citation.document_name} · page {citation.page}
              {citation.section ? ` · ${citation.section}` : ""}
            </span>
          </span>
          <span className="block italic text-ink-soft">&ldquo;{citation.excerpt}&rdquo;</span>
        </span>
      )}
    </span>
  );
}

export function CitationList({ citations }: { citations: Citation[] }) {
  if (!citations || citations.length === 0) return null;
  return (
    <div className="mt-3 flex flex-wrap items-center gap-1.5 border-t border-line pt-3">
      <span className="mr-0.5 font-mono text-[10.5px] uppercase tracking-wide text-ink-faint">Sources</span>
      {citations.map((c, i) => (
        <CitationTab key={`${c.document_id}-${c.page}-${i}`} citation={c} />
      ))}
    </div>
  );
}
