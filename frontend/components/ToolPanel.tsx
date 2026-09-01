"use client";

import { useState } from "react";
import { Play, X } from "lucide-react";
import type { ToolCatalogEntry, ToolExecuteResponse } from "@/lib/types";
import { api, ApiError } from "@/lib/api";
import { getToolIcon } from "@/lib/icons";
import { TOOL_PARAM_FIELDS } from "@/lib/toolParams";
import { Spinner } from "./ui";

const CATEGORY_LABELS: Record<string, string> = {
  universal: "Common Tools",
  research_paper: "Research Paper Tools",
  literature: "Literature Tools",
  study_material: "Study Tools",
  technical_documentation: "Technical Documentation Tools",
  business_report: "Business Report Tools",
};

const CATEGORY_ORDER = [
  "universal",
  "research_paper",
  "literature",
  "study_material",
  "technical_documentation",
  "business_report",
];

export default function ToolPanel({
  tools,
  workspaceId,
  documentIds,
  onResult,
  disabled,
}: {
  tools: ToolCatalogEntry[];
  workspaceId: string;
  documentIds: string[];
  onResult: (tool: ToolCatalogEntry, result: ToolExecuteResponse) => void;
  disabled?: boolean;
}) {
  const [activeToolId, setActiveToolId] = useState<string | null>(null);
  const [params, setParams] = useState<Record<string, unknown>>({});
  const [loading, setLoading] = useState(false);
  const [runningLabel, setRunningLabel] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const grouped: Record<string, ToolCatalogEntry[]> = {};
  for (const t of tools) {
    (grouped[t.category] ||= []).push(t);
  }
  const categories = CATEGORY_ORDER.filter((c) => grouped[c]?.length);

  const activeTool = tools.find((t) => t.id === activeToolId) || null;
  const fields = activeTool ? TOOL_PARAM_FIELDS[activeTool.id] : undefined;

  function selectTool(tool: ToolCatalogEntry) {
    setError(null);
    if (TOOL_PARAM_FIELDS[tool.id]) {
      setActiveToolId(tool.id);
      const defaults: Record<string, unknown> = {};
      TOOL_PARAM_FIELDS[tool.id].forEach((f) => {
        if (f.default !== undefined) defaults[f.key] = f.default;
      });
      setParams(defaults);
    } else {
      run(tool, {});
    }
  }

  async function run(tool: ToolCatalogEntry, p: Record<string, unknown>) {
    setLoading(true);
    setRunningLabel(tool.name);
    setError(null);
    try {
      const result = await api.executeTool(workspaceId, tool.id, documentIds, p);
      onResult(tool, result);
      setActiveToolId(null);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Something went wrong running this tool.");
    } finally {
      setLoading(false);
      setRunningLabel(null);
    }
  }

  return (
    <div>
      {categories.map((cat) => (
        <div key={cat} className="mb-5">
          <div className="mb-2 font-mono text-[10.5px] uppercase tracking-wide text-ink-faint">
            {CATEGORY_LABELS[cat] || cat}
          </div>
          <div className="grid grid-cols-2 gap-1.5 sm:grid-cols-1 lg:grid-cols-2">
            {grouped[cat].map((tool) => {
              const Icon = getToolIcon(tool.icon);
              return (
                <button
                  key={tool.id}
                  onClick={() => selectTool(tool)}
                  disabled={disabled || loading}
                  title={tool.description}
                  className="flex items-center gap-2 rounded-[5px] border border-line bg-paper-soft px-2.5 py-2 text-left text-[12.5px] font-medium text-ink transition-colors hover:border-cobalt/40 hover:bg-cobalt-soft disabled:cursor-not-allowed disabled:opacity-50"
                >
                  <Icon size={14} className="shrink-0 text-ink-faint" />
                  <span className="truncate">{tool.name}</span>
                </button>
              );
            })}
          </div>
        </div>
      ))}

      {activeTool && fields && (
        <div className="card sticky top-4 mt-2 p-4 animate-fade-in">
          <div className="mb-3 flex items-center justify-between">
            <div className="font-serif text-[14px] font-semibold text-ink">{activeTool.name}</div>
            <button onClick={() => setActiveToolId(null)} className="text-ink-faint hover:text-ink">
              <X size={15} />
            </button>
          </div>
          <div className="flex flex-col gap-3">
            {fields.map((f) => (
              <div key={f.key}>
                <label className="mb-1 block text-[12px] font-medium text-ink-soft">{f.label}</label>
                {f.type === "select" ? (
                  <select
                    className="field"
                    value={(params[f.key] as string) ?? ""}
                    onChange={(e) => setParams((p) => ({ ...p, [f.key]: e.target.value }))}
                  >
                    {f.options?.map((o) => (
                      <option key={o.value} value={o.value}>
                        {o.label}
                      </option>
                    ))}
                  </select>
                ) : f.type === "checkbox" ? (
                  <label className="flex items-center gap-2 text-[13px] text-ink">
                    <input
                      type="checkbox"
                      checked={!!params[f.key]}
                      onChange={(e) => setParams((p) => ({ ...p, [f.key]: e.target.checked }))}
                    />
                    Yes
                  </label>
                ) : f.type === "textarea" ? (
                  <textarea
                    className="field"
                    rows={2}
                    placeholder={f.placeholder}
                    value={(params[f.key] as string) ?? ""}
                    onChange={(e) => setParams((p) => ({ ...p, [f.key]: e.target.value }))}
                  />
                ) : (
                  <input
                    className="field"
                    type={f.type === "number" ? "number" : "text"}
                    placeholder={f.placeholder}
                    value={(params[f.key] as string | number) ?? ""}
                    onChange={(e) =>
                      setParams((p) => ({
                        ...p,
                        [f.key]: f.type === "number" ? Number(e.target.value) : e.target.value,
                      }))
                    }
                  />
                )}
              </div>
            ))}
          </div>
          {error && <div className="mt-3 text-[12.5px] text-danger">{error}</div>}
          <button className="btn-primary mt-4 w-full" disabled={loading} onClick={() => run(activeTool, params)}>
            {loading ? <Spinner size={14} /> : <Play size={14} />}
            Run {activeTool.name}
          </button>
        </div>
      )}

      {loading && !activeTool && (
        <div className="mt-3 flex items-center gap-2 text-[12.5px] text-ink-soft">
          <Spinner size={14} /> Running {runningLabel}…
        </div>
      )}
      {error && !activeTool && <div className="mt-3 text-[12.5px] text-danger">{error}</div>}
    </div>
  );
}
