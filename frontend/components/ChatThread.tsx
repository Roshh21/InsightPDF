"use client";

import { useEffect, useRef, useState } from "react";
import { ArrowUp, Sparkles, Wrench } from "lucide-react";
import { api, ApiError, streamChat } from "@/lib/api";
import type { ChatMessage, ChatSession } from "@/lib/types";
import { CitationList } from "./Citations";
import ToolResultView from "./ToolResultView";
import { Spinner } from "./ui";

export default function ChatThread({
  workspaceId,
  documentIds,
  spoilerLevel,
  placeholder,
  compact,
  suggestions,
}: {
  workspaceId: string;
  documentIds: string[];
  spoilerLevel?: string;
  placeholder?: string;
  compact?: boolean;
  suggestions?: string[];
}) {
  const [session, setSession] = useState<ChatSession | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [progress, setProgress] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const scopeKey = `${workspaceId}:${documentIds.slice().sort().join(",")}:${spoilerLevel || ""}`;

  useEffect(() => {
    let cancelled = false;
    setSession(null);
    setMessages([]);
    api
      .createChatSession(workspaceId, documentIds, spoilerLevel)
      .then((s) => !cancelled && setSession(s))
      .catch((e) => !cancelled && setError(e instanceof ApiError ? e.message : "Could not start chat session."));
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scopeKey]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, progress]);

  async function send(text?: string) {
    const content = (text ?? input).trim();
    if (!content || !session || sending) return;
    setInput("");
    setError(null);
    setMessages((m) => [
      ...m,
      { id: `local-${Date.now()}`, role: "user", content, citations: [], tool_used: null, result_payload: null, model_used: null, provider_used: null, latency_ms: null, created_at: new Date().toISOString() },
    ]);
    setSending(true);
    setProgress("Starting");
    try {
      for await (const event of streamChat(session.id, content)) {
        if (event.event === "progress") {
          setProgress(event.label);
        } else if (event.event === "error") {
          setError(event.message);
          setSending(false);
          setProgress(null);
        } else if (event.event === "done") {
          setMessages((m) => [...m, event.assistant_message]);
          setSending(false);
          setProgress(null);
        }
      }
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Chat request failed.");
      setSending(false);
      setProgress(null);
    }
  }

  return (
    <div className="flex flex-col">
      <div ref={scrollRef} className={`flex flex-col gap-4 overflow-y-auto ${compact ? "max-h-[420px]" : "min-h-[50vh]"} pr-1`}>
        {messages.length === 0 && !sending && (
          <div className="flex flex-col items-center justify-center gap-3 py-10 text-center">
            <div className="flex h-9 w-9 items-center justify-center rounded-full bg-cobalt-soft">
              <Sparkles size={16} className="text-cobalt" />
            </div>
            <p className="max-w-xs text-[13px] text-ink-soft">
              Ask a question, or reference what you just ran — e.g. &ldquo;explain the second section more simply.&rdquo;
            </p>
            {suggestions && suggestions.length > 0 && (
              <div className="flex flex-wrap justify-center gap-1.5">
                {suggestions.map((s) => (
                  <button key={s} onClick={() => send(s)} className="rounded-full border border-line bg-paper-soft px-2.5 py-1 text-[11.5px] text-ink-soft hover:bg-paper-dim">
                    {s}
                  </button>
                ))}
              </div>
            )}
          </div>
        )}

        {messages.map((m) => (
          <MessageBubble key={m.id} message={m} />
        ))}

        {sending && (
          <div className="flex items-center gap-2 self-start rounded-[6px] bg-paper-dim px-3 py-2 text-[12.5px] text-ink-soft">
            <Spinner size={13} />
            {progress || "Working…"}
          </div>
        )}
        {error && <div className="self-start rounded-[6px] border border-danger/30 bg-danger-soft px-3 py-2 text-[12.5px] text-danger">{error}</div>}
      </div>

      <div className="mt-3 flex items-end gap-2 border-t border-line pt-3">
        <textarea
          className="field resize-none text-[13.5px]"
          rows={compact ? 2 : 2}
          placeholder={placeholder || "Ask about this document…"}
          value={input}
          disabled={!session || sending}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send();
            }
          }}
        />
        <button className="btn-primary h-[38px] w-[38px] shrink-0 !px-0" disabled={!session || sending || !input.trim()} onClick={() => send()}>
          {sending ? <Spinner size={14} /> : <ArrowUp size={15} />}
        </button>
      </div>
    </div>
  );
}

function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";
  return (
    <div className={`flex flex-col ${isUser ? "items-end" : "items-start"}`}>
      <div
        className={`max-w-[85%] rounded-[8px] px-3.5 py-2.5 text-[13.5px] leading-relaxed ${
          isUser ? "bg-cobalt text-paper-soft" : "border border-line bg-paper-soft text-ink"
        }`}
      >
        {message.tool_used && message.result_payload ? (
          <div>
            <div className="mb-2 flex items-center gap-1.5 font-mono text-[10.5px] uppercase tracking-wide text-amber-dark">
              <Wrench size={11} /> Ran: {message.result_payload.title || message.tool_used}
            </div>
            <ToolResultView result={message.result_payload} />
          </div>
        ) : (
          <>
            <span className="whitespace-pre-line">{message.content}</span>
            {!isUser && <CitationList citations={message.citations} />}
          </>
        )}
      </div>
    </div>
  );
}
