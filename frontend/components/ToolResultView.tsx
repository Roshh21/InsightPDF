"use client";

import { useState } from "react";
import { ChevronDown, ExternalLink, RotateCw } from "lucide-react";
import type {
  ComparisonContent,
  EntitiesContent,
  Flashcard,
  FlashcardsContent,
  ListContent,
  QuestionPaperContent,
  QuizContent,
  QuizQuestion,
  ResultKind,
  SectionsContent,
  TextContent,
  ToolExecuteResponse,
  WebResearchContent,
} from "@/lib/types";
import { CitationList } from "./Citations";

export default function ToolResultView({ result }: { result: ToolExecuteResponse }) {
  return (
    <div className="animate-fade-in">
      {result.warnings && result.warnings.length > 0 && (
        <div className="mb-3 rounded-[5px] border border-amber/30 bg-amber-soft px-3 py-2 text-[12.5px] text-amber-dark">
          {result.warnings.join(" ")}
        </div>
      )}
      <ResultBody kind={result.result_kind} content={result.content} />
      <CitationList citations={result.citations} />
    </div>
  );
}

function ResultBody({ kind, content }: { kind: ResultKind; content: any }) {
  switch (kind) {
    case "sections":
    case "report":
      return <SectionsView content={content} />;
    case "text":
      return <TextView content={content} />;
    case "list":
      return <ListView content={content} />;
    case "entities":
      return <EntitiesView content={content} />;
    case "comparison":
      return <ComparisonView content={content} />;
    case "quiz":
      return <QuizView content={content} />;
    case "question_paper":
      return <QuestionPaperView content={content} />;
    case "flashcards":
      return <FlashcardsView content={content} />;
    case "table":
      return <TableView content={content} />;
    default:
      return <pre className="overflow-x-auto text-[12px] text-ink-soft">{JSON.stringify(content, null, 2)}</pre>;
  }
}

function SectionsView({ content }: { content: SectionsContent & Partial<WebResearchContent> }) {
  return (
    <div>
      {content.summary && <p className="mb-4 text-[14.5px] leading-relaxed text-ink">{content.summary}</p>}
      <div className="flex flex-col gap-3">
        {content.sections?.map((s, i) => (
          <div key={i} className="card p-4">
            <div className="mb-1.5 font-serif text-[14.5px] font-semibold text-ink">{s.heading}</div>
            <div className="whitespace-pre-line text-[13.5px] leading-relaxed text-ink-soft">{s.content}</div>
          </div>
        ))}
      </div>
      {content.web_sources && content.web_sources.length > 0 && (
        <div className="mt-4 border-t border-line pt-3">
          <div className="mb-2 font-mono text-[10.5px] uppercase tracking-wide text-ink-faint">Web sources</div>
          <div className="flex flex-col gap-1.5">
            {content.web_sources.map((s, i) => (
              <a
                key={i}
                href={s.url}
                target="_blank"
                rel="noreferrer"
                className="flex items-center gap-1.5 text-[12.5px] text-cobalt hover:underline"
              >
                <ExternalLink size={11} className="shrink-0" />
                <span className="truncate">{s.title}</span>
              </a>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function TextView({ content }: { content: TextContent }) {
  return <p className="whitespace-pre-line text-[14.5px] leading-relaxed text-ink">{content.answer}</p>;
}

function ListView({ content }: { content: ListContent }) {
  return (
    <ul className="flex flex-col gap-1.5">
      {content.items?.map((item, i) => (
        <li key={i} className="flex gap-2 text-[13.5px] leading-relaxed text-ink">
          <span className="mt-2 h-1 w-1 shrink-0 rounded-full bg-amber" />
          <span>{item}</span>
        </li>
      ))}
    </ul>
  );
}

function EntitiesView({ content }: { content: EntitiesContent }) {
  if (!content.entities || content.entities.length === 0) {
    return <p className="text-[13px] text-ink-faint">No entities were identified for this document.</p>;
  }
  return (
    <div className="flex flex-wrap gap-2">
      {content.entities.map((e, i) => (
        <div key={i} className="rounded-[5px] border border-line bg-paper-soft px-2.5 py-1.5">
          <div className="text-[13px] font-medium text-ink">{e.name}</div>
          <div className="font-mono text-[10px] uppercase tracking-wide text-amber-dark">{e.type}</div>
        </div>
      ))}
    </div>
  );
}

function ComparisonView({ content }: { content: ComparisonContent }) {
  return (
    <div>
      <div className="overflow-x-auto rounded-md border border-line">
        <table className="w-full border-collapse text-[13px]">
          <thead>
            <tr className="bg-paper-dim">
              <th className="border-b border-line px-3 py-2 text-left font-mono text-[10.5px] uppercase tracking-wide text-ink-faint">
                Attribute
              </th>
              {content.rows?.map((r) => (
                <th
                  key={r.document_id}
                  className="border-b border-l border-line px-3 py-2 text-left font-serif text-[13.5px] font-semibold text-ink"
                >
                  {r.document_name}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {content.attributes?.map((attr) => (
              <tr key={attr} className="odd:bg-paper-soft even:bg-transparent">
                <td className="border-b border-line px-3 py-2 font-medium text-ink-soft">{attr}</td>
                {content.rows?.map((r) => (
                  <td key={r.document_id} className="border-b border-l border-line px-3 py-2 align-top text-ink">
                    {r.values?.[attr] ?? "—"}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {content.narrative && (
        <p className="mt-4 text-[13.5px] leading-relaxed text-ink-soft">{content.narrative}</p>
      )}
    </div>
  );
}

function QuizView({ content }: { content: QuizContent }) {
  return (
    <div className="flex flex-col gap-3">
      {content.questions?.map((q, i) => (
        <QuizQuestionCard key={i} index={i + 1} q={q} />
      ))}
    </div>
  );
}

function QuizQuestionCard({ index, q }: { index: number; q: QuizQuestion }) {
  const [selected, setSelected] = useState<string | null>(null);
  const [revealed, setRevealed] = useState(false);
  const hasOptions = q.options && q.options.length > 0;

  return (
    <div className="card p-4">
      <div className="mb-2.5 flex items-start gap-2">
        <span className="mt-0.5 font-mono text-[11px] text-ink-faint">{String(index).padStart(2, "0")}</span>
        <span className="text-[14px] font-medium text-ink">{q.question}</span>
      </div>
      {hasOptions ? (
        <div className="ml-6 flex flex-col gap-1.5">
          {q.options.map((opt, oi) => {
            const isCorrect = revealed && opt === q.correct_answer;
            const isWrongPick = revealed && selected === opt && opt !== q.correct_answer;
            return (
              <button
                key={oi}
                onClick={() => !revealed && setSelected(opt)}
                className={`rounded-[5px] border px-3 py-1.5 text-left text-[13px] transition-colors ${
                  isCorrect
                    ? "border-success/40 bg-success-soft text-success"
                    : isWrongPick
                    ? "border-danger/40 bg-danger-soft text-danger"
                    : selected === opt
                    ? "border-cobalt bg-cobalt-soft text-cobalt-dark"
                    : "border-line bg-paper-soft text-ink hover:bg-paper-dim"
                }`}
              >
                {opt}
              </button>
            );
          })}
        </div>
      ) : (
        <div className="ml-6">
          <textarea
            className="field text-[13px]"
            rows={2}
            placeholder="Your answer (not graded automatically for short-answer questions)"
          />
        </div>
      )}
      <div className="ml-6 mt-2.5 flex items-center gap-2">
        {!revealed ? (
          <button className="btn-ghost !px-0 text-cobalt" onClick={() => setRevealed(true)}>
            Reveal answer
          </button>
        ) : (
          <div className="w-full rounded-[5px] bg-paper-dim px-3 py-2 text-[12.5px] text-ink-soft">
            <span className="font-medium text-ink">Answer: {q.correct_answer}.</span> {q.explanation}
          </div>
        )}
        {q.marks != null && <span className="ml-auto font-mono text-[10.5px] text-ink-faint">{q.marks} marks</span>}
      </div>
    </div>
  );
}

function QuestionPaperView({ content }: { content: QuestionPaperContent }) {
  const [showKey, setShowKey] = useState(false);
  return (
    <div>
      <div className="mb-4 flex items-center justify-between border-b border-line pb-3">
        <div>
          <div className="font-serif text-[17px] font-semibold text-ink">{content.title}</div>
          <div className="mt-0.5 font-mono text-[11px] text-ink-faint">
            {content.total_marks} marks
            {content.duration_minutes ? ` · ${content.duration_minutes} min` : ""}
          </div>
        </div>
      </div>
      <div className="flex flex-col gap-5">
        {content.sections?.map((s, si) => (
          <div key={si}>
            <div className="mb-2 flex items-baseline justify-between">
              <div className="font-serif text-[14.5px] font-semibold text-ink">{s.section_title}</div>
              <div className="font-mono text-[10.5px] text-ink-faint">{s.marks_per_question} marks each</div>
            </div>
            {s.instructions && <p className="mb-2 text-[12.5px] italic text-ink-faint">{s.instructions}</p>}
            <ol className="flex flex-col gap-2">
              {s.questions.map((q, qi) => (
                <li key={qi} className="flex gap-2 text-[13.5px] leading-relaxed text-ink">
                  <span className="font-mono text-ink-faint">{qi + 1}.</span>
                  <span>{q}</span>
                </li>
              ))}
            </ol>
          </div>
        ))}
      </div>
      {content.answer_key && content.answer_key.length > 0 && (
        <div className="mt-5 border-t border-line pt-3">
          <button className="btn-ghost text-cobalt" onClick={() => setShowKey((v) => !v)}>
            <ChevronDown size={14} className={`transition-transform ${showKey ? "rotate-180" : ""}`} />
            {showKey ? "Hide" : "Show"} answer key
          </button>
          {showKey && (
            <ol className="mt-2 flex flex-col gap-1.5 rounded-[5px] bg-paper-dim p-3">
              {content.answer_key.map((a, i) => (
                <li key={i} className="text-[12.5px] text-ink-soft">
                  <span className="font-mono text-ink-faint">{i + 1}.</span> {a}
                </li>
              ))}
            </ol>
          )}
        </div>
      )}
    </div>
  );
}

function FlashcardsView({ content }: { content: FlashcardsContent }) {
  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
      {content.cards?.map((c, i) => (
        <FlipCard key={i} card={c} />
      ))}
    </div>
  );
}

function FlipCard({ card }: { card: Flashcard }) {
  const [flipped, setFlipped] = useState(false);
  return (
    <button
      onClick={() => setFlipped((v) => !v)}
      className="card flex min-h-[92px] flex-col justify-center px-4 py-3 text-left transition-colors hover:bg-paper-dim"
    >
      <div className="mb-1 flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-wide text-ink-faint">
        <RotateCw size={10} />
        {flipped ? "Answer" : "Term"}
      </div>
      <div className="text-[13.5px] leading-relaxed text-ink">{flipped ? card.back : card.front}</div>
    </button>
  );
}

function TableView({ content }: { content: { rows: string[][] } }) {
  if (!content.rows || content.rows.length === 0) {
    return <p className="text-[13px] text-ink-faint">No table found at that location.</p>;
  }
  const [header, ...rest] = content.rows;
  return (
    <div className="overflow-x-auto rounded-md border border-line">
      <table className="w-full border-collapse text-[12.5px]">
        <thead>
          <tr className="bg-paper-dim">
            {header.map((h, i) => (
              <th key={i} className="border-b border-line px-2.5 py-1.5 text-left font-medium text-ink">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rest.map((row, ri) => (
            <tr key={ri} className="odd:bg-paper-soft">
              {row.map((cell, ci) => (
                <td key={ci} className="border-b border-line px-2.5 py-1.5 text-ink-soft">
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
