import React, { useState } from "react";
import UploadPDF from "../components/UploadPDF";
import SummaryViewer from "../components/SummaryViewer";
import ChatInterface from "../components/ChatInterface";
import QuizGenerator from "../components/QuizGenerator";

const Analyzer = () => {
  const [docId, setDocId] = useState(null);
  const [summary, setSummary] = useState(null);
  const [quiz, setQuiz] = useState(null);
  const [mode, setMode] = useState("summary");

  const handleUploadComplete = (payload) => {
    // payload: { docId, summary }
    setDocId(payload.docId);
    setSummary(payload.summary);
    setQuiz(null);
    setMode("summary");
  };

  return (
    <div className="stack-v">
      <UploadPDF onUploadComplete={handleUploadComplete} />

      {docId && (
        <>
          <section className="card" style={{ marginTop: 16 }}>
            <div style={{ display: "flex", gap: 8 }}>
              <button
                className={`btn ${mode === "summary" ? "" : "btn-ghost"}`}
                type="button"
                onClick={() => setMode("summary")}
              >
                Summary
              </button>
              <button
                className={`btn ${mode === "qa" ? "" : "btn-ghost"}`}
                type="button"
                onClick={() => setMode("qa")}
              >
                Q&A chatbot
              </button>
              <button
                className={`btn ${mode === "quiz" ? "" : "btn-ghost"}`}
                type="button"
                onClick={() => setMode("quiz")}
              >
                Quiz mode
              </button>
            </div>
          </section>

          {mode === "summary" && (
            <div style={{ marginTop: 16 }}>
              <SummaryViewer summary={summary} />
            </div>
          )}

          {mode === "qa" && (
            <div style={{ marginTop: 16 }}>
              <ChatInterface docId={docId} />
            </div>
          )}

          {mode === "quiz" && (
            <div style={{ marginTop: 16 }}>
              <QuizGenerator docId={docId} quiz={quiz} setQuiz={setQuiz} />
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default Analyzer;