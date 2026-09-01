import React, { useState } from "react";
import "./styles/globals.css";
import UploadPDF from "./components/UploadPDF";
import SummaryViewer from "./components/SummaryViewer";
import ChatInterface from "./components/ChatInterface";
import QuizGenerator from "./components/QuizGenerator";

const App = () => {
  const [docId, setDocId] = useState(null);
  const [summary, setSummary] = useState(null);
  const [quiz, setQuiz] = useState(null);
  const [mode, setMode] = useState("summary"); // "summary" | "qa" | "quiz"

  const handleUploadComplete = (payload) => {
    // payload from backend: { docId, summary }
    setDocId(payload.docId);
    setSummary(payload.summary);
    setQuiz(null);
    setMode("summary");
  };

  return (
    <div className="app-root">
      <header className="header">
        <div className="header-inner">
          <span className="header-title">InsightPDF</span>
          <span className="header-subtitle">
            Intelligent agentic PDF analyzer
          </span>
        </div>
      </header>

      <main className="app-main">
        <div className="stack-v">
          {/* Upload section */}
          <UploadPDF onUploadComplete={handleUploadComplete} />

          {/* Mode selector appears only after a PDF is processed */}
          {docId && (
            <section className="card" style={{ marginTop: 16 }}>
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
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
          )}

          {/* Panels for each mode */}
          {docId && mode === "summary" && (
            <div style={{ marginTop: 16 }}>
              <SummaryViewer summary={summary} />
            </div>
          )}

          {docId && mode === "qa" && (
            <div style={{ marginTop: 16 }}>
              <ChatInterface docId={docId} />
            </div>
          )}

          {docId && mode === "quiz" && (
            <div style={{ marginTop: 16 }}>
              <QuizGenerator docId={docId} quiz={quiz} setQuiz={setQuiz} />
            </div>
          )}
        </div>
      </main>
    </div>
  );
};

export default App;