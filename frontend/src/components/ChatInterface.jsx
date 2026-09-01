import React, { useState } from "react";
import "../styles/ChatInterface.module.css";
import { askQuestion, generateQuiz } from "../utils/api";
import LoadingSpinner from "./LoadingSpinner";

const ChatInterface = ({ docId }) => {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      text: "Hi! Ask anything about this document, or say “Quiz me on this document” to start a quiz. 🍫📚",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const pushMessage = (msg) => setMessages((prev) => [...prev, msg]);

  const handleSend = async () => {
    const trimmed = input.trim();
    if (!trimmed || !docId) return;
    const isQuiz = /quiz me/i.test(trimmed);

    pushMessage({ role: "user", text: trimmed });
    setInput("");
    setLoading(true);

    try {
      if (isQuiz) {
        const quiz = await generateQuiz(docId);
        pushMessage({
          role: "assistant",
          text: "Here’s a quiz generated from your document:",
          quiz,
        });
      } else {
        const answer = await askQuestion(docId, trimmed);
        pushMessage({ role: "assistant", text: answer.text });
      }
    } catch (err) {
      pushMessage({
        role: "assistant",
        text:
          "Something went wrong while chatting with the document. Please try again.",
      });
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="card card-soft chat-root">
      <div className="chat-header">
        <div>
          <h2 className="chat-title">Document Q&A</h2>
          <p className="chat-subtitle">
            Ask follow-up questions and explore details with a grounded RAG
            assistant.
          </p>
        </div>
        <span className="tag">Agentic RAG</span>
      </div>

      <div className="scroll-panel chat-messages">
        {messages.map((m, idx) => (
          <div
            key={idx}
            className={
              m.role === "user"
                ? "chat-bubble chat-bubble-user"
                : "chat-bubble chat-bubble-assistant"
            }
          >
            <p className="chat-text">{m.text}</p>
            {m.quiz && (
              <ul className="chat-quiz-list">
                {m.quiz.questions?.map((q, qIdx) => (
                  <li key={qIdx} className="chat-quiz-item">
                    <strong>Q{qIdx + 1}.</strong> {q.prompt}
                  </li>
                ))}
              </ul>
            )}
          </div>
        ))}
        {loading && (
          <div className="chat-loading">
            <LoadingSpinner />
            <span>Agents are thinking…</span>
          </div>
        )}
      </div>

      <div className="chat-input-row">
        <textarea
          className="textarea"
          rows={2}
          placeholder={
            docId
              ? "Ask about methods, results, characters, key ideas…"
              : "Upload and analyze a PDF first to start chatting."
          }
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={!docId || loading}
        />
        <button
          className="btn"
          type="button"
          onClick={handleSend}
          disabled={!docId || loading}
          style={{
            opacity: !docId ? 0.6 : 1,
            cursor: !docId ? "not-allowed" : "pointer",
          }}
        >
          Send
        </button>
      </div>
    </div>
  );
};

export default ChatInterface;
