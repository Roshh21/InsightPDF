import React, { useState } from "react";
import { generateQuiz } from "../utils/api";

const QuizGenerator = ({ docId, quiz, setQuiz }) => {
  const [loading, setLoading] = useState(false);

  const handleGenerate = async () => {
    if (!docId) return;
    setLoading(true);
    try {
      const result = await generateQuiz(docId, 5);
      setQuiz(result);
    } catch (e) {
      setQuiz({
        questions: [
          {
            type: "short",
            prompt: "Quiz generation failed. Try again.",
            options: [],
            answer: "",
          },
        ],
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card card-soft">
      <h2 style={{ marginTop: 0, fontSize: "1.05rem" }}>Quiz mode</h2>
      <p
        style={{
          margin: "4px 0 12px",
          fontSize: "0.86rem",
          color: "var(--text-muted)",
        }}
      >
        Generate questions grounded only in the uploaded PDF.
      </p>

      <button
        className="btn"
        type="button"
        onClick={handleGenerate}
        disabled={loading}
      >
        {loading ? "Creating quiz…" : "Generate quiz"}
      </button>

      {quiz && (
        <ul style={{ marginTop: 16, paddingLeft: 18, fontSize: "0.9rem" }}>
          {quiz.questions?.map((q, idx) => (
            <li key={idx} style={{ marginBottom: 10 }}>
              <strong>Q{idx + 1} ({q.type}):</strong> {q.prompt}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

export default QuizGenerator;