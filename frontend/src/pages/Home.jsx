import React from "react";
import { useNavigate } from "react-router-dom";
import UploadPDF from "../components/UploadPDF";

const Home = () => {
  const navigate = useNavigate();

  const handleUploadComplete = () => {
    navigate("/analyzer");
  };

  return (
    <div className="stack-v">
      <section className="card card-soft">
        <div className="stack-v">
          <h1 style={{ margin: 0, fontSize: "1.6rem" }}>
            Welcome to InsightPDF
          </h1>
          <p
            style={{
              margin: "6px 0 0",
              fontSize: "0.95rem",
              color: "var(--text-muted)",
            }}
          >
            A cozy, intelligent agentic RAG space where your PDFs become
            summaries, chats, and quizzes in a warm dark-brown theme.
          </p>

          <div className="stack-h" style={{ flexWrap: "wrap" }}>
            <span className="tag">Smart classification</span>
            <span className="tag">Structured summaries</span>
            <span className="tag">Document-grounded Q&A</span>
            <span className="tag">Quiz generator</span>
          </div>
        </div>
      </section>

      <UploadPDF onUploadComplete={handleUploadComplete} />
    </div>
  );
};

export default Home;
