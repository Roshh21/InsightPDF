import React from "react";

const SummaryViewer = ({ summary }) => {
  if (!summary) {
    return (
      <div className="card card-soft">
        <p style={{ color: "var(--text-muted)", fontSize: "0.9rem" }}>
          Once your PDF is analyzed, a structured cozy summary will appear here.
        </p>
      </div>
    );
  }

  const { documentType, sections = [] } = summary;

  return (
    <div className="card card-soft">
      <div className="stack-v">
        <div className="stack-h" style={{ justifyContent: "space-between" }}>
          <div>
            <h2 style={{ margin: 0, fontSize: "1.1rem" }}>Structured summary</h2>
            <p
              style={{
                margin: "4px 0 0",
                fontSize: "0.86rem",
                color: "var(--text-muted)",
              }}
            >
              Tailored to the detected document type.
            </p>
          </div>
          {documentType && (
            <div className="tag" style={{ alignSelf: "flex-start" }}>
              {documentType}
            </div>
          )}
        </div>

        <div className="scroll-panel">
          {sections.map((section, idx) => (
            <div
              key={idx}
              style={{
                marginTop: idx === 0 ? 0 : 14,
                paddingTop: idx === 0 ? 0 : 10,
                borderTop:
                  idx === 0
                    ? "none"
                    : "1px dashed rgba(244, 212, 160, 0.25)",
              }}
            >
              <h3
                style={{
                  margin: 0,
                  fontSize: "0.95rem",
                  color: "var(--accent)",
                }}
              >
                {section.title}
              </h3>
              {section.points && (
                <ul
                  style={{
                    margin: "6px 0 0 18px",
                    padding: 0,
                    fontSize: "0.88rem",
                    color: "var(--text-main)",
                  }}
                >
                  {section.points.map((point, pIdx) => (
                    <li key={pIdx} style={{ marginBottom: 4 }}>
                      {point}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default SummaryViewer;
