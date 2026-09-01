import React, { useRef, useState } from "react";
import "./../styles/UploadPDF.module.css";
import { uploadPDF } from "../utils/api";

const UploadPDF = ({ onUploadComplete }) => {
  const inputRef = useRef(null);
  const [dragActive, setDragActive] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");
  const [selectedFile, setSelectedFile] = useState(null);

  const handleUpload = async () => {
    if (!selectedFile) return;
    if (selectedFile.type !== "application/pdf") {
      setErrorMsg("Please upload a valid PDF file.");
      return;
    }
  
    setErrorMsg("");
    setUploading(true);
    try {
      const result = await uploadPDF(selectedFile); // backend call
      onUploadComplete?.(result);                   // { docId, summary }
    } catch (err) {
      setErrorMsg("Upload failed. Please try again.");
    } finally {
      setUploading(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    const file = e.dataTransfer.files?.[0];
    if (!file) return;
    setSelectedFile(file);
    setErrorMsg("");
  };

  const handleChange = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setSelectedFile(file);
    setErrorMsg("");
  };

  const clearSelection = () => {
    setSelectedFile(null);
    setErrorMsg("");
    if (inputRef.current) {
      inputRef.current.value = "";
    }
  };

  return (
    <div className="card card-soft upload-root">
      <div
        className={`upload-dropzone ${dragActive ? "upload-dropzone-active" : ""}`}
        onDragEnter={(e) => {
          e.preventDefault();
          setDragActive(true);
        }}
        onDragOver={(e) => e.preventDefault()}
        onDragLeave={(e) => {
          e.preventDefault();
          setDragActive(false);
        }}
        onDrop={handleDrop}
      >
        <div className="upload-inner">
          <h2 className="upload-title">Upload a PDF</h2>
          <p className="upload-subtitle">
            Drag and drop a document here or choose a file. InsightPDF will
            classify, summarize, and prepare it for Q&amp;A.
          </p>

          <div className="upload-controls">
            <button
              className="btn"
              type="button"
              onClick={() => inputRef.current?.click()}
              disabled={uploading}
            >
              {selectedFile ? "Change PDF" : "Choose PDF"}
            </button>
            {selectedFile && (
              <button
                type="button"
                className="btn btn-ghost"
                onClick={clearSelection}
                disabled={uploading}
              >
                Remove file
              </button>
            )}
          </div>

          <input
            ref={inputRef}
            type="file"
            accept="application/pdf"
            hidden
            onChange={handleChange}
          />

          <div className="upload-file-info">
            {selectedFile ? (
              <span className="upload-file-name">
                Selected: {selectedFile.name}
              </span>
            ) : (
              <span className="upload-file-placeholder">
                No file selected yet.
              </span>
            )}
          </div>

          <div className="upload-action-row">
            <button
              className="btn"
              type="button"
              onClick={handleFileUpload}
              disabled={!selectedFile || uploading}
              style={{
                opacity: !selectedFile ? 0.6 : 1,
                cursor: !selectedFile ? "not-allowed" : "pointer",
              }}
            >
              {uploading ? "Uploading…" : "Upload and analyze"}
            </button>
          </div>

          <div className="upload-hints">
            <span className="tag">PDF only</span>
            <span className="tag">Best under 20MB</span>
          </div>
        </div>
      </div>
      {errorMsg && <p className="upload-error">{errorMsg}</p>}
    </div>
  );
};

export default UploadPDF;