"use client";

import { useCallback, useRef, useState } from "react";
import { FileUp, UploadCloud } from "lucide-react";

export default function UploadDropzone({
  onFiles,
  uploading,
}: {
  onFiles: (files: File[]) => void;
  uploading?: boolean;
}) {
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFiles = useCallback(
    (fileList: FileList | null) => {
      if (!fileList) return;
      const files = Array.from(fileList).filter((f) => f.name.toLowerCase().endsWith(".pdf"));
      if (files.length > 0) onFiles(files);
    },
    [onFiles]
  );

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        handleFiles(e.dataTransfer.files);
      }}
      onClick={() => inputRef.current?.click()}
      className={`flex cursor-pointer flex-col items-center justify-center rounded-md border-2 border-dashed px-6 py-10 text-center transition-colors ${
        dragOver ? "border-cobalt bg-cobalt-soft" : "border-line bg-paper-soft hover:bg-paper-dim"
      }`}
    >
      <input
        ref={inputRef}
        type="file"
        accept="application/pdf"
        multiple
        className="hidden"
        onChange={(e) => handleFiles(e.target.files)}
      />
      <div className="mb-3 flex h-11 w-11 items-center justify-center rounded-full bg-amber-soft">
        {uploading ? <FileUp size={18} className="text-amber-dark" /> : <UploadCloud size={18} className="text-amber-dark" />}
      </div>
      <div className="font-serif text-[15px] font-semibold text-ink">
        {uploading ? "Uploading…" : "Drop PDFs here, or click to browse"}
      </div>
      <p className="mt-1 text-[12.5px] text-ink-soft">
        Upload one or many documents at once — research papers, novels, textbooks, docs, or reports.
      </p>
    </div>
  );
}
