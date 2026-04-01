"use client";

import { useState, useRef, useCallback } from "react";
import { apiFetch } from "@/lib/api";
import { useToast } from "./Toast";

interface UploadZoneProps {
  onUploaded: () => void;
}

export default function UploadZone({ onUploaded }: UploadZoneProps) {
  const [uploading, setUploading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const { showToast } = useToast();

  const uploadFile = useCallback(
    async (file: File) => {
      setUploading(true);
      const formData = new FormData();
      formData.append("file", file);

      try {
        const res = await apiFetch("/api/v1/upload", {
          method: "POST",
          body: formData,
        });
        const data = await res.json();
        if (res.ok) {
          showToast(`${file.name} ingested — ${data.chunks_created} chunks`, "success");
          onUploaded();
        } else {
          showToast(data.detail || "Upload failed", "error");
        }
      } catch {
        showToast("Cannot connect to backend", "error");
      }
      setUploading(false);
    },
    [onUploaded, showToast]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const files = e.dataTransfer.files;
      for (let i = 0; i < files.length; i++) uploadFile(files[i]);
    },
    [uploadFile]
  );

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      className={`border-2 border-dashed rounded-xl p-5 text-center cursor-pointer transition-all ${
        dragging
          ? "border-cyan-400 bg-cyan-400/10"
          : "border-[#1e2d45] bg-[#1a2235] hover:border-cyan-400/50"
      }`}
      onClick={() => inputRef.current?.click()}
    >
      <div className="text-2xl mb-2">📄</div>
      <div className="text-xs text-slate-400 leading-relaxed">
        Drop PDF or TXT files
        <br />
        for financial analysis
      </div>
      <button
        disabled={uploading}
        className="mt-3 w-full py-2 rounded-lg bg-gradient-to-r from-cyan-500 to-purple-600 text-white text-xs font-medium hover:opacity-90 disabled:opacity-50 transition"
      >
        {uploading ? "Uploading..." : "Choose File"}
      </button>
      <input
        ref={inputRef}
        type="file"
        accept=".pdf,.txt"
        multiple
        className="hidden"
        onChange={(e) => {
          const files = e.target.files;
          if (files) for (let i = 0; i < files.length; i++) uploadFile(files[i]);
          e.target.value = "";
        }}
      />
    </div>
  );
}
