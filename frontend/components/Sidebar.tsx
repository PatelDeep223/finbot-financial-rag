"use client";

import { useState, useEffect, useCallback } from "react";
import { apiFetch } from "@/lib/api";
import { useStats } from "@/hooks/useStats";
import { DocumentInfo } from "@/lib/types";
import UploadZone from "./UploadZone";
import { FiFileText, FiFile } from "react-icons/fi";

export default function Sidebar() {
  const [docs, setDocs] = useState<DocumentInfo[]>([]);
  const stats = useStats();

  const loadDocs = useCallback(async () => {
    try {
      const res = await apiFetch("/api/v1/documents");
      if (res.ok) {
        const data = await res.json();
        setDocs(data.documents || []);
      }
    } catch {}
  }, []);

  useEffect(() => {
    loadDocs();
  }, [loadDocs]);

  const uptime = stats?.uptime_seconds
    ? stats.uptime_seconds > 3600
      ? `${Math.floor(stats.uptime_seconds / 3600)}h`
      : `${Math.round(stats.uptime_seconds)}s`
    : "0s";

  return (
    <aside className="w-72 bg-[#111827] border-r border-[#1e2d45] flex flex-col overflow-hidden shrink-0">
      {/* Upload */}
      <div className="p-4 border-b border-[#1e2d45]">
        <div className="text-[11px] text-slate-500 tracking-widest uppercase mb-3">
          Upload Documents
        </div>
        <UploadZone onUploaded={loadDocs} />
      </div>

      {/* Documents */}
      <div className="p-4 border-b border-[#1e2d45] flex-1 overflow-y-auto">
        <div className="text-[11px] text-slate-500 tracking-widest uppercase mb-3">
          Loaded Documents
        </div>
        {docs.length === 0 ? (
          <p className="text-xs text-slate-600 text-center py-3">No documents yet</p>
        ) : (
          <div className="space-y-2">
            {docs.map((doc) => (
              <div
                key={doc.filename}
                className="flex items-center gap-2.5 p-2.5 rounded-lg bg-[#1a2235] border border-[#1e2d45] text-xs"
              >
                {doc.filename.endsWith(".pdf") ? (
                  <FiFileText className="w-5 h-5 text-red-400 shrink-0" />
                ) : (
                  <FiFile className="w-5 h-5 text-slate-400 shrink-0" />
                )}
                <div className="min-w-0 flex-1">
                  <div className="text-white font-medium truncate">{doc.filename}</div>
                  <div className="text-slate-500 text-[11px]">
                    {doc.size_kb} KB{doc.chunks_created ? ` · ${doc.chunks_created} chunks` : ""}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Stats */}
      <div className="p-4 mt-auto">
        <div className="text-[11px] text-slate-500 tracking-widest uppercase mb-3">
          System Stats
        </div>
        <div className="grid grid-cols-2 gap-2">
          {[
            { value: stats?.total_queries || 0, label: "Queries" },
            { value: `${stats?.cache_hit_rate || 0}%`, label: "Cache Hit" },
            { value: docs.length, label: "Documents" },
            { value: uptime, label: "Uptime" },
          ].map((s) => (
            <div
              key={s.label}
              className="bg-[#1a2235] border border-[#1e2d45] rounded-lg p-2.5 text-center"
            >
              <div className="text-lg font-mono text-cyan-400">{s.value}</div>
              <div className="text-[10px] text-slate-500">{s.label}</div>
            </div>
          ))}
        </div>
      </div>
    </aside>
  );
}
