"use client";

import { Message } from "@/lib/types";

const INTENT_ICONS: Record<string, string> = {
  factual: "🔍",
  comparison: "⚖️",
  summary: "📋",
  risk: "⚠️",
  off_topic: "🚫",
};

function formatAnswer(text: string): string {
  return text
    .replace(/^(Answer:)\s*/im, '<span class="text-cyan-400 text-xs font-semibold uppercase tracking-wider block mt-0 mb-0.5">Answer</span>')
    .replace(/\n(Key Insight:)\s*/im, '<span class="text-cyan-400 text-xs font-semibold uppercase tracking-wider block mt-3 mb-0.5">Key Insight</span><span class="text-amber-400 italic">')
    .replace(/\n(Source:)\s*/im, '</span><span class="text-cyan-400 text-xs font-semibold uppercase tracking-wider block mt-3 mb-0.5">Source</span>')
    .replace(/\n(Summary:)\s*/im, '<span class="text-cyan-400 text-xs font-semibold uppercase tracking-wider block mt-3 mb-0.5">Summary</span>');
}

export default function MessageBubble({ msg }: { msg: Message }) {
  const isUser = msg.role === "user";

  return (
    <div className={`flex gap-3 animate-slide-in ${isUser ? "flex-row-reverse" : ""}`}>
      {/* Avatar */}
      <div
        className={`w-8 h-8 rounded-full flex items-center justify-center text-sm shrink-0 mt-0.5 ${
          isUser
            ? "bg-gradient-to-br from-purple-600 to-cyan-400"
            : "bg-gradient-to-br from-amber-500 to-orange-500"
        }`}
      >
        {isUser ? "👤" : "🤖"}
      </div>

      {/* Bubble */}
      <div
        className={`max-w-[72%] border rounded-2xl px-4 py-3 ${
          isUser
            ? "bg-gradient-to-br from-cyan-500/10 to-purple-600/10 border-cyan-500/20 rounded-tr-sm"
            : "bg-[#1a2235] border-[#1e2d45] rounded-tl-sm"
        }`}
      >
        {/* Rewrite notice */}
        {msg.query_rewritten && (
          <div className="text-[11px] text-slate-500 italic mb-2 flex items-center gap-1">
            🔄 Query optimized: &quot;{msg.query_rewritten}&quot;
          </div>
        )}

        {/* Message text */}
        <div
          className="text-sm leading-relaxed text-slate-200 whitespace-pre-wrap"
          dangerouslySetInnerHTML={{ __html: isUser ? msg.content : formatAnswer(msg.content) }}
        />

        {/* Streaming cursor */}
        {msg.isStreaming && (
          <span className="inline-block w-2 h-4 bg-cyan-400 animate-pulse ml-0.5" />
        )}

        {/* Sources */}
        {msg.sources && msg.sources.length > 0 && (
          <div className="mt-3">
            <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-1.5">
              Sources
            </div>
            {msg.sources.map((s, i) => (
              <div
                key={i}
                className="bg-black/20 border border-[#1e2d45] rounded-lg p-2 mb-1 text-xs"
              >
                <div className="text-cyan-400 font-medium mb-0.5">
                  📄 {s.source}
                  {s.page ? ` · Page ${s.page}` : ""}
                  {s.score ? ` · Score ${s.score.toFixed(2)}` : ""}
                </div>
                <div className="text-slate-400 text-[11px] leading-snug">{s.content}</div>
              </div>
            ))}
          </div>
        )}

        {/* Confidence bar */}
        {msg.confidence_score !== undefined && !msg.isStreaming && msg.role === "bot" && (
          <div className="flex items-center gap-2 mt-3 pt-3 border-t border-[#1e2d45] flex-wrap">
            <span
              className={`px-2 py-0.5 rounded-full text-[11px] font-medium border ${
                msg.confident
                  ? "bg-emerald-500/15 text-emerald-400 border-emerald-500/20"
                  : "bg-red-500/15 text-red-400 border-red-500/20"
              }`}
            >
              {msg.confident ? "✅" : "⚠️"} {Math.round(msg.confidence_score * 100)}% confident
            </span>

            {msg.intent && (
              <span className="px-2 py-0.5 rounded-full text-[11px] font-medium bg-purple-500/15 text-purple-300 border border-purple-500/20">
                {INTENT_ICONS[msg.intent] || "🔍"} {msg.intent}
              </span>
            )}

            {msg.from_cache && (
              <span className="px-2 py-0.5 rounded-full text-[11px] font-medium bg-amber-500/15 text-amber-400 border border-amber-500/20">
                ⚡ cached
              </span>
            )}

            {msg.response_time_ms !== undefined && (
              <span className="text-[11px] text-slate-500 font-mono ml-auto">
                {msg.response_time_ms}ms
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
