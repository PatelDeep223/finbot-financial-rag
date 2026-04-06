"use client";

import { Message } from "@/lib/types";
import ReactMarkdown, { Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  FiUser,
  FiCpu,
  FiRefreshCw,
  FiCheckCircle,
  FiAlertTriangle,
  FiSearch,
  FiBarChart2,
  FiFileText,
  FiSlash,
  FiZap,
  FiFile,
} from "react-icons/fi";
import { ComponentPropsWithoutRef } from "react";

// ─── Intent icon mapping (react-icons) ──────────────────────────────────────

const INTENT_ICONS: Record<string, React.ReactNode> = {
  factual: <FiSearch className="w-3 h-3" />,
  comparison: <FiBarChart2 className="w-3 h-3" />,
  summary: <FiFileText className="w-3 h-3" />,
  risk: <FiAlertTriangle className="w-3 h-3" />,
  off_topic: <FiSlash className="w-3 h-3" />,
};

// ─── Markdown component overrides (dark theme) ─────────────────────────────

const mdComponents: Components = {
  table: ({ children, ...props }: ComponentPropsWithoutRef<"table">) => (
    <div className="overflow-x-auto my-3 rounded-lg border border-[#1e2d45]">
      <table className="w-full text-sm border-collapse" {...props}>
        {children}
      </table>
    </div>
  ),
  thead: ({ children, ...props }: ComponentPropsWithoutRef<"thead">) => (
    <thead className="bg-cyan-500/8 border-b border-cyan-500/20" {...props}>
      {children}
    </thead>
  ),
  th: ({ children, ...props }: ComponentPropsWithoutRef<"th">) => (
    <th
      className="text-left px-3 py-2 text-cyan-400 font-semibold text-xs uppercase tracking-wider"
      {...props}
    >
      {children}
    </th>
  ),
  td: ({ children, ...props }: ComponentPropsWithoutRef<"td">) => (
    <td className="px-3 py-2 border-b border-[#1e2d45] text-slate-300" {...props}>
      {children}
    </td>
  ),
  tr: ({ children, ...props }: ComponentPropsWithoutRef<"tr">) => (
    <tr className="even:bg-[#1a2235]/50" {...props}>
      {children}
    </tr>
  ),
  strong: ({ children, ...props }: ComponentPropsWithoutRef<"strong">) => (
    <strong className="text-cyan-300 font-semibold" {...props}>
      {children}
    </strong>
  ),
  ul: ({ children, ...props }: ComponentPropsWithoutRef<"ul">) => (
    <ul className="list-disc list-inside space-y-1 my-2 text-slate-300" {...props}>
      {children}
    </ul>
  ),
  ol: ({ children, ...props }: ComponentPropsWithoutRef<"ol">) => (
    <ol className="list-decimal list-inside space-y-1 my-2 text-slate-300" {...props}>
      {children}
    </ol>
  ),
  li: ({ children, ...props }: ComponentPropsWithoutRef<"li">) => (
    <li className="leading-relaxed" {...props}>
      {children}
    </li>
  ),
  p: ({ children, ...props }: ComponentPropsWithoutRef<"p">) => (
    <p className="mb-2 last:mb-0 leading-relaxed" {...props}>
      {children}
    </p>
  ),
  code: ({ children, className, ...props }: ComponentPropsWithoutRef<"code">) => {
    const isBlock = className?.includes("language-");
    if (isBlock) {
      return (
        <code className="block bg-[#0d1321] text-cyan-400 text-xs font-mono p-3 rounded-lg overflow-x-auto" {...props}>
          {children}
        </code>
      );
    }
    return (
      <code className="bg-[#1a2235] text-cyan-400 px-1.5 py-0.5 rounded text-xs font-mono" {...props}>
        {children}
      </code>
    );
  },
  pre: ({ children, ...props }: ComponentPropsWithoutRef<"pre">) => (
    <pre className="bg-[#0d1321] border border-[#1e2d45] rounded-lg my-2 overflow-x-auto" {...props}>
      {children}
    </pre>
  ),
  hr: (props: ComponentPropsWithoutRef<"hr">) => (
    <hr className="my-4 border-[#1e2d45]" {...props} />
  ),
  h3: ({ children, ...props }: ComponentPropsWithoutRef<"h3">) => (
    <h3 className="text-base font-semibold text-white mt-3 mb-1" {...props}>
      {children}
    </h3>
  ),
  h4: ({ children, ...props }: ComponentPropsWithoutRef<"h4">) => (
    <h4 className="text-sm font-semibold text-slate-200 mt-2 mb-1" {...props}>
      {children}
    </h4>
  ),
};

const remarkPlugins = [remarkGfm];

// ─── Section parser ─────────────────────────────────────────────────────────

interface Section {
  type: "answer" | "insight" | "source" | "summary" | "body";
  content: string;
}

function parseSections(text: string): Section[] {
  const sections: Section[] = [];
  const headerRegex = /^(Answer:|Key Insight:|Source:|Summary:)\s*/gim;
  let lastIndex = 0;
  let lastType: Section["type"] = "body";
  let match;

  while ((match = headerRegex.exec(text)) !== null) {
    // Push content before this header
    if (match.index > lastIndex) {
      const content = text.slice(lastIndex, match.index).trim();
      if (content) sections.push({ type: lastType, content });
    }

    const header = match[1].toLowerCase();
    if (header.startsWith("answer")) lastType = "answer";
    else if (header.startsWith("key")) lastType = "insight";
    else if (header.startsWith("source")) lastType = "source";
    else if (header.startsWith("summary")) lastType = "summary";

    lastIndex = match.index + match[0].length;
  }

  // Push remaining content
  const remaining = text.slice(lastIndex).trim();
  if (remaining) sections.push({ type: lastType, content: remaining });

  // If no sections found, return entire text as body
  if (sections.length === 0 && text.trim()) {
    sections.push({ type: "body", content: text.trim() });
  }

  return sections;
}

const SECTION_LABELS: Record<string, string> = {
  answer: "Answer",
  insight: "Key Insight",
  source: "Source",
  summary: "Summary",
};

// ─── Component ──────────────────────────────────────────────────────────────

export default function MessageBubble({ msg }: { msg: Message }) {
  const isUser = msg.role === "user";
  const sections = isUser ? [] : parseSections(msg.content);

  return (
    <div className={`flex gap-3 animate-slide-in ${isUser ? "flex-row-reverse" : ""}`}>
      {/* Avatar */}
      <div
        className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 mt-0.5 ${
          isUser
            ? "bg-gradient-to-br from-purple-600 to-cyan-400 text-white"
            : "bg-gradient-to-br from-amber-500 to-orange-500 text-white"
        }`}
      >
        {isUser ? <FiUser className="w-4 h-4" /> : <FiCpu className="w-4 h-4" />}
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
            <FiRefreshCw className="w-3 h-3" />
            Query optimized: &quot;{msg.query_rewritten}&quot;
          </div>
        )}

        {/* Message text */}
        {isUser ? (
          <div className="text-sm leading-relaxed text-slate-200 whitespace-pre-wrap">
            {msg.content}
          </div>
        ) : (
          <div className="text-sm leading-relaxed text-slate-200 markdown-content">
            {sections.map((section, i) => (
              <div key={i}>
                {/* Section header */}
                {section.type !== "body" && SECTION_LABELS[section.type] && (
                  <div className="text-cyan-400 text-xs font-semibold uppercase tracking-wider mt-3 first:mt-0 mb-1">
                    {SECTION_LABELS[section.type]}
                  </div>
                )}

                {/* Section body */}
                {section.type === "insight" ? (
                  <div className="bg-amber-500/5 border border-amber-500/15 rounded-lg px-3 py-2 text-amber-300 italic text-sm">
                    <ReactMarkdown remarkPlugins={remarkPlugins} components={mdComponents}>
                      {section.content}
                    </ReactMarkdown>
                  </div>
                ) : (
                  <ReactMarkdown remarkPlugins={remarkPlugins} components={mdComponents}>
                    {section.content}
                  </ReactMarkdown>
                )}
              </div>
            ))}

            {/* If no sections parsed yet (early streaming), show raw text */}
            {sections.length === 0 && msg.content && (
              <ReactMarkdown remarkPlugins={remarkPlugins} components={mdComponents}>
                {msg.content}
              </ReactMarkdown>
            )}
          </div>
        )}

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
                <div className="text-cyan-400 font-medium mb-0.5 flex items-center gap-1">
                  <FiFile className="w-3 h-3" />
                  {s.source}
                  {s.page ? ` \u00B7 Page ${s.page}` : ""}
                  {s.score ? ` \u00B7 Score ${s.score.toFixed(2)}` : ""}
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
              className={`flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium border ${
                msg.confident
                  ? "bg-emerald-500/15 text-emerald-400 border-emerald-500/20"
                  : "bg-red-500/15 text-red-400 border-red-500/20"
              }`}
            >
              {msg.confident ? (
                <FiCheckCircle className="w-3 h-3" />
              ) : (
                <FiAlertTriangle className="w-3 h-3" />
              )}
              {Math.round(msg.confidence_score * 100)}% confident
            </span>

            {msg.intent && (
              <span className="flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium bg-purple-500/15 text-purple-300 border border-purple-500/20">
                {INTENT_ICONS[msg.intent] || <FiSearch className="w-3 h-3" />}
                {msg.intent}
              </span>
            )}

            {msg.from_cache && (
              <span className="flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium bg-amber-500/15 text-amber-400 border border-amber-500/20">
                <FiZap className="w-3 h-3" />
                cached
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
