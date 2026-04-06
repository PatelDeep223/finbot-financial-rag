"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { apiUrl } from "@/lib/api";
import { getToken } from "@/lib/auth";
import { Message } from "@/lib/types";
import MessageBubble from "./MessageBubble";
import { FiTrendingUp, FiSend } from "react-icons/fi";

const SAMPLE_QUESTIONS = [
  "What is the Q3 revenue?",
  "Compare Q3 vs Q2 revenue",
  "Summarize the risk factors",
  "What is the operating margin?",
  "How much free cash flow?",
];

export default function ChatArea() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const sessionId = useRef(`session_${Date.now()}`);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(scrollToBottom, [messages, scrollToBottom]);

  const sendQuery = useCallback(
    async (question: string) => {
      if (!question.trim() || isLoading) return;
      setIsLoading(true);
      setInput("");

      // Add user message
      const userMsg: Message = { id: `u_${Date.now()}`, role: "user", content: question };
      const botId = `b_${Date.now()}`;
      const botMsg: Message = { id: botId, role: "bot", content: "", isStreaming: true };

      setMessages((prev) => [...prev, userMsg, botMsg]);

      try {
        const token = getToken();
        const headers: Record<string, string> = { "Content-Type": "application/json" };
        if (token) headers["Authorization"] = `Bearer ${token}`;

        const res = await fetch(apiUrl("/api/v1/query/stream"), {
          method: "POST",
          headers,
          body: JSON.stringify({
            question,
            user_id: "user",
            session_id: sessionId.current,
          }),
        });

        if (!res.ok) {
          const err = await res.json().catch(() => ({ detail: "Request failed" }));
          setMessages((prev) =>
            prev.map((m) =>
              m.id === botId ? { ...m, content: `Error: ${err.detail}`, isStreaming: false } : m
            )
          );
          setIsLoading(false);
          return;
        }

        const reader = res.body!.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let fullText = "";
        let meta: Partial<Message> = {};

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          let eventType = "";
          for (const line of lines) {
            if (line.startsWith("event: ")) {
              eventType = line.slice(7).trim();
            } else if (line.startsWith("data: ")) {
              const data = line.slice(6);

              if (eventType === "token") {
                try {
                  const parsed = JSON.parse(data);
                  fullText += parsed.token;
                  setMessages((prev) =>
                    prev.map((m) => (m.id === botId ? { ...m, content: fullText } : m))
                  );
                } catch {}
              } else if (eventType === "meta") {
                try {
                  const parsed = JSON.parse(data);
                  meta = {
                    intent: parsed.intent,
                    query_rewritten: parsed.query_rewritten,
                    from_cache: parsed.from_cache,
                  };
                  setMessages((prev) =>
                    prev.map((m) => (m.id === botId ? { ...m, ...meta } : m))
                  );
                } catch {}
              } else if (eventType === "sources") {
                try {
                  const parsed = JSON.parse(data);
                  setMessages((prev) =>
                    prev.map((m) =>
                      m.id === botId
                        ? {
                            ...m,
                            sources: parsed.sources,
                            confident: parsed.confident,
                            confidence_score: parsed.confidence_score,
                            response_time_ms: parsed.response_time_ms,
                            isStreaming: false,
                          }
                        : m
                    )
                  );
                } catch {}
              } else if (eventType === "done") {
                setMessages((prev) =>
                  prev.map((m) => (m.id === botId ? { ...m, isStreaming: false } : m))
                );
              }
            }
          }
        }
      } catch {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === botId
              ? { ...m, content: "Cannot connect to FinBot API. Is the backend running?", isStreaming: false }
              : m
          )
        );
      }

      setIsLoading(false);
      textareaRef.current?.focus();
    },
    [isLoading]
  );

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendQuery(input);
    }
  };

  return (
    <div className="flex flex-col flex-1 overflow-hidden">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-6 py-6 space-y-5 scrollbar-thin">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center gap-4">
            <div className="animate-bounce-slow">
              <FiTrendingUp className="w-12 h-12 text-cyan-400" />
            </div>
            <h2 className="text-2xl font-serif bg-gradient-to-r from-white to-cyan-400 bg-clip-text text-transparent">
              Financial Intelligence Assistant
            </h2>
            <p className="text-slate-400 text-sm max-w-md leading-relaxed">
              Upload financial documents and ask questions. Powered by hybrid retrieval, cross-encoder reranking, and anti-hallucination.
            </p>
            <div className="flex flex-wrap gap-2 justify-center mt-2">
              {SAMPLE_QUESTIONS.map((q) => (
                <button
                  key={q}
                  onClick={() => sendQuery(q)}
                  className="px-3 py-1.5 rounded-full bg-[#1a2235] border border-[#1e2d45] text-xs text-slate-400 hover:border-cyan-400/50 hover:text-cyan-400 transition"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((msg) => <MessageBubble key={msg.id} msg={msg} />)
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="px-6 py-4 bg-[#111827]/95 border-t border-[#1e2d45] backdrop-blur-xl">
        <div className="flex gap-3 items-end">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => {
              setInput(e.target.value);
              e.target.style.height = "auto";
              e.target.style.height = Math.min(e.target.scrollHeight, 120) + "px";
            }}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question about your financial documents..."
            rows={1}
            className="flex-1 px-4 py-3 rounded-xl bg-[#1a2235] border border-[#1e2d45] text-white text-sm placeholder-slate-500 focus:border-cyan-500/40 focus:outline-none resize-none min-h-[46px] max-h-[120px] leading-relaxed"
          />
          <button
            onClick={() => sendQuery(input)}
            disabled={isLoading || !input.trim()}
            className="w-11 h-11 rounded-xl bg-gradient-to-r from-cyan-500 to-purple-600 flex items-center justify-center text-white text-lg hover:opacity-90 disabled:opacity-40 transition shrink-0"
          >
            <FiSend className="w-5 h-5" />
          </button>
        </div>
        <div className="text-[11px] text-slate-600 text-center mt-2">
          Enter to send · Shift+Enter for new line · Hybrid RAG + Reranker + Anti-Hallucination
        </div>
      </div>
    </div>
  );
}
