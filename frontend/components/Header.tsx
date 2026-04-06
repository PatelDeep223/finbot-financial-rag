"use client";

import { useAuth } from "@/hooks/useAuth";
import { useStats } from "@/hooks/useStats";
import { FiTrendingUp, FiZap, FiTarget, FiLogOut, FiUser } from "react-icons/fi";

export default function Header() {
  const { user, logout } = useAuth();
  const stats = useStats();

  return (
    <header className="flex items-center justify-between px-7 py-4 bg-[#111827]/95 border-b border-[#1e2d45] backdrop-blur-xl z-10">
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-400 to-purple-600 flex items-center justify-center text-white">
          <FiTrendingUp className="w-5 h-5" />
        </div>
        <div>
          <div className="text-xl font-serif bg-gradient-to-r from-cyan-400 to-purple-300 bg-clip-text text-transparent">
            FinBot
          </div>
          <div className="text-[11px] text-slate-500 tracking-widest uppercase">
            Financial Intelligence RAG
          </div>
        </div>
      </div>

      <div className="flex items-center gap-4">
        <div className="flex items-center gap-1.5 bg-[#1a2235] border border-[#1e2d45] rounded-full px-3 py-1.5 text-xs text-slate-400">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
          {stats?.vectorstore_loaded ? "FAISS" : "System"}{" "}
          {stats?.bm25_loaded ? "+ BM25" : ""} Ready
        </div>
        <div className="hidden sm:flex items-center gap-1.5 bg-[#1a2235] border border-[#1e2d45] rounded-full px-3 py-1.5 text-xs text-slate-400">
          <FiZap className="w-3 h-3" /> {stats?.total_queries || 0} queries
        </div>
        <div className="hidden sm:flex items-center gap-1.5 bg-[#1a2235] border border-[#1e2d45] rounded-full px-3 py-1.5 text-xs text-slate-400">
          <FiTarget className="w-3 h-3" /> {stats?.cache_hit_rate || 0}% cache
        </div>

        {user && (
          <div className="flex items-center gap-3 ml-2">
            <span className="text-sm text-slate-400 flex items-center gap-1">
              <FiUser className="w-3.5 h-3.5" /> {user.username}
            </span>
            <button
              onClick={logout}
              className="flex items-center gap-1 text-xs text-slate-500 hover:text-red-400 transition"
            >
              <FiLogOut className="w-3 h-3" /> Logout
            </button>
          </div>
        )}
      </div>
    </header>
  );
}
