"use client";

import Link from "next/link";
import { useAuth } from "@/hooks/useAuth";
import { useStats } from "@/hooks/useStats";
import { FiTrendingUp, FiZap, FiTarget, FiLogOut, FiUser } from "react-icons/fi";

export default function Header() {
  const { user, logout } = useAuth();
  const stats = useStats();

  return (
    <header className="flex items-center justify-between px-7 py-3.5 bg-white border-b border-slate-200 z-10">
      <Link href="/" className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-xl bg-emerald-600 flex items-center justify-center text-white">
          <FiTrendingUp className="w-5 h-5" />
        </div>
        <div>
          <div className="text-xl font-serif text-slate-900">FinBot</div>
          <div className="text-[11px] text-slate-400 tracking-widest uppercase">
            Financial Intelligence RAG
          </div>
        </div>
      </Link>

      <div className="flex items-center gap-3">
        <div className="flex items-center gap-1.5 bg-slate-50 border border-slate-200 rounded-full px-3 py-1.5 text-xs text-slate-500">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
          {stats?.vectorstore_loaded ? "FAISS" : "System"}{" "}
          {stats?.bm25_loaded ? "+ BM25" : ""} Ready
        </div>
        <div className="hidden sm:flex items-center gap-1.5 bg-slate-50 border border-slate-200 rounded-full px-3 py-1.5 text-xs text-slate-500">
          <FiZap className="w-3 h-3" /> {stats?.total_queries || 0} queries
        </div>
        <div className="hidden sm:flex items-center gap-1.5 bg-slate-50 border border-slate-200 rounded-full px-3 py-1.5 text-xs text-slate-500">
          <FiTarget className="w-3 h-3" /> {stats?.cache_hit_rate || 0}% cache
        </div>

        {user && (
          <div className="flex items-center gap-3 ml-2">
            <span className="text-sm text-slate-600 flex items-center gap-1">
              <FiUser className="w-3.5 h-3.5" /> {user.username}
            </span>
            <button
              onClick={logout}
              className="flex items-center gap-1 text-xs text-slate-400 hover:text-red-500 transition"
            >
              <FiLogOut className="w-3 h-3" /> Logout
            </button>
          </div>
        )}
      </div>
    </header>
  );
}
