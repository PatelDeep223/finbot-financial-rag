"use client";

import { useState } from "react";
import Link from "next/link";
import { FiTrendingUp } from "react-icons/fi";

interface AuthFormProps {
  mode: "login" | "signup";
  onSubmit: (data: { username?: string; email: string; password: string }) => Promise<string | null>;
}

export default function AuthForm({ mode, onSubmit }: AuthFormProps) {
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    const err = await onSubmit({
      ...(mode === "signup" ? { username } : {}),
      email,
      password,
    });

    if (err) setError(err);
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-[#0a0f1e] flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        {/* Logo */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-14 h-14 rounded-xl bg-gradient-to-br from-cyan-400 to-purple-600 text-white mb-4">
            <FiTrendingUp className="w-7 h-7" />
          </div>
          <h1 className="text-3xl font-serif bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
            FinBot
          </h1>
          <p className="text-slate-400 text-sm mt-1">Financial Intelligence RAG</p>
        </div>

        {/* Form Card */}
        <div className="bg-[#111827] border border-[#1e2d45] rounded-2xl p-8">
          <h2 className="text-xl font-semibold text-white mb-6">
            {mode === "login" ? "Welcome back" : "Create your account"}
          </h2>

          {error && (
            <div className="mb-4 p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 text-sm">
              {error}
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            {mode === "signup" && (
              <div>
                <label className="block text-sm text-slate-400 mb-1.5">Username</label>
                <input
                  type="text"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  required
                  minLength={3}
                  className="w-full px-4 py-3 rounded-xl bg-[#1a2235] border border-[#1e2d45] text-white placeholder-slate-500 focus:border-cyan-500/40 focus:outline-none transition"
                  placeholder="deepak"
                />
              </div>
            )}

            <div>
              <label className="block text-sm text-slate-400 mb-1.5">Email</label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                className="w-full px-4 py-3 rounded-xl bg-[#1a2235] border border-[#1e2d45] text-white placeholder-slate-500 focus:border-cyan-500/40 focus:outline-none transition"
                placeholder="you@example.com"
              />
            </div>

            <div>
              <label className="block text-sm text-slate-400 mb-1.5">Password</label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                minLength={6}
                className="w-full px-4 py-3 rounded-xl bg-[#1a2235] border border-[#1e2d45] text-white placeholder-slate-500 focus:border-cyan-500/40 focus:outline-none transition"
                placeholder="••••••••"
              />
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full py-3 rounded-xl bg-gradient-to-r from-cyan-500 to-purple-600 text-white font-medium hover:opacity-90 disabled:opacity-50 transition"
            >
              {loading ? "..." : mode === "login" ? "Sign In" : "Create Account"}
            </button>
          </form>

          <p className="text-center text-slate-500 text-sm mt-6">
            {mode === "login" ? (
              <>
                Don&apos;t have an account?{" "}
                <Link href="/signup" className="text-cyan-400 hover:text-cyan-300">
                  Sign up
                </Link>
              </>
            ) : (
              <>
                Already have an account?{" "}
                <Link href="/login" className="text-cyan-400 hover:text-cyan-300">
                  Sign in
                </Link>
              </>
            )}
          </p>
        </div>
      </div>
    </div>
  );
}
