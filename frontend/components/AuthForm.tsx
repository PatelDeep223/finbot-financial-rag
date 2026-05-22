"use client";

import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { FiTrendingUp, FiArrowLeft, FiArrowRight } from "react-icons/fi";
import { useAuth } from "@/hooks/useAuth";

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
  const { demoLogin } = useAuth();
  const router = useRouter();

  const handleDemo = () => {
    demoLogin();
    router.push("/app");
  };

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

  const inputClass =
    "w-full px-4 py-3 rounded-xl bg-white border border-slate-200 text-slate-900 placeholder-slate-400 focus:border-emerald-500 focus:outline-none transition";

  return (
    <div className="min-h-screen bg-slate-50 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        {/* Logo */}
        <div className="text-center mb-8">
          <Link href="/" className="inline-flex items-center justify-center w-14 h-14 rounded-xl bg-emerald-600 text-white mb-4">
            <FiTrendingUp className="w-7 h-7" />
          </Link>
          <h1 className="text-3xl font-serif text-slate-900">FinBot</h1>
          <p className="text-slate-500 text-sm mt-1">Financial Intelligence RAG</p>
        </div>

        {/* Form Card */}
        <div className="bg-white border border-slate-200 rounded-2xl p-8 shadow-sm">
          <h2 className="text-xl font-semibold text-slate-900 mb-6">
            {mode === "login" ? "Welcome back" : "Create your account"}
          </h2>

          {error && (
            <div className="mb-4 p-3 rounded-lg bg-red-50 border border-red-200 text-red-600 text-sm">
              {error}
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            {mode === "signup" && (
              <div>
                <label className="block text-sm text-slate-600 mb-1.5">Username</label>
                <input
                  type="text"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  required
                  minLength={3}
                  className={inputClass}
                  placeholder="username"
                />
              </div>
            )}

            <div>
              <label className="block text-sm text-slate-600 mb-1.5">Email</label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                className={inputClass}
                placeholder="you@example.com"
              />
            </div>

            <div>
              <label className="block text-sm text-slate-600 mb-1.5">Password</label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                minLength={6}
                className={inputClass}
                placeholder="••••••••"
              />
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full py-3 rounded-xl bg-emerald-600 text-white font-medium hover:bg-emerald-700 disabled:opacity-50 transition"
            >
              {loading ? "..." : mode === "login" ? "Sign In" : "Create Account"}
            </button>
          </form>

          {/* Demo bypass */}
          <div className="flex items-center gap-3 my-5">
            <div className="h-px bg-slate-200 flex-1" />
            <span className="text-xs text-slate-400">or</span>
            <div className="h-px bg-slate-200 flex-1" />
          </div>

          <button
            type="button"
            onClick={handleDemo}
            className="w-full py-3 rounded-xl border border-emerald-300 bg-emerald-50 text-emerald-700 font-medium hover:bg-emerald-100 transition flex items-center justify-center gap-2"
          >
            Continue with demo account <FiArrowRight className="w-4 h-4" />
          </button>
          <p className="text-center text-xs text-slate-400 mt-2">
            No sign-up needed — explore FinBot instantly
          </p>

          <p className="text-center text-slate-500 text-sm mt-6">
            {mode === "login" ? (
              <>
                Don&apos;t have an account?{" "}
                <Link href="/signup" className="text-emerald-600 hover:text-emerald-700 font-medium">
                  Sign up
                </Link>
              </>
            ) : (
              <>
                Already have an account?{" "}
                <Link href="/login" className="text-emerald-600 hover:text-emerald-700 font-medium">
                  Sign in
                </Link>
              </>
            )}
          </p>
        </div>

        <div className="text-center mt-6">
          <Link
            href="/"
            className="inline-flex items-center gap-1.5 text-sm text-slate-400 hover:text-slate-700 transition"
          >
            <FiArrowLeft className="w-4 h-4" /> Back to home
          </Link>
        </div>
      </div>
    </div>
  );
}
