"use client";

import { createContext, useContext, useState, useEffect, useCallback, ReactNode } from "react";
import { useRouter } from "next/navigation";
import { apiFetch } from "@/lib/api";
import { getToken, setToken, clearToken, setStoredUser } from "@/lib/auth";
import { UserResponse } from "@/lib/types";
import React from "react";

// ─── Demo account (no backend required) ─────────────────────────────────────
// Lets visitors explore the app without a working auth backend — useful for
// portfolio / showcase deployments.
const DEMO_TOKEN = "finbot-demo-session";
const DEMO_USER: UserResponse = {
  id: 0,
  username: "Demo User",
  email: "demo@finbot.app",
  is_active: true,
};

interface AuthContextType {
  user: UserResponse | null;
  token: string | null;
  isLoading: boolean;
  isDemo: boolean;
  login: (email: string, password: string) => Promise<string | null>;
  signup: (username: string, email: string, password: string) => Promise<string | null>;
  demoLogin: () => void;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<UserResponse | null>(null);
  const [token, setTokenState] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const router = useRouter();

  // Check auth on mount
  useEffect(() => {
    const savedToken = getToken();
    if (!savedToken) {
      setIsLoading(false);
      return;
    }

    // Restore a demo session without contacting the backend
    if (savedToken === DEMO_TOKEN) {
      setTokenState(DEMO_TOKEN);
      setUser(DEMO_USER);
      setIsLoading(false);
      return;
    }

    setTokenState(savedToken);

    apiFetch("/api/v1/auth/me")
      .then((res) => {
        if (res.ok) return res.json();
        throw new Error("Invalid token");
      })
      .then((data) => {
        setUser(data);
        setStoredUser(data);
      })
      .catch(() => {
        clearToken();
        setTokenState(null);
      })
      .finally(() => setIsLoading(false));
  }, []);

  const login = useCallback(async (email: string, password: string): Promise<string | null> => {
    const res = await apiFetch("/api/v1/auth/login", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    });
    const data = await res.json();
    if (!res.ok) return data.detail || "Login failed";
    setToken(data.access_token);
    setTokenState(data.access_token);
    setUser(data.user);
    setStoredUser(data.user);
    return null;
  }, []);

  const signup = useCallback(async (username: string, email: string, password: string): Promise<string | null> => {
    const res = await apiFetch("/api/v1/auth/signup", {
      method: "POST",
      body: JSON.stringify({ username, email, password }),
    });
    const data = await res.json();
    if (!res.ok) return data.detail || "Signup failed";
    setToken(data.access_token);
    setTokenState(data.access_token);
    setUser(data.user);
    setStoredUser(data.user);
    return null;
  }, []);

  // Instant demo access — no backend call, no credentials
  const demoLogin = useCallback(() => {
    setToken(DEMO_TOKEN);
    setTokenState(DEMO_TOKEN);
    setUser(DEMO_USER);
    setStoredUser(DEMO_USER);
  }, []);

  const logout = useCallback(() => {
    clearToken();
    setTokenState(null);
    setUser(null);
    router.push("/login");
  }, [router]);

  return React.createElement(
    AuthContext.Provider,
    {
      value: {
        user,
        token,
        isLoading,
        isDemo: token === DEMO_TOKEN,
        login,
        signup,
        demoLogin,
        logout,
      },
    },
    children
  );
}

export function useAuth(): AuthContextType {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
