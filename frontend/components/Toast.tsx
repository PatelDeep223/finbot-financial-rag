"use client";

import { useState, useCallback, createContext, useContext, ReactNode } from "react";
import React from "react";

interface Toast {
  id: number;
  message: string;
  type: "success" | "error";
}

interface ToastContextType {
  showToast: (message: string, type?: "success" | "error") => void;
}

const ToastContext = createContext<ToastContextType>({ showToast: () => {} });

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const showToast = useCallback((message: string, type: "success" | "error" = "success") => {
    const id = Date.now();
    setToasts((prev) => [...prev, { id, message, type }]);
    setTimeout(() => setToasts((prev) => prev.filter((t) => t.id !== id)), 3500);
  }, []);

  return React.createElement(
    ToastContext.Provider,
    { value: { showToast } },
    children,
    React.createElement(
      "div",
      { className: "fixed bottom-6 right-6 z-50 flex flex-col gap-2" },
      toasts.map((t) =>
        React.createElement(
          "div",
          {
            key: t.id,
            className: `px-4 py-3 rounded-xl border text-sm animate-slide-in ${
              t.type === "success"
                ? "bg-[#1a2235] border-emerald-500/30 text-emerald-400"
                : "bg-[#1a2235] border-red-500/30 text-red-400"
            }`,
          },
          t.message
        )
      )
    )
  );
}

export function useToast() {
  return useContext(ToastContext);
}
