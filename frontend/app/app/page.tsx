"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/hooks/useAuth";
import Header from "@/components/Header";
import Sidebar from "@/components/Sidebar";
import ChatArea from "@/components/ChatArea";

export default function ChatApp() {
  const { user, isLoading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!isLoading && !user) {
      fetch(`${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/health`)
        .then((r) => r.json())
        .then((data) => {
          if (data.auth_enabled) router.push("/login");
        })
        .catch(() => {});
    }
  }, [user, isLoading, router]);

  if (isLoading) {
    return (
      <div className="h-screen flex items-center justify-center bg-white">
        <div className="text-emerald-600 text-lg animate-pulse">Loading FinBot...</div>
      </div>
    );
  }

  return (
    <div className="h-screen flex flex-col bg-white">
      <Header />
      <div className="flex flex-1 overflow-hidden">
        <Sidebar />
        <ChatArea />
      </div>
    </div>
  );
}
