"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/hooks/useAuth";
import Header from "@/components/Header";
import Sidebar from "@/components/Sidebar";
import ChatArea from "@/components/ChatArea";

export default function Home() {
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
      <div className="h-screen flex items-center justify-center">
        <div className="text-cyan-400 text-lg animate-pulse">Loading FinBot...</div>
      </div>
    );
  }

  return (
    <div className="h-screen flex flex-col relative z-[1]">
      <Header />
      <div className="flex flex-1 overflow-hidden">
        <Sidebar />
        <ChatArea />
      </div>
    </div>
  );
}
