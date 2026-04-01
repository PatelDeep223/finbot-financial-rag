"use client";

import { useState, useEffect } from "react";
import { apiFetch } from "@/lib/api";
import { SystemStats } from "@/lib/types";

export function useStats(intervalMs = 10000) {
  const [stats, setStats] = useState<SystemStats | null>(null);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const res = await apiFetch("/api/v1/stats");
        if (res.ok) setStats(await res.json());
      } catch {}
    };

    fetchStats();
    const id = setInterval(fetchStats, intervalMs);
    return () => clearInterval(id);
  }, [intervalMs]);

  return stats;
}
