"use client";

import { useRouter } from "next/navigation";
import { useAuth } from "@/hooks/useAuth";

interface DemoLaunchButtonProps {
  className?: string;
  children: React.ReactNode;
}

/**
 * Landing-page CTA that drops the visitor straight into the app with a
 * demo session — no login, no backend auth required.
 */
export default function DemoLaunchButton({ className, children }: DemoLaunchButtonProps) {
  const { demoLogin } = useAuth();
  const router = useRouter();

  const handleClick = () => {
    demoLogin();
    router.push("/app");
  };

  return (
    <button type="button" onClick={handleClick} className={className}>
      {children}
    </button>
  );
}
