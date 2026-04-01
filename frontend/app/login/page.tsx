"use client";

import { useRouter } from "next/navigation";
import { useAuth } from "@/hooks/useAuth";
import AuthForm from "@/components/AuthForm";
import { useEffect } from "react";

export default function LoginPage() {
  const { login, user } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (user) router.push("/");
  }, [user, router]);

  const handleLogin = async (data: { email: string; password: string }) => {
    const err = await login(data.email, data.password);
    if (!err) router.push("/");
    return err;
  };

  return <AuthForm mode="login" onSubmit={handleLogin} />;
}
