"use client";

import { useRouter } from "next/navigation";
import { useAuth } from "@/hooks/useAuth";
import AuthForm from "@/components/AuthForm";
import { useEffect } from "react";

export default function SignupPage() {
  const { signup, user } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (user) router.push("/");
  }, [user, router]);

  const handleSignup = async (data: { username?: string; email: string; password: string }) => {
    const err = await signup(data.username || "", data.email, data.password);
    if (!err) router.push("/");
    return err;
  };

  return <AuthForm mode="signup" onSubmit={handleSignup} />;
}
