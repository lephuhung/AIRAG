import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { toast } from "sonner";
import { useAuthStore } from "@/stores/authStore";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent } from "@/components/ui/card";
import { LogIn } from "lucide-react";
import { useTranslation } from "@/hooks/useTranslation";

export function LoginPage() {
  const navigate = useNavigate();
  const login = useAuthStore((s) => s.login);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const { t } = useTranslation();

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email || !password) return;

    setLoading(true);
    try {
      await login(email, password);
      toast.success("Logged in successfully");
      navigate("/");
    } catch (err: any) {
      toast.error(err.message || "Login failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-background relative overflow-hidden px-4">
      {/* Background decoration */}
      <div className="absolute top-[-20%] left-[-10%] w-[50%] h-[50%] rounded-full bg-primary/20 blur-[120px] pointer-events-none animate-pulse duration-1000" />
      <div className="absolute bottom-[-20%] right-[-10%] w-[50%] h-[50%] rounded-full bg-blue-500/20 blur-[120px] pointer-events-none animate-pulse duration-1000 delay-500" />
      
      <Card className="w-full max-w-sm shadow-2xl border-primary/10 relative z-10 bg-card/80 backdrop-blur-xl">
        <CardContent className="pt-8 pb-6">
          <div className="flex flex-col items-center mb-6">
            <div className="w-16 h-16 flex items-center justify-center mb-3">
              <img src="/logo.png" alt="AIRAG Logo" className="w-full h-full object-contain" />
            </div>
            <h1 className="text-xl font-bold">{t("app.name")}</h1>
            <p className="text-sm text-muted-foreground mt-1">{t("auth.login_title")}</p>
          </div>

          <form onSubmit={handleLogin} className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-1.5">{t("auth.email")}</label>
              <Input
                type="email"
                placeholder="you@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                autoFocus
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-1.5">{t("auth.password")}</label>
              <Input
                type="password"
                placeholder="Enter your password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
            </div>
            <Button type="submit" className="w-full" disabled={loading}>
              <LogIn className="w-4 h-4 mr-2" />
              {loading ? t("auth.signing_in") : t("auth.login")}
            </Button>
          </form>

          <div className="mt-4 text-center text-sm text-muted-foreground">
            {t("auth.no_account")}{" "}
            <Link to="/register" className="text-primary hover:underline font-medium">
              {t("auth.register")}
            </Link>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
