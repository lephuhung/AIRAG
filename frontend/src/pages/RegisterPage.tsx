import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { toast } from "sonner";
import { useAuthStore } from "@/stores/authStore";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent } from "@/components/ui/card";
import { Database, UserPlus } from "lucide-react";

export function RegisterPage() {
  const navigate = useNavigate();
  const register = useAuthStore((s) => s.register);
  const [form, setForm] = useState({
    email: "",
    password: "",
    full_name: "",
    tenant_slug: "",
  });
  const [loading, setLoading] = useState(false);

  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!form.email || !form.password || !form.full_name) return;

    setLoading(true);
    try {
      const result = await register({
        email: form.email,
        password: form.password,
        full_name: form.full_name,
        tenant_slug: form.tenant_slug || undefined,
      });
      toast.success(result.message);
      navigate("/login");
    } catch (err: any) {
      toast.error(err.message || "Registration failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-background px-4">
      <Card className="w-full max-w-sm shadow-xl">
        <CardContent className="pt-8 pb-6">
          <div className="flex flex-col items-center mb-6">
            <div className="w-14 h-14 rounded-2xl bg-primary/10 flex items-center justify-center mb-3">
              <Database className="w-7 h-7 text-primary" />
            </div>
            <h1 className="text-xl font-bold">NexusRAG</h1>
            <p className="text-sm text-muted-foreground mt-1">Create a new account</p>
          </div>

          <form onSubmit={handleRegister} className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-1.5">Full Name</label>
              <Input
                placeholder="Your full name"
                value={form.full_name}
                onChange={(e) => setForm((s) => ({ ...s, full_name: e.target.value }))}
                autoFocus
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-1.5">Email</label>
              <Input
                type="email"
                placeholder="you@example.com"
                value={form.email}
                onChange={(e) => setForm((s) => ({ ...s, email: e.target.value }))}
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-1.5">Password</label>
              <Input
                type="password"
                placeholder="Min 6 characters"
                value={form.password}
                onChange={(e) => setForm((s) => ({ ...s, password: e.target.value }))}
                minLength={6}
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-1.5">
                Tenant Code <span className="text-muted-foreground font-normal">(optional)</span>
              </label>
              <Input
                placeholder="e.g. my-company"
                value={form.tenant_slug}
                onChange={(e) => setForm((s) => ({ ...s, tenant_slug: e.target.value }))}
              />
              <p className="text-xs text-muted-foreground mt-1">
                Enter a tenant code if you want to join an organization.
              </p>
            </div>
            <Button type="submit" className="w-full" disabled={loading}>
              <UserPlus className="w-4 h-4 mr-2" />
              {loading ? "Creating account..." : "Create Account"}
            </Button>
          </form>

          <div className="mt-4 text-center text-sm text-muted-foreground">
            Already have an account?{" "}
            <Link to="/login" className="text-primary hover:underline font-medium">
              Sign in
            </Link>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
