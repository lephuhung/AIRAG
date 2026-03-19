import { useState, useEffect } from "react";
import { useNavigate, Link, useSearchParams } from "react-router-dom";
import { toast } from "sonner";
import { useAuthStore } from "@/stores/authStore";
import { useValidateInvite } from "@/hooks/useInvites";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent } from "@/components/ui/card";
import { Database, UserPlus, Building2, Loader2, AlertCircle, CheckCircle2 } from "lucide-react";

export function RegisterPage() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const inviteToken = searchParams.get("invite");

  const register = useAuthStore((s) => s.register);
  const { data: inviteData, isLoading: inviteLoading } = useValidateInvite(inviteToken);

  const [form, setForm] = useState({
    email: "",
    password: "",
    full_name: "",
    tenant_slug: "",
  });
  const [loading, setLoading] = useState(false);

  // Pre-fill email if invite has a locked email
  useEffect(() => {
    if (inviteData?.valid && inviteData.email) {
      setForm((s) => ({ ...s, email: inviteData.email! }));
    }
  }, [inviteData]);

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
        invite_token: inviteToken || undefined,
      });
      toast.success(result.message);
      navigate("/login");
    } catch (err: any) {
      toast.error(err.message || "Registration failed");
    } finally {
      setLoading(false);
    }
  };

  const isInviteValid = !!(inviteToken && inviteData?.valid);
  const isInviteInvalid = !!(inviteToken && !inviteLoading && !inviteData?.valid);
  const emailLocked = isInviteValid && !!inviteData?.email;

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

          {/* Invite validation status */}
          {inviteToken && inviteLoading && (
            <div className="flex items-center gap-2 p-3 mb-4 rounded-lg bg-muted/50 border">
              <Loader2 className="w-4 h-4 animate-spin text-muted-foreground" />
              <span className="text-sm text-muted-foreground">Validating invite link...</span>
            </div>
          )}

          {isInviteValid && (
            <div className="flex items-center gap-2 p-3 mb-4 rounded-lg bg-emerald-500/10 border border-emerald-500/20">
              <Building2 className="w-4 h-4 text-emerald-600 flex-shrink-0" />
              <div className="min-w-0">
                <p className="text-sm font-medium text-emerald-700 dark:text-emerald-400">
                  Invited to {inviteData.tenant_name}
                </p>
                <p className="text-xs text-emerald-600/70 dark:text-emerald-400/70">
                  Your account will be activated automatically
                </p>
              </div>
            </div>
          )}

          {isInviteInvalid && (
            <div className="flex items-center gap-2 p-3 mb-4 rounded-lg bg-destructive/10 border border-destructive/20">
              <AlertCircle className="w-4 h-4 text-destructive flex-shrink-0" />
              <div className="min-w-0">
                <p className="text-sm font-medium text-destructive">
                  Invalid or expired invite link
                </p>
                <p className="text-xs text-destructive/70">
                  You can still register normally below
                </p>
              </div>
            </div>
          )}

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
                disabled={emailLocked}
                className={emailLocked ? "opacity-70 cursor-not-allowed" : ""}
              />
              {emailLocked && (
                <p className="text-xs text-muted-foreground mt-1 flex items-center gap-1">
                  <CheckCircle2 className="w-3 h-3" />
                  Email set by invite link
                </p>
              )}
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

            {/* Only show tenant slug field if NOT using invite link */}
            {!isInviteValid && (
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
            )}

            <Button type="submit" className="w-full" disabled={loading || inviteLoading}>
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
