/**
 * TwoFactorSection
 * ================
 * Enable/disable TOTP two-factor auth (Google Authenticator) from the profile
 * security tab. Enrollment is two-step: POST /auth/2fa/setup mints a pending
 * secret + QR, then POST /auth/2fa/enable activates it once the user proves they
 * can produce a valid code. Disabling requires a current code or the password.
 */
import { useState } from "react";
import { ShieldCheck, ShieldOff, Loader2, Copy, Check } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";
import { useAuthStore } from "@/stores/authStore";
import { copyToClipboard } from "@/lib/clipboard";
import type { User as UserType } from "@/types";

interface SetupData {
  secret: string;
  otpauth_uri: string;
  qr_data_uri: string;
}

export function TwoFactorSection() {
  const user = useAuthStore((s) => s.user)!;
  const updateUser = useAuthStore((s) => s.updateUser);

  const enabled = !!user.two_factor_enabled;

  const [setup, setSetup] = useState<SetupData | null>(null);
  const [code, setCode] = useState("");
  const [disablePassword, setDisablePassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [copied, setCopied] = useState(false);

  const onlyDigits = (v: string) => v.replace(/\D/g, "").slice(0, 6);

  const startSetup = async () => {
    setBusy(true);
    try {
      const data = await api.post<SetupData>("/auth/2fa/setup");
      setSetup(data);
      setCode("");
    } catch (err: any) {
      toast.error(err.message || "Failed to start 2FA setup");
    } finally {
      setBusy(false);
    }
  };

  const confirmEnable = async () => {
    if (code.length !== 6) return;
    setBusy(true);
    try {
      await api.post<{ enabled: boolean }>("/auth/2fa/enable", { code });
      updateUser({ ...user, two_factor_enabled: true } as UserType);
      setSetup(null);
      setCode("");
      toast.success("Two-factor authentication enabled");
    } catch (err: any) {
      toast.error(err.message || "Invalid code");
    } finally {
      setBusy(false);
    }
  };

  const disable = async () => {
    if (code.length !== 6 && !disablePassword) {
      toast.error("Enter a current code or your password");
      return;
    }
    setBusy(true);
    try {
      await api.post<{ enabled: boolean }>("/auth/2fa/disable", {
        ...(code.length === 6 ? { code } : {}),
        ...(disablePassword ? { password: disablePassword } : {}),
      });
      updateUser({ ...user, two_factor_enabled: false } as UserType);
      setCode("");
      setDisablePassword("");
      toast.success("Two-factor authentication disabled");
    } catch (err: any) {
      toast.error(err.message || "Failed to disable 2FA");
    } finally {
      setBusy(false);
    }
  };

  const copySecret = async () => {
    if (!setup) return;
    await copyToClipboard(setup.secret);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <div className="space-y-3 border-t pt-4">
      <div className="flex items-center gap-2">
        {enabled ? (
          <ShieldCheck className="w-4 h-4 text-green-600" />
        ) : (
          <ShieldOff className="w-4 h-4 text-muted-foreground" />
        )}
        <h3 className="text-sm font-semibold">Two-Factor Authentication</h3>
        <span
          className={
            "ml-auto text-xs px-2 py-0.5 rounded-full " +
            (enabled
              ? "bg-green-500/15 text-green-600"
              : "bg-muted text-muted-foreground")
          }
        >
          {enabled ? "On" : "Off"}
        </span>
      </div>

      {/* ── OFF: not enrolling yet ─────────────────────────────────────── */}
      {!enabled && !setup && (
        <>
          <p className="text-xs text-muted-foreground">
            Add an extra layer of security using Google Authenticator (or any TOTP
            app). You'll enter a 6-digit code each time you sign in.
          </p>
          <Button size="sm" onClick={startSetup} disabled={busy}>
            {busy && <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />}
            Enable 2FA
          </Button>
        </>
      )}

      {/* ── OFF: enrolling (QR shown, awaiting code) ───────────────────── */}
      {!enabled && setup && (
        <div className="space-y-3">
          <p className="text-xs text-muted-foreground">
            Scan this QR code with Google Authenticator, then enter the 6-digit
            code below to finish enabling 2FA.
          </p>
          <div className="flex justify-center">
            <img
              src={setup.qr_data_uri}
              alt="2FA QR code"
              className="w-44 h-44 rounded-lg border bg-white p-2"
            />
          </div>
          <div className="space-y-1.5">
            <label className="text-xs font-medium text-muted-foreground">
              Or enter this key manually
            </label>
            <div className="flex items-center gap-2">
              <code className="flex-1 px-2 py-1.5 text-xs rounded-lg border bg-muted/40 font-mono break-all">
                {setup.secret}
              </code>
              <button
                type="button"
                onClick={copySecret}
                className="shrink-0 rounded-lg p-1.5 hover:bg-muted transition-colors"
                title="Copy key"
              >
                {copied ? (
                  <Check className="w-3.5 h-3.5 text-green-600" />
                ) : (
                  <Copy className="w-3.5 h-3.5" />
                )}
              </button>
            </div>
          </div>
          <div className="space-y-1.5">
            <label className="text-xs font-medium text-muted-foreground">
              Verification code
            </label>
            <input
              type="text"
              inputMode="numeric"
              autoComplete="one-time-code"
              placeholder="6-digit code"
              value={code}
              onChange={(e) => setCode(onlyDigits(e.target.value))}
              className="w-full px-3 py-2 text-sm rounded-lg border bg-background focus:outline-none focus:ring-2 focus:ring-primary/30 tracking-widest"
            />
          </div>
          <div className="flex gap-2">
            <Button
              size="sm"
              onClick={confirmEnable}
              disabled={busy || code.length !== 6}
            >
              {busy && <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />}
              Verify & Enable
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => {
                setSetup(null);
                setCode("");
              }}
              disabled={busy}
            >
              Cancel
            </Button>
          </div>
        </div>
      )}

      {/* ── ON: disable form ───────────────────────────────────────────── */}
      {enabled && (
        <div className="space-y-3">
          <p className="text-xs text-muted-foreground">
            2FA is active. To turn it off, confirm with a current authenticator
            code or your account password.
          </p>
          <div className="space-y-1.5">
            <label className="text-xs font-medium text-muted-foreground">
              Authenticator code
            </label>
            <input
              type="text"
              inputMode="numeric"
              autoComplete="one-time-code"
              placeholder="6-digit code"
              value={code}
              onChange={(e) => setCode(onlyDigits(e.target.value))}
              className="w-full px-3 py-2 text-sm rounded-lg border bg-background focus:outline-none focus:ring-2 focus:ring-primary/30 tracking-widest"
            />
          </div>
          <div className="space-y-1.5">
            <label className="text-xs font-medium text-muted-foreground">
              …or account password
            </label>
            <input
              type="password"
              placeholder="Current password"
              value={disablePassword}
              onChange={(e) => setDisablePassword(e.target.value)}
              className="w-full px-3 py-2 text-sm rounded-lg border bg-background focus:outline-none focus:ring-2 focus:ring-primary/30"
            />
          </div>
          <Button
            size="sm"
            variant="destructive"
            onClick={disable}
            disabled={busy || (code.length !== 6 && !disablePassword)}
          >
            {busy && <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />}
            Disable 2FA
          </Button>
        </div>
      )}
    </div>
  );
}
