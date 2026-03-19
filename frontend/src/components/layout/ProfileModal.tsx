/**
 * ProfileModal
 * ============
 * Tabbed modal for editing profile (name + avatar) and changing password.
 */
import { useState, useRef } from "react";
import { User, Lock, Camera, Loader2, X, Eye, EyeOff } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { api, rewritePresignedUrl } from "@/lib/api";
import { useAuthStore } from "@/stores/authStore";
import type { User as UserType } from "@/types";

type Tab = "profile" | "security";

interface ProfileModalProps {
  onClose: () => void;
}

export function ProfileModal({ onClose }: ProfileModalProps) {
  const user = useAuthStore((s) => s.user)!;
  const updateUser = useAuthStore((s) => s.updateUser);
  const [tab, setTab] = useState<Tab>("profile");

  // ── Profile tab state ──────────────────────────────────────────────────────
  const [fullName, setFullName] = useState(user.full_name);
  const [avatarPreview, setAvatarPreview] = useState<string | null>(
    user.avatar_url ? rewritePresignedUrl(user.avatar_url) : null
  );
  const [avatarFile, setAvatarFile] = useState<File | null>(null);
  const [savingProfile, setSavingProfile] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // ── Security tab state ─────────────────────────────────────────────────────
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [showCurrent, setShowCurrent] = useState(false);
  const [showNew, setShowNew] = useState(false);
  const [savingPassword, setSavingPassword] = useState(false);

  // ── Avatar file pick ───────────────────────────────────────────────────────
  const handleAvatarChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setAvatarFile(file);
    const reader = new FileReader();
    reader.onload = () => setAvatarPreview(reader.result as string);
    reader.readAsDataURL(file);
  };

  // ── Save profile ───────────────────────────────────────────────────────────
  const handleSaveProfile = async () => {
    setSavingProfile(true);
    try {
      let updatedUser: UserType = user;

      // Upload avatar first if a new file was chosen
      if (avatarFile) {
        updatedUser = await api.uploadFile<UserType>("/auth/me/avatar", avatarFile);
        toast.success("Avatar updated");
      }

      // Update name if changed
      if (fullName.trim() !== user.full_name) {
        updatedUser = await api.put<UserType>("/auth/me", {
          full_name: fullName.trim(),
        });
        toast.success("Name updated");
      }

      updateUser(updatedUser);
      setAvatarFile(null);
    } catch (err: any) {
      toast.error(err.message || "Failed to save profile");
    } finally {
      setSavingProfile(false);
    }
  };

  // ── Save password ──────────────────────────────────────────────────────────
  const handleSavePassword = async () => {
    if (!currentPassword || !newPassword || !confirmPassword) {
      toast.error("All password fields are required");
      return;
    }
    if (newPassword.length < 6) {
      toast.error("New password must be at least 6 characters");
      return;
    }
    if (newPassword !== confirmPassword) {
      toast.error("New passwords do not match");
      return;
    }

    setSavingPassword(true);
    try {
      await api.put<UserType>("/auth/me", {
        current_password: currentPassword,
        new_password: newPassword,
      });
      toast.success("Password changed successfully");
      setCurrentPassword("");
      setNewPassword("");
      setConfirmPassword("");
    } catch (err: any) {
      toast.error(err.message || "Failed to change password");
    } finally {
      setSavingPassword(false);
    }
  };

  const initials = user.full_name
    .split(" ")
    .map((n) => n[0])
    .join("")
    .slice(0, 2)
    .toUpperCase();

  const profileDirty =
    fullName.trim() !== user.full_name || avatarFile !== null;

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 z-50 bg-black/50 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 z-50 w-full max-w-md bg-card border rounded-2xl shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between px-5 pt-5 pb-4 border-b">
          <h2 className="text-base font-semibold">Edit Profile</h2>
          <button
            onClick={onClose}
            className="rounded-full p-1 hover:bg-muted transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Tabs */}
        <div className="flex border-b px-5">
          <TabButton
            active={tab === "profile"}
            onClick={() => setTab("profile")}
            icon={<User className="w-3.5 h-3.5" />}
            label="Profile"
          />
          <TabButton
            active={tab === "security"}
            onClick={() => setTab("security")}
            icon={<Lock className="w-3.5 h-3.5" />}
            label="Security"
          />
        </div>

        {/* Content */}
        <div className="px-5 py-5 space-y-4">
          {tab === "profile" && (
            <>
              {/* Avatar */}
              <div className="flex flex-col items-center gap-3">
                <div className="relative group">
                  <div className="w-20 h-20 rounded-full overflow-hidden bg-primary/10 flex items-center justify-center text-2xl font-bold text-primary">
                    {avatarPreview ? (
                      <img
                        src={avatarPreview}
                        alt="avatar"
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      initials
                    )}
                  </div>
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="absolute inset-0 rounded-full bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center"
                    title="Change avatar"
                  >
                    <Camera className="w-5 h-5 text-white" />
                  </button>
                </div>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/jpeg,image/png,image/gif,image/webp"
                  className="hidden"
                  onChange={handleAvatarChange}
                />
                <p className="text-xs text-muted-foreground">
                  Click avatar to change · JPEG, PNG, GIF, WebP · Max 5 MB
                </p>
              </div>

              {/* Full name */}
              <div className="space-y-1.5">
                <label className="text-xs font-medium text-muted-foreground">
                  Full Name
                </label>
                <input
                  type="text"
                  value={fullName}
                  onChange={(e) => setFullName(e.target.value)}
                  className="w-full px-3 py-2 text-sm rounded-lg border bg-background focus:outline-none focus:ring-2 focus:ring-primary/30"
                  placeholder="Your full name"
                />
              </div>

              {/* Email (read-only) */}
              <div className="space-y-1.5">
                <label className="text-xs font-medium text-muted-foreground">
                  Email (cannot be changed)
                </label>
                <input
                  type="email"
                  value={user.email}
                  readOnly
                  className="w-full px-3 py-2 text-sm rounded-lg border bg-muted/30 text-muted-foreground cursor-not-allowed"
                />
              </div>
            </>
          )}

          {tab === "security" && (
            <>
              <PasswordField
                label="Current Password"
                value={currentPassword}
                onChange={setCurrentPassword}
                show={showCurrent}
                onToggleShow={() => setShowCurrent((v) => !v)}
                placeholder="Enter current password"
              />
              <PasswordField
                label="New Password"
                value={newPassword}
                onChange={setNewPassword}
                show={showNew}
                onToggleShow={() => setShowNew((v) => !v)}
                placeholder="Min 6 characters"
              />
              <PasswordField
                label="Confirm New Password"
                value={confirmPassword}
                onChange={setConfirmPassword}
                show={showNew}
                onToggleShow={() => setShowNew((v) => !v)}
                placeholder="Repeat new password"
              />
            </>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-2 px-5 pb-5">
          <Button variant="ghost" size="sm" onClick={onClose}>
            Cancel
          </Button>

          {tab === "profile" && (
            <Button
              size="sm"
              onClick={handleSaveProfile}
              disabled={savingProfile || !profileDirty}
            >
              {savingProfile && (
                <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />
              )}
              Save Profile
            </Button>
          )}

          {tab === "security" && (
            <Button
              size="sm"
              onClick={handleSavePassword}
              disabled={
                savingPassword ||
                !currentPassword ||
                !newPassword ||
                !confirmPassword
              }
            >
              {savingPassword && (
                <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />
              )}
              Change Password
            </Button>
          )}
        </div>
      </div>
    </>
  );
}

// ── Sub-components ─────────────────────────────────────────────────────────

function TabButton({
  active,
  onClick,
  icon,
  label,
}: {
  active: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  label: string;
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "flex items-center gap-1.5 px-3 py-2.5 text-xs font-medium border-b-2 -mb-px transition-colors",
        active
          ? "border-primary text-primary relative"
          : "border-transparent text-muted-foreground hover:text-foreground"
      )}
    >
      {icon}
      {label}
    </button>
  );
}

function PasswordField({
  label,
  value,
  onChange,
  show,
  onToggleShow,
  placeholder,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  show: boolean;
  onToggleShow: () => void;
  placeholder?: string;
}) {
  return (
    <div className="space-y-1.5">
      <label className="text-xs font-medium text-muted-foreground">{label}</label>
      <div className="relative">
        <input
          type={show ? "text" : "password"}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          className="w-full px-3 py-2 pr-9 text-sm rounded-lg border bg-background focus:outline-none focus:ring-2 focus:ring-primary/30"
        />
        <button
          type="button"
          onClick={onToggleShow}
          className="absolute right-2.5 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
        >
          {show ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
        </button>
      </div>
    </div>
  );
}
