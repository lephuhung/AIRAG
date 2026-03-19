import { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { useAuthStore } from "@/stores/authStore";
import { LogOut, Shield, Users, Building2, UserCircle } from "lucide-react";
import { cn } from "@/lib/utils";
import { ProfileModal } from "./ProfileModal";
import { rewritePresignedUrl } from "@/lib/api";

export function UserMenu() {
  const navigate = useNavigate();
  const user = useAuthStore((s) => s.user);
  const logout = useAuthStore((s) => s.logout);
  const [open, setOpen] = useState(false);
  const [showProfile, setShowProfile] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  // Close on outside click
  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, [open]);

  if (!user) return null;

  const initials = user.full_name
    .split(" ")
    .map((n) => n[0])
    .join("")
    .slice(0, 2)
    .toUpperCase();

  const handleLogout = () => {
    logout();
    navigate("/login");
  };

  return (
    <>
      <div ref={menuRef} className="relative">
        <button
          onClick={() => setOpen(!open)}
          className={cn(
            "w-8 h-8 rounded-full overflow-hidden flex items-center justify-center text-xs font-semibold transition-colors",
            user.avatar_url
              ? "ring-2 ring-primary/30"
              : "bg-primary/10 text-primary hover:bg-primary/20"
          )}
          title={user.full_name}
        >
          {user.avatar_url ? (
            <img
              src={rewritePresignedUrl(user.avatar_url)}
              alt={user.full_name}
              className="w-full h-full object-cover"
            />
          ) : (
            initials
          )}
        </button>

        {open && (
          <div className="absolute right-0 top-10 z-50 bg-card border rounded-lg shadow-xl py-1 w-56">
            {/* User info */}
            <div className="px-3 py-2 border-b border-border">
              <div className="flex items-center gap-2">
                <p className="text-sm font-medium truncate">{user.full_name}</p>
                {user.is_superadmin && (
                  <span title="Super Admin">
                    <Shield className="w-3.5 h-3.5 text-amber-500 flex-shrink-0" />
                  </span>
                )}
              </div>
              <p className="text-xs text-muted-foreground truncate">{user.email}</p>
            </div>

            {/* Profile */}
            <button
              onClick={() => {
                setShowProfile(true);
                setOpen(false);
              }}
              className="w-full flex items-center gap-2 px-3 py-2 text-sm text-foreground hover:bg-muted transition-colors"
            >
              <UserCircle className="w-4 h-4" />
              Edit Profile
            </button>

            {/* Superadmin items */}
            {user.is_superadmin && (
              <>
                <div className="border-t border-border my-1" />
                <button
                  onClick={() => {
                    navigate("/admin/users");
                    setOpen(false);
                  }}
                  className="w-full flex items-center gap-2 px-3 py-2 text-sm text-foreground hover:bg-muted transition-colors"
                >
                  <Users className="w-4 h-4" />
                  Manage Users
                </button>
                <button
                  onClick={() => {
                    navigate("/admin/tenants");
                    setOpen(false);
                  }}
                  className="w-full flex items-center gap-2 px-3 py-2 text-sm text-foreground hover:bg-muted transition-colors"
                >
                  <Building2 className="w-4 h-4" />
                  Manage Tenants
                </button>
              </>
            )}

            <div className="border-t border-border my-1" />
            <button
              onClick={handleLogout}
              className="w-full flex items-center gap-2 px-3 py-2 text-sm text-destructive hover:bg-muted transition-colors"
            >
              <LogOut className="w-4 h-4" />
              Sign Out
            </button>
          </div>
        )}
      </div>

      {showProfile && <ProfileModal onClose={() => setShowProfile(false)} />}
    </>
  );
}
