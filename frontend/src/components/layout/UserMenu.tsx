import { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { useAuthStore } from "@/stores/authStore";
import { LogOut, User, Shield } from "lucide-react";
import { cn } from "@/lib/utils";

export function UserMenu() {
  const navigate = useNavigate();
  const user = useAuthStore((s) => s.user);
  const logout = useAuthStore((s) => s.logout);
  const [open, setOpen] = useState(false);
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
    <div ref={menuRef} className="relative">
      <button
        onClick={() => setOpen(!open)}
        className={cn(
          "w-8 h-8 rounded-full flex items-center justify-center text-xs font-semibold transition-colors",
          "bg-primary/10 text-primary hover:bg-primary/20"
        )}
        title={user.full_name}
      >
        {initials}
      </button>

      {open && (
        <div className="absolute right-0 top-10 z-50 bg-card border rounded-lg shadow-xl py-1 w-56">
          {/* User info */}
          <div className="px-3 py-2 border-b border-border">
            <div className="flex items-center gap-2">
              <p className="text-sm font-medium truncate">{user.full_name}</p>
              {user.is_superadmin && (
                <Shield className="w-3.5 h-3.5 text-amber-500 flex-shrink-0" title="Super Admin" />
              )}
            </div>
            <p className="text-xs text-muted-foreground truncate">{user.email}</p>
          </div>

          {/* Menu items */}
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
  );
}
