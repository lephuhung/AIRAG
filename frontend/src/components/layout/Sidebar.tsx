import { memo } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import {
  Database,
  ChevronLeft,
  ChevronRight,
  FolderOpen,
  Activity,
  Building2,
} from "lucide-react";
import { useWorkspaces } from "@/hooks/useWorkspaces";
import { useAuthStore } from "@/stores/authStore";
import { ThemeToggle } from "@/components/ui/theme-toggle";
import { cn } from "@/lib/utils";

interface SidebarProps {
  collapsed: boolean;
  onToggle: () => void;
}

export const Sidebar = memo(function Sidebar({ collapsed, onToggle }: SidebarProps) {
  const navigate = useNavigate();
  const location = useLocation();
  const { data: workspaces } = useWorkspaces();
  const user = useAuthStore((s) => s.user);

  const urlWorkspaceId = location.pathname.match(/\/knowledge-bases\/(\d+)/)?.[1];
  const isHome = location.pathname === "/";
  const isFilesPage = location.pathname === "/files" || location.pathname.endsWith("/files");
  const isWorkersPage = location.pathname === "/workers";
  const isTenantPage = location.pathname.startsWith("/tenants/");

  return (
    <aside
      className={cn(
        "flex flex-col h-full bg-card border-r border-border transition-all duration-200 flex-shrink-0",
        collapsed ? "w-14" : "w-60"
      )}
    >
      {/* Logo */}
      <div className="flex items-center gap-2.5 px-3 h-12 border-b border-border flex-shrink-0">
        <Database className="w-6 h-6 text-primary flex-shrink-0" />
        {!collapsed && (
          <span className="font-bold text-primary text-base truncate">NexusRAG</span>
        )}
      </div>

      {/* Navigation */}
      <nav className="flex-shrink-0 px-2 pt-3 space-y-0.5">
        <button
          onClick={() => navigate("/")}
          className={cn(
            "w-full flex items-center gap-2.5 px-2.5 py-2 rounded-lg text-sm transition-colors",
            isHome
              ? "bg-primary/10 text-primary font-medium"
              : "text-muted-foreground hover:bg-muted/50 hover:text-foreground"
          )}
          title={collapsed ? "Knowledge Bases" : undefined}
        >
          <Database className="w-4 h-4 flex-shrink-0" />
          {!collapsed && <span className="truncate">Knowledge Bases</span>}
        </button>

        <button
          onClick={() => navigate("/files")}
          className={cn(
            "w-full flex items-center gap-2.5 px-2.5 py-2 rounded-lg text-sm transition-colors",
            isFilesPage
              ? "bg-primary/10 text-primary font-medium"
              : "text-muted-foreground hover:bg-muted/50 hover:text-foreground"
          )}
          title={collapsed ? "Files" : undefined}
        >
          <FolderOpen className="w-4 h-4 flex-shrink-0" />
          {!collapsed && <span className="truncate">Files</span>}
        </button>

        {user?.is_superadmin && (
          <button
            onClick={() => navigate("/workers")}
            className={cn(
              "w-full flex items-center gap-2.5 px-2.5 py-2 rounded-lg text-sm transition-colors",
              isWorkersPage
                ? "bg-primary/10 text-primary font-medium"
                : "text-muted-foreground hover:bg-muted/50 hover:text-foreground"
            )}
            title={collapsed ? "Workers" : undefined}
          >
            <Activity className="w-4 h-4 flex-shrink-0" />
            {!collapsed && <span className="truncate">Workers</span>}
          </button>
        )}
      </nav>

      {/* Scrollable workspace list */}
      <div className="flex-1 overflow-y-auto min-h-0">
        {!collapsed && workspaces && workspaces.length > 0 && (
          <div className="mt-4 px-2">
            <p className="px-2.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground mb-1.5">
              Workspaces
            </p>
            <div className="space-y-0.5">
              {workspaces.slice(0, 20).map((ws) => {
                const isActive = urlWorkspaceId === String(ws.id);
                return (
                  <button
                    key={ws.id}
                    onClick={() => navigate(`/knowledge-bases/${ws.id}`)}
                    className={cn(
                      "w-full flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-sm transition-colors",
                      isActive
                        ? "bg-primary/10 text-primary border-l-2 border-primary font-medium"
                        : "text-muted-foreground hover:bg-muted/50 hover:text-foreground"
                    )}
                  >
                    <Database className="w-3.5 h-3.5 flex-shrink-0" />
                    <span className="truncate">{ws.name}</span>
                    <span className="ml-auto text-[10px] text-muted-foreground/60 tabular-nums">
                      {ws.document_count}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>
        )}

        {/* Collapsed indicators */}
        {collapsed && (
          <div className="mt-4 px-2 space-y-1">
            {workspaces?.slice(0, 6).map((ws) => {
              const isActive = urlWorkspaceId === String(ws.id);
              return (
                <button
                  key={`ws-${ws.id}`}
                  onClick={() => navigate(`/knowledge-bases/${ws.id}`)}
                  className={cn(
                    "w-full flex items-center justify-center py-1.5 rounded-lg transition-colors",
                    isActive
                      ? "bg-primary/10 text-primary"
                      : "text-muted-foreground hover:bg-muted/50"
                  )}
                  title={ws.name}
                >
                  <Database className="w-3.5 h-3.5" />
                </button>
              );
            })}
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="flex-shrink-0 border-t border-border px-2 py-2 space-y-2">
        {/* User info */}
        {!collapsed && user && (
          <div className="px-2.5 py-1">
            <p className="text-xs font-medium truncate">{user.full_name}</p>
            <p className="text-[10px] text-muted-foreground truncate">{user.email}</p>
          </div>
        )}
        <div className="flex items-center justify-between">
          <ThemeToggle />
          <button
            onClick={onToggle}
            className="w-8 h-8 flex items-center justify-center rounded-lg text-muted-foreground hover:bg-muted/50 hover:text-foreground transition-colors"
            title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            {collapsed ? (
              <ChevronRight className="w-4 h-4" />
            ) : (
              <ChevronLeft className="w-4 h-4" />
            )}
          </button>
        </div>
      </div>
    </aside>
  );
});
