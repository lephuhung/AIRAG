import { memo, useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import {
  Database,
  ChevronLeft,
  ChevronRight,
  ChevronDown,
  FolderOpen,
  Activity,
  Building2,
  Users,
  FileText,
  MessageSquare,
  Plus,
  Trash2,
  Edit,
  PieChart,
  Loader2,
} from "lucide-react";
import { useWorkspaces } from "@/hooks/useWorkspaces";
import { useMyTenants } from "@/hooks/useMyTenants";
import { useAuthStore } from "@/stores/authStore";
import { useChatSessions, useCreateChatSession, useDeleteChatSession } from "@/hooks/useChatSessions";
import { useTranslation } from "@/hooks/useTranslation";
import { ThemeToggle } from "@/components/ui/theme-toggle";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import logo from "@/assets/logo.png";

interface SidebarProps {
  collapsed: boolean;
  onToggle: () => void;
}

export const Sidebar = memo(function Sidebar({ collapsed, onToggle }: SidebarProps) {
  const navigate = useNavigate();
  const location = useLocation();
  const { data: workspaces } = useWorkspaces();
  const { data: myTenants } = useMyTenants();
  const user = useAuthStore((s) => s.user);

  const { data: sessions } = useChatSessions();
  const createSession = useCreateChatSession();
  const deleteSession = useDeleteChatSession();
  const { t } = useTranslation();

  const [workspacesExpanded, setWorkspacesExpanded] = useState(true);

  const handleNewSession = async (e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      const newSession = await createSession.mutateAsync({ title: t("nav.new_chat") });
      navigate(`/chat/${newSession.id}`);
    } catch (error) {
      toast.error(t("chat.create_failed"));
    }
  };

  const handleDeleteSession = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    e.preventDefault();
    if (confirm(t("chat.delete_confirm"))) {
      try {
        await deleteSession.mutateAsync(id);
        toast.success(t("chat.delete_success"));
        if (location.pathname === `/chat/${id}`) {
          navigate("/chat");
        }
      } catch (error) {
        toast.error(t("chat.delete_failed"));
      }
    }
  };

  const urlWorkspaceId = location.pathname.match(/\/knowledge-bases\/(\d+)/)?.[1];
  const isHome = location.pathname === "/";
  const isFilesPage = location.pathname === "/files" || location.pathname.endsWith("/files");
  const isChatPage = location.pathname === "/chat" || location.pathname.startsWith("/chat/");
  const isWorkersPage = location.pathname === "/workers";
  const isAdminUsersPage = location.pathname === "/admin/users";
  const isAdminTenantsPage = location.pathname === "/admin/tenants";
  const isAdminDocTypesPage = location.pathname === "/admin/document-types";
  const isAdminDashboardPage = location.pathname === "/admin/dashboard";

  return (
    <aside
      className={cn(
        "flex flex-col h-full bg-card border-r border-border transition-all duration-200 flex-shrink-0",
        collapsed ? "w-14" : "w-60"
      )}
    >
      {/* Logo */}
      <div className="flex items-center gap-2.5 px-3 h-12 border-b border-border flex-shrink-0">
        <img src={logo} alt="Logo" className="w-7 h-7 object-contain flex-shrink-0" />
        {!collapsed && (
          <span className="font-bold text-primary text-base truncate">{t("app.name")}</span>
        )}
      </div>

      {/* Navigation */}
      <nav className="flex-shrink-0 px-2 pt-3 space-y-0.5">
        <button
          onClick={handleNewSession}
          disabled={createSession.isPending}
          className={cn(
            "w-full flex items-center justify-between px-2.5 py-2 rounded-lg text-sm transition-colors group",
            isChatPage && !location.pathname.match(/\/chat\/\d+/)
              ? "bg-primary/10 text-primary font-medium"
              : "text-muted-foreground hover:bg-muted/50 hover:text-foreground",
            createSession.isPending && "opacity-70 cursor-not-allowed"
          )}
          title={collapsed ? t("nav.new_chat") : undefined}
        >
          <div className="flex items-center gap-2.5">
            {createSession.isPending ? (
              <Loader2 className="w-4 h-4 animate-spin flex-shrink-0" />
            ) : (
              <Edit className="w-4 h-4 flex-shrink-0" />
            )}
            {!collapsed && <span className="truncate">{t("nav.new_chat")}</span>}
          </div>
          {!collapsed && !createSession.isPending && (
            <div className="opacity-0 group-hover:opacity-100 transition-opacity p-0.5" title="New Chat">
               <Plus className="w-4 h-4" />
            </div>
          )}
        </button>

        <button
          onClick={() => navigate("/")}
          className={cn(
            "w-full flex items-center gap-2.5 px-2.5 py-2 rounded-lg text-sm transition-colors",
            isHome
              ? "bg-primary/10 text-primary font-medium"
              : "text-muted-foreground hover:bg-muted/50 hover:text-foreground"
          )}
          title={collapsed ? t("nav.knowledge_bases") : undefined}
        >
          <Database className="w-4 h-4 flex-shrink-0" />
          {!collapsed && <span className="truncate">{t("nav.knowledge_bases")}</span>}
        </button>

        <button
          onClick={() => navigate("/files")}
          className={cn(
            "w-full flex items-center gap-2.5 px-2.5 py-2 rounded-lg text-sm transition-colors",
            isFilesPage
              ? "bg-primary/10 text-primary font-medium"
              : "text-muted-foreground hover:bg-muted/50 hover:text-foreground"
          )}
          title={collapsed ? t("nav.files") : undefined}
        >
          <FolderOpen className="w-4 h-4 flex-shrink-0" />
          {!collapsed && <span className="truncate">{t("nav.files")}</span>}
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
            title={collapsed ? t("nav.workers") : undefined}
          >
            <Activity className="w-4 h-4 flex-shrink-0" />
            {!collapsed && <span className="truncate">{t("nav.workers")}</span>}
          </button>
        )}

        {/* Admin section (superadmin only) */}
        {user?.is_superadmin && (
          <>
            {!collapsed && (
              <p className="px-2.5 pt-4 pb-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                {t("common.admin")}
              </p>
            )}
            {collapsed && <div className="my-2 mx-2 border-t border-border" />}
            <button
              onClick={() => navigate("/admin/dashboard")}
              className={cn(
                "w-full flex items-center gap-2.5 px-2.5 py-2 rounded-lg text-sm transition-colors",
                isAdminDashboardPage
                  ? "bg-primary/10 text-primary font-medium"
                  : "text-muted-foreground hover:bg-muted/50 hover:text-foreground"
              )}
              title={collapsed ? t("nav.admin.dashboard") : undefined}
            >
              <PieChart className="w-4 h-4 flex-shrink-0" />
              {!collapsed && <span className="truncate">{t("nav.admin.dashboard")}</span>}
            </button>
            <button
              onClick={() => navigate("/admin/users")}
              className={cn(
                "w-full flex items-center gap-2.5 px-2.5 py-2 rounded-lg text-sm transition-colors",
                isAdminUsersPage
                  ? "bg-primary/10 text-primary font-medium"
                  : "text-muted-foreground hover:bg-muted/50 hover:text-foreground"
              )}
              title={collapsed ? t("nav.admin.users") : undefined}
            >
              <Users className="w-4 h-4 flex-shrink-0" />
              {!collapsed && <span className="truncate">{t("nav.admin.users")}</span>}
            </button>
            <button
              onClick={() => navigate("/admin/tenants")}
              className={cn(
                "w-full flex items-center gap-2.5 px-2.5 py-2 rounded-lg text-sm transition-colors",
                isAdminTenantsPage
                  ? "bg-primary/10 text-primary font-medium"
                  : "text-muted-foreground hover:bg-muted/50 hover:text-foreground"
              )}
              title={collapsed ? t("nav.admin.tenants") : undefined}
            >
              <Building2 className="w-4 h-4 flex-shrink-0" />
              {!collapsed && <span className="truncate">{t("nav.admin.tenants")}</span>}
            </button>
            <button
              onClick={() => navigate("/admin/document-types")}
              className={cn(
                "w-full flex items-center gap-2.5 px-2.5 py-2 rounded-lg text-sm transition-colors",
                isAdminDocTypesPage
                  ? "bg-primary/10 text-primary font-medium"
                  : "text-muted-foreground hover:bg-muted/50 hover:text-foreground"
              )}
              title={collapsed ? t("nav.admin.document_types") : undefined}
            >
              <FileText className="w-4 h-4 flex-shrink-0" />
              {!collapsed && <span className="truncate">{t("nav.admin.document_types")}</span>}
            </button>
          </>
        )}
      </nav>

      {/* Scrollable lists */}
      <div className="flex-1 overflow-y-auto min-h-0 pt-2 pb-4">
        {/* Workspaces */}
        {!collapsed && workspaces && workspaces.length > 0 && (
          <div className="mt-2 px-2">
            <div 
              className="flex items-center justify-between px-2.5 mb-1.5 cursor-pointer group"
              onClick={() => setWorkspacesExpanded(!workspacesExpanded)}
            >
              <div className="flex items-center gap-1.5">
                <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground group-hover:text-foreground transition-colors">
                  {t("nav.knowledge_bases")}
                </p>
                <span className="text-[10px] font-bold bg-muted group-hover:bg-muted/80 text-muted-foreground px-1.5 rounded-full min-w-[20px] text-center tabular-nums transition-colors">
                  {workspaces.length}
                </span>
              </div>
              <button 
                className="text-muted-foreground group-hover:text-foreground transition-colors"
              >
                {workspacesExpanded ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
              </button>
            </div>
            
            <AnimatePresence>
              {workspacesExpanded && (
                <motion.div 
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: "auto", opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  className="space-y-0.5 overflow-hidden"
                >
                  {workspaces.slice(0, 20).map((ws) => {
                    const isActive = urlWorkspaceId === String(ws.id);
                    return (
                      <button
                        key={ws.id}
                        onClick={() => navigate(`/knowledge-bases/${ws.id}`)}
                        className={cn(
                          "w-full flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-sm transition-colors group",
                          isActive
                            ? "bg-primary/10 text-primary border-l-2 border-primary font-medium"
                            : "text-muted-foreground hover:bg-muted/50 hover:text-foreground"
                        )}
                      >
                        <Database className="w-3.5 h-3.5 flex-shrink-0" />
                        <span className="truncate">{ws.name}</span>
                        <span className={cn(
                          "ml-auto text-xs font-bold px-1.5 rounded-full min-w-[20px] tabular-nums transition-colors tracking-tight",
                          isActive
                            ? "bg-primary/20 text-primary"
                            : "bg-muted text-muted-foreground group-hover:bg-muted-foreground/15 group-hover:text-foreground"
                        )}>
                          {ws.document_count}
                        </span>
                      </button>
                    );
                  })}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        )}

        {/* Collapsed indicators */}
        {collapsed && (
          <div className="mt-2 px-2 space-y-1">
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

        {/* My Tenants */}
        {!collapsed && myTenants && myTenants.length > 0 && (
          <div className="mt-6 px-2">
            <p className="px-2.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground mb-1.5">
              {t("nav.my_tenants")}
            </p>
            <div className="space-y-0.5">
              {myTenants.map((t) => {
                const isActive = location.pathname === `/tenants/${t.id}`;
                return (
                  <button
                    key={`t-${t.id}`}
                    onClick={() => navigate(`/tenants/${t.id}`)}
                    className={cn(
                      "w-full flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-sm transition-colors",
                      isActive
                        ? "bg-primary/10 text-primary border-l-2 border-primary font-medium"
                        : "text-muted-foreground hover:bg-muted/50 hover:text-foreground"
                    )}
                  >
                    <Building2 className="w-3.5 h-3.5 flex-shrink-0" />
                    <span className="truncate">{t.name}</span>
                  </button>
                );
              })}
            </div>
          </div>
        )}
        {collapsed && myTenants && myTenants.length > 0 && (
          <div className="mt-4 px-2 space-y-1">
            {myTenants.map((t) => {
              const isActive = location.pathname === `/tenants/${t.id}`;
              return (
                <button
                  key={`tc-${t.id}`}
                  onClick={() => navigate(`/tenants/${t.id}`)}
                  className={cn(
                    "w-full flex items-center justify-center py-1.5 rounded-lg transition-colors",
                    isActive
                      ? "bg-primary/10 text-primary"
                      : "text-muted-foreground hover:bg-muted/50"
                  )}
                  title={t.name}
                >
                  <Building2 className="w-3.5 h-3.5" />
                </button>
              );
            })}
          </div>
        )}

        {/* Chat History placed at the bottom */}
        {!collapsed && sessions && sessions.length > 0 && (
          <div className="mt-6 px-2">
            <p className="px-2.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground mb-1.5">
              {t("nav.your_chats")}
            </p>
            <div className="space-y-0.5">
              {sessions.map((session) => {
                const isActive = location.pathname === `/chat/${session.id}`;
                return (
                  <button
                    key={session.id}
                    onClick={() => navigate(`/chat/${session.id}`)}
                    className={cn(
                      "w-full flex items-center justify-between px-2.5 py-1.5 rounded-lg text-sm transition-colors group",
                      isActive
                        ? "bg-primary/10 text-primary font-medium"
                        : "text-muted-foreground hover:bg-muted/50 hover:text-foreground"
                    )}
                    title={session.title}
                  >
                    <div className="flex items-center gap-2 overflow-hidden flex-1">
                      <MessageSquare className={cn("w-3.5 h-3.5 flex-shrink-0", isActive ? "text-primary" : "text-muted-foreground")} />
                      <span className="truncate text-left font-medium">{session.title}</span>
                    </div>
                    <div
                      onClick={(e) => handleDeleteSession(e, session.id)}
                      className="opacity-0 group-hover:opacity-100 hover:text-destructive shrink-0 transition-opacity ml-1 p-0.5"
                      title="Delete Chat"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        )}
        
        {collapsed && sessions && sessions.length > 0 && (
          <div className="mt-4 px-2 space-y-1">
            {sessions.map((session) => {
              const isActive = location.pathname === `/chat/${session.id}`;
              return (
                <button
                  key={`sc-${session.id}`}
                  onClick={() => navigate(`/chat/${session.id}`)}
                  className={cn(
                    "w-full flex items-center justify-center py-1.5 rounded-lg transition-colors",
                    isActive
                      ? "bg-primary/10 text-primary"
                      : "text-muted-foreground hover:bg-muted/50"
                  )}
                  title={session.title}
                >
                  <MessageSquare className="w-3.5 h-3.5" />
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
