import { useState, useEffect } from "react";
import { useTranslation } from "@/hooks/useTranslation";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";
import { useWorkspaces, useCreateWorkspace, useDeleteWorkspace, useUpdateWorkspace } from "@/hooks/useWorkspaces";
import { useMyTenants } from "@/hooks/useMyTenants";
import { useAdminTenants } from "@/hooks/useAdminTenants";
import { useAuthStore } from "@/stores/authStore";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  Plus,
  Database,
  FileText,
  Trash2,
  MoreHorizontal,
  X,
  Globe,
  Building2,
  User,
  ChevronDown,
  Edit,
} from "lucide-react";
import { ConfirmDialog } from "@/components/ui/confirm-dialog";
import { cn } from "@/lib/utils";
import type { KnowledgeBase, CreateWorkspace } from "@/types";

type VisibilityOption = "personal" | "tenant" | "public";

export function KnowledgeBasesPage() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const { data: workspaces, isLoading } = useWorkspaces();
  const { data: myTenants } = useMyTenants();
  const { data: allTenants } = useAdminTenants();
  const isSuperadmin = useAuthStore((s) => s.user?.is_superadmin ?? false);
  // Superadmin sees all tenants; regular users see only their own
  const tenantsForDropdown = isSuperadmin ? allTenants : myTenants;
  const createWorkspace = useCreateWorkspace();
  const deleteWorkspace = useDeleteWorkspace();
  const updateWorkspace = useUpdateWorkspace();
  const [showNewWorkspace, setShowNewWorkspace] = useState(false);
  const [newWorkspaceName, setNewWorkspaceName] = useState("");
  const [newVisibility, setNewVisibility] = useState<VisibilityOption>("personal");
  const [selectedTenantId, setSelectedTenantId] = useState<number | null>(null);
  const [deleteConfirm, setDeleteConfirm] = useState<number | null>(null);
  const [openMenu, setOpenMenu] = useState<number | null>(null);

  const [editWorkspace, setEditWorkspace] = useState<KnowledgeBase | null>(null);
  const [editWorkspaceName, setEditWorkspaceName] = useState("");
  const [editVisibility, setEditVisibility] = useState<VisibilityOption>("personal");
  const [editTenantId, setEditTenantId] = useState<number | null>(null);

  useEffect(() => {
    if (editWorkspace) {
      setEditWorkspaceName(editWorkspace.name);
      setEditVisibility(editWorkspace.visibility as VisibilityOption);
      setEditTenantId(editWorkspace.tenant_id);
    }
  }, [editWorkspace]);

  const handleUpdateWorkspace = async () => {
    if (!editWorkspace || !editWorkspaceName.trim()) return;
    if (editVisibility === "tenant" && !editTenantId) {
      toast.error(t("kb.org_required"));
      return;
    }
    try {
      await updateWorkspace.mutateAsync({
        id: editWorkspace.id,
        data: {
          name: editWorkspaceName,
          visibility: editVisibility,
          tenant_id: editVisibility === "tenant" ? editTenantId : null,
        }
      });
      toast.success(t("kb.update_success"));
      setEditWorkspace(null);
    } catch {
      toast.error(t("kb.update_failed"));
    }
  };

  // Close menu on outside click
  useEffect(() => {
    if (openMenu === null) return;
    const close = () => setOpenMenu(null);
    document.addEventListener("click", close);
    return () => document.removeEventListener("click", close);
  }, [openMenu]);

  const handleCreateWorkspace = async () => {
    if (!newWorkspaceName.trim()) return;
    // Require tenant selection when visibility is "tenant"
    if (newVisibility === "tenant" && !selectedTenantId) {
      toast.error(t("kb.org_required"));
      return;
    }
    try {
      const payload: CreateWorkspace = {
        name: newWorkspaceName,
        visibility: newVisibility,
        tenant_id: newVisibility === "tenant" ? selectedTenantId : undefined,
      };
      const ws = await createWorkspace.mutateAsync(payload);
      toast.success(t("kb.create_success"));
      setNewWorkspaceName("");
      setNewVisibility("personal");
      setSelectedTenantId(null);
      setShowNewWorkspace(false);
      navigate(`/knowledge-bases/${ws.id}`);
    } catch {
      toast.error(t("kb.create_failed"));
    }
  };

  const handleDeleteWorkspace = async (id: number) => {
    try {
      await deleteWorkspace.mutateAsync(id);
      toast.success(t("kb.delete_success"));
    } catch {
      toast.error(t("kb.delete_failed"));
    }
    setDeleteConfirm(null);
  };

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));
    if (days === 0) return t("common.today");
    if (days === 1) return t("common.yesterday");
    if (days < 7) return t("common.days_ago", { count: days });
    return date.toLocaleDateString();
  };

  // Split workspaces into sections
  const publicWorkspaces = workspaces?.filter((ws) => ws.visibility === "public") ?? [];
  const tenantWorkspaces = workspaces?.filter((ws) => ws.visibility === "tenant") ?? [];
  const personalWorkspaces = workspaces?.filter((ws) => ws.visibility === "personal") ?? [];
  // Legacy workspaces (no owner) — treat as shared/public
  const legacyWorkspaces = workspaces?.filter(
    (ws) => !ws.owner_id && ws.visibility !== "public" && ws.visibility !== "tenant" && ws.visibility !== "personal"
  ) ?? [];

  const renderWorkspaceCard = (ws: KnowledgeBase) => (
    <Card
      key={ws.id}
      className="group cursor-pointer transition-all hover:ring-1 hover:ring-primary/20 hover:-translate-y-0.5 hover:shadow-lg"
      onClick={() => navigate(`/knowledge-bases/${ws.id}`)}
    >
      <CardContent className="pt-5 pb-4">
        <div className="flex items-start justify-between mb-1">
          <div className="flex items-center gap-2.5 min-w-0">
            <div className="w-8 h-8 rounded-lg bg-blue-500/10 flex items-center justify-center flex-shrink-0">
              <Database className="w-4 h-4 text-blue-500" />
            </div>
            <div className="min-w-0">
              <h3 className="font-medium text-sm truncate">{ws.name}</h3>
              {ws.description && (
                <p className="text-xs text-muted-foreground truncate mt-0.5">
                  {ws.description}
                </p>
              )}
            </div>
          </div>
          <div className="relative flex-shrink-0">
            <button
              onClick={(e) => {
                e.stopPropagation();
                setOpenMenu(openMenu === ws.id ? null : ws.id);
              }}
              className="w-7 h-7 flex items-center justify-center rounded-md text-muted-foreground opacity-0 group-hover:opacity-100 hover:bg-muted transition-all"
            >
              <MoreHorizontal className="w-4 h-4" />
            </button>
            {openMenu === ws.id && (
              <div className="absolute right-0 top-8 z-20 bg-card border rounded-lg shadow-lg py-1 w-32">
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    setEditWorkspace(ws);
                    setOpenMenu(null);
                  }}
                  className="w-full flex items-center gap-2 px-3 py-1.5 text-sm hover:bg-muted transition-colors"
                >
                  <Edit className="w-3.5 h-3.5" />
                  {t("common.edit")}
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    setDeleteConfirm(ws.id);
                    setOpenMenu(null);
                  }}
                  className="w-full flex items-center gap-2 px-3 py-1.5 text-sm text-destructive hover:bg-muted transition-colors"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                  {t("common.delete")}
                </button>
              </div>
            )}
          </div>
        </div>
        <div className="flex items-center gap-3 mt-3 text-xs text-muted-foreground">
          <span className="flex items-center gap-1">
            <FileText className="w-3 h-3" />
            {t("kb.docs_count", { count: ws.document_count })}
          </span>
          <span className="flex items-center gap-1 text-green-500">
            {t("kb.indexed_count", { count: ws.indexed_count })}
          </span>
          {ws.updated_at && (
            <>
              <span className="text-border">|</span>
              <span>{formatDate(ws.updated_at)}</span>
            </>
          )}
        </div>
      </CardContent>
    </Card>
  );

  const renderSection = (
    title: string,
    icon: React.ReactNode,
    items: KnowledgeBase[],
    emptyText?: string,
  ) => {
    if (items.length === 0 && !emptyText) return null;
    return (
      <div className="mb-8">
        <div className="flex items-center gap-2 mb-3">
          {icon}
          <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">
            {title}
          </h3>
          <span className="text-xs text-muted-foreground/60">({items.length})</span>
        </div>
        {items.length === 0 ? (
          <p className="text-sm text-muted-foreground/60 pl-6">{emptyText}</p>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {items.map(renderWorkspaceCard)}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="h-full flex flex-col overflow-hidden">
      {/* ── Header ── */}
      <div className="flex-shrink-0 border-b px-6 py-4">
        <div className="flex items-center gap-1.5 text-xs text-muted-foreground mb-2">
          <button 
            onClick={() => navigate("/")} 
            className="hover:text-foreground transition-colors underline-offset-4 hover:underline"
          >
            {t("nav.dashboard")}
          </button>
          <span>/</span>
          <span className="text-foreground font-medium">{t("kb.title")}</span>
        </div>
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-lg font-bold flex items-center gap-2">
              <Database className="w-5 h-5 text-primary" />
              {t("kb.title")}
            </h1>
            {workspaces && workspaces.length > 0 && (
              <p className="text-xs text-muted-foreground mt-0.5">
                {t(workspaces.length === 1 ? 'kb.kb_count' : 'kb.kb_count_plural', { count: workspaces.length })}
              </p>
            )}
          </div>
          <Button onClick={() => setShowNewWorkspace(true)} size="sm" className="shadow-sm shadow-primary/20">
            <Plus className="w-4 h-4 mr-1.5" />
            {t("kb.new")}
          </Button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto">
        <div className="max-w-5xl mx-auto px-6 py-8">

        {/* New Workspace Modal */}
        {showNewWorkspace && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
            <Card className="w-full max-w-md mx-4 shadow-2xl">
              <CardContent className="pt-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold">{t("kb.new")}</h3>
                  <button
                    onClick={() => setShowNewWorkspace(false)}
                    className="w-8 h-8 flex items-center justify-center rounded-lg text-muted-foreground hover:bg-muted transition-colors"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
                <Input
                  placeholder={t("kb.placeholder_name")}
                  value={newWorkspaceName}
                  onChange={(e) => setNewWorkspaceName(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleCreateWorkspace()}
                  autoFocus
                />
                {/* Visibility selector */}
                <div className="mt-4">
                  <label className="block text-sm font-medium mb-2">{t("kb.visibility")}</label>
                  <div className="grid grid-cols-3 gap-2">
                    {([
                      { value: "personal" as const, label: t("kb.personal"), icon: User, desc: t("kb.personal_desc") },
                      { value: "tenant" as const, label: t("kb.tenant"), icon: Building2, desc: t("kb.tenant_desc") },
                      { value: "public" as const, label: t("kb.public"), icon: Globe, desc: t("kb.public_desc") },
                    ]).map((opt) => (
                      <button
                        key={opt.value}
                        type="button"
                        onClick={() => setNewVisibility(opt.value)}
                        className={cn(
                          "flex flex-col items-center gap-1 p-3 rounded-lg border text-sm transition-colors",
                          newVisibility === opt.value
                            ? "border-primary bg-primary/5 text-primary"
                            : "border-border text-muted-foreground hover:bg-muted/50"
                        )}
                      >
                        <opt.icon className="w-4 h-4" />
                        <span className="font-medium text-xs">{opt.label}</span>
                        <span className="text-[10px] text-muted-foreground">{opt.desc}</span>
                      </button>
                    ))}
                  </div>
                </div>

                {/* Tenant selector — shown only when visibility = "tenant" */}
                {newVisibility === "tenant" && (
                  <div className="mt-3">
                    <label className="block text-sm font-medium mb-1.5">
                      {t("kb.organization")} <span className="text-destructive">*</span>
                    </label>
                    {!tenantsForDropdown || tenantsForDropdown.length === 0 ? (
                      <p className="text-xs text-muted-foreground bg-muted/50 rounded-lg px-3 py-2">
                        {isSuperadmin
                          ? t("kb.no_org_admin")
                          : t("kb.no_org_user")}
                      </p>
                    ) : (
                      <div className="relative">
                        <select
                          value={selectedTenantId ?? ""}
                          onChange={(e) => setSelectedTenantId(e.target.value ? Number(e.target.value) : null)}
                          className={cn(
                            "w-full appearance-none rounded-lg border bg-background px-3 py-2 pr-8 text-sm",
                            "focus:outline-none focus:ring-1 focus:ring-ring",
                            !selectedTenantId && "text-muted-foreground"
                          )}
                        >
                          <option value="">{t("kb.org_placeholder")}</option>
                          {tenantsForDropdown.map((t) => (
                            <option key={t.id} value={t.id}>{t.name}</option>
                          ))}
                        </select>
                        <ChevronDown className="pointer-events-none absolute right-2.5 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                      </div>
                    )}
                  </div>
                )}
                <div className="flex justify-end gap-2 mt-4">
                  <Button variant="ghost" onClick={() => setShowNewWorkspace(false)}>
                    {t("common.cancel")}
                  </Button>
                  <Button onClick={handleCreateWorkspace} disabled={createWorkspace.isPending || !newWorkspaceName.trim()}>
                    {createWorkspace.isPending ? t("common.creating") : t("common.create")}
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Edit Workspace Modal */}
        {editWorkspace && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
            <Card className="w-full max-w-md mx-4 shadow-2xl">
              <CardContent className="pt-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold">{t("kb.edit")}</h3>
                  <button
                    onClick={() => setEditWorkspace(null)}
                    className="w-8 h-8 flex items-center justify-center rounded-lg text-muted-foreground hover:bg-muted transition-colors"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
                <Input
                  placeholder={t("kb.placeholder_name")}
                  value={editWorkspaceName}
                  onChange={(e) => setEditWorkspaceName(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleUpdateWorkspace()}
                  autoFocus
                />
                <div className="mt-4">
                  <label className="block text-sm font-medium mb-2">{t("kb.visibility")}</label>
                  <div className="grid grid-cols-3 gap-2">
                    {([
                      { value: "personal" as const, label: t("kb.personal"), icon: User, desc: t("kb.personal_desc") },
                      { value: "tenant" as const, label: t("kb.tenant"), icon: Building2, desc: t("kb.tenant_desc") },
                      { value: "public" as const, label: t("kb.public"), icon: Globe, desc: t("kb.public_desc") },
                    ]).map((opt) => (
                      <button
                        key={opt.value}
                        type="button"
                        onClick={() => setEditVisibility(opt.value)}
                        className={cn(
                          "flex flex-col items-center gap-1 p-3 rounded-lg border text-sm transition-colors",
                          editVisibility === opt.value
                            ? "border-primary bg-primary/5 text-primary"
                            : "border-border text-muted-foreground hover:bg-muted/50"
                        )}
                      >
                        <opt.icon className="w-4 h-4" />
                        <span className="font-medium text-xs">{opt.label}</span>
                        <span className="text-[10px] text-muted-foreground">{opt.desc}</span>
                      </button>
                    ))}
                  </div>
                </div>

                {editVisibility === "tenant" && (
                  <div className="mt-3">
                    <label className="block text-sm font-medium mb-1.5">
                      {t("kb.organization")} <span className="text-destructive">*</span>
                    </label>
                    {!tenantsForDropdown || tenantsForDropdown.length === 0 ? (
                      <p className="text-xs text-muted-foreground bg-muted/50 rounded-lg px-3 py-2">
                        {isSuperadmin
                          ? t("kb.no_org_admin")
                          : t("kb.no_org_user")}
                      </p>
                    ) : (
                      <div className="relative">
                        <select
                          value={editTenantId ?? ""}
                          onChange={(e) => setEditTenantId(e.target.value ? Number(e.target.value) : null)}
                          className={cn(
                            "w-full appearance-none rounded-lg border bg-background px-3 py-2 pr-8 text-sm",
                            "focus:outline-none focus:ring-1 focus:ring-ring",
                            !editTenantId && "text-muted-foreground"
                          )}
                        >
                          <option value="">{t("kb.org_placeholder")}</option>
                          {tenantsForDropdown.map((t) => (
                            <option key={t.id} value={t.id}>{t.name}</option>
                          ))}
                        </select>
                        <ChevronDown className="pointer-events-none absolute right-2.5 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                      </div>
                    )}
                  </div>
                )}
                <div className="flex justify-end gap-2 mt-4">
                  <Button variant="ghost" onClick={() => setEditWorkspace(null)}>
                    {t("common.cancel")}
                  </Button>
                  <Button 
                    onClick={handleUpdateWorkspace} 
                    disabled={updateWorkspace.isPending || !editWorkspaceName.trim() || (editWorkspaceName === editWorkspace.name && editVisibility === editWorkspace.visibility && editTenantId === editWorkspace.tenant_id)}
                  >
                    {updateWorkspace.isPending ? t("common.saving") : t("common.save")}
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Loading skeleton */}
        {isLoading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {[1, 2].map((i) => (
              <Card key={i} className="animate-pulse">
                <CardContent className="pt-5 pb-4">
                  <div className="h-5 bg-muted rounded w-3/4 mb-3" />
                  <div className="h-3 bg-muted rounded w-1/2" />
                </CardContent>
              </Card>
            ))}
          </div>
        ) : !workspaces || workspaces.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-20">
            <div className="w-20 h-20 rounded-2xl bg-blue-500/10 flex items-center justify-center mb-6">
              <Database className="w-10 h-10 text-blue-500" />
            </div>
            <h3 className="text-xl font-semibold mb-2">{t("kb.empty_title")}</h3>
            <p className="text-muted-foreground text-center max-w-sm mb-6">
              {t("kb.empty_desc1")}<br />
              {t("kb.empty_desc2")}
            </p>
            <Button onClick={() => setShowNewWorkspace(true)} size="lg">
              <Plus className="w-4 h-4 mr-2" />
              {t("kb.new")}
            </Button>
          </div>
        ) : (
          <>
            {/* Legacy workspaces (no owner — from before auth) */}
            {legacyWorkspaces.length > 0 && renderSection(
              t("kb.section_legacy"),
              <Globe className="w-4 h-4 text-muted-foreground" />,
              legacyWorkspaces,
            )}

            {renderSection(
              t("kb.section_public"),
              <Globe className="w-4 h-4 text-blue-500" />,
              publicWorkspaces,
              t("kb.no_public"),
            )}

            {renderSection(
              t("kb.section_org"),
              <Building2 className="w-4 h-4 text-amber-500" />,
              tenantWorkspaces,
            )}

            {renderSection(
              t("kb.section_personal"),
              <User className="w-4 h-4 text-green-500" />,
              personalWorkspaces,
              t("kb.no_personal"),
            )}
          </>
        )}
      </div>

      {/* Delete confirmation */}
      <ConfirmDialog
        open={deleteConfirm !== null}
        onConfirm={() => deleteConfirm !== null && handleDeleteWorkspace(deleteConfirm)}
        onCancel={() => setDeleteConfirm(null)}
        title={t("kb.delete_confirm_title")}
        message={t("kb.delete_confirm_msg")}
        confirmLabel={t("common.delete")}
        variant="danger"
      />
      </div>
    </div>
  );
}
