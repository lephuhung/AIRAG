import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";
import { useWorkspaces, useCreateWorkspace, useDeleteWorkspace } from "@/hooks/useWorkspaces";
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
} from "lucide-react";
import { ConfirmDialog } from "@/components/ui/confirm-dialog";
import { cn } from "@/lib/utils";
import type { KnowledgeBase, CreateWorkspace } from "@/types";

type VisibilityOption = "personal" | "tenant" | "public";

export function KnowledgeBasesPage() {
  const navigate = useNavigate();
  const { data: workspaces, isLoading } = useWorkspaces();
  const { data: myTenants } = useMyTenants();
  const { data: allTenants } = useAdminTenants();
  const isSuperadmin = useAuthStore((s) => s.user?.is_superadmin ?? false);
  // Superadmin sees all tenants; regular users see only their own
  const tenantsForDropdown = isSuperadmin ? allTenants : myTenants;
  const createWorkspace = useCreateWorkspace();
  const deleteWorkspace = useDeleteWorkspace();
  const [showNewWorkspace, setShowNewWorkspace] = useState(false);
  const [newWorkspaceName, setNewWorkspaceName] = useState("");
  const [newVisibility, setNewVisibility] = useState<VisibilityOption>("personal");
  const [selectedTenantId, setSelectedTenantId] = useState<number | null>(null);
  const [deleteConfirm, setDeleteConfirm] = useState<number | null>(null);
  const [openMenu, setOpenMenu] = useState<number | null>(null);

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
      toast.error("Please select an organization for this workspace");
      return;
    }
    try {
      const payload: CreateWorkspace = {
        name: newWorkspaceName,
        visibility: newVisibility,
        tenant_id: newVisibility === "tenant" ? selectedTenantId : undefined,
      };
      const ws = await createWorkspace.mutateAsync(payload);
      toast.success("Knowledge base created");
      setNewWorkspaceName("");
      setNewVisibility("personal");
      setSelectedTenantId(null);
      setShowNewWorkspace(false);
      navigate(`/knowledge-bases/${ws.id}`);
    } catch {
      toast.error("Failed to create knowledge base");
    }
  };

  const handleDeleteWorkspace = async (id: number) => {
    try {
      await deleteWorkspace.mutateAsync(id);
      toast.success("Knowledge base deleted");
    } catch {
      toast.error("Failed to delete knowledge base");
    }
    setDeleteConfirm(null);
  };

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));
    if (days === 0) return "Today";
    if (days === 1) return "Yesterday";
    if (days < 7) return `${days} days ago`;
    return date.toLocaleDateString("en-US", { month: "short", day: "numeric" });
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
                    setDeleteConfirm(ws.id);
                    setOpenMenu(null);
                  }}
                  className="w-full flex items-center gap-2 px-3 py-1.5 text-sm text-destructive hover:bg-muted transition-colors"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                  Delete
                </button>
              </div>
            )}
          </div>
        </div>
        <div className="flex items-center gap-3 mt-3 text-xs text-muted-foreground">
          <span className="flex items-center gap-1">
            <FileText className="w-3 h-3" />
            {ws.document_count} docs
          </span>
          <span className="flex items-center gap-1 text-green-500">
            {ws.indexed_count} indexed
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
    <div className="h-full overflow-y-auto">
      <div className="max-w-5xl mx-auto px-6 py-8">
        {/* Breadcrumb */}
        <div className="flex items-center gap-1.5 text-xs text-muted-foreground mb-4">
          <button onClick={() => navigate("/")} className="hover:text-foreground transition-colors">
            Dashboard
          </button>
          <span>/</span>
          <span className="text-foreground font-medium">Knowledge Bases</span>
        </div>

        {/* Section header + action */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-lg font-bold flex items-center gap-2">
              <Database className="w-5 h-5 text-primary" />
              Knowledge Bases
            </h1>
            {workspaces && workspaces.length > 0 && (
              <p className="text-xs text-muted-foreground mt-0.5">
                {workspaces.length} knowledge base{workspaces.length !== 1 ? "s" : ""}
              </p>
            )}
          </div>
          <Button onClick={() => setShowNewWorkspace(true)} size="sm">
            <Plus className="w-4 h-4 mr-1.5" />
            New Knowledge Base
          </Button>
        </div>

        {/* New Workspace Modal */}
        {showNewWorkspace && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
            <Card className="w-full max-w-md mx-4 shadow-2xl">
              <CardContent className="pt-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold">New Knowledge Base</h3>
                  <button
                    onClick={() => setShowNewWorkspace(false)}
                    className="w-8 h-8 flex items-center justify-center rounded-lg text-muted-foreground hover:bg-muted transition-colors"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
                <Input
                  placeholder="Knowledge base name"
                  value={newWorkspaceName}
                  onChange={(e) => setNewWorkspaceName(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleCreateWorkspace()}
                  autoFocus
                />
                {/* Visibility selector */}
                <div className="mt-4">
                  <label className="block text-sm font-medium mb-2">Visibility</label>
                  <div className="grid grid-cols-3 gap-2">
                    {([
                      { value: "personal" as const, label: "Personal", icon: User, desc: "Only you" },
                      { value: "tenant" as const, label: "Tenant", icon: Building2, desc: "Your org" },
                      { value: "public" as const, label: "Public", icon: Globe, desc: "Everyone" },
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
                      Organization <span className="text-destructive">*</span>
                    </label>
                    {!tenantsForDropdown || tenantsForDropdown.length === 0 ? (
                      <p className="text-xs text-muted-foreground bg-muted/50 rounded-lg px-3 py-2">
                        {isSuperadmin
                          ? "No organizations exist yet. Create one first."
                          : "You're not a member of any organization. Join or create one first."}
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
                          <option value="">Select organization…</option>
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
                    Cancel
                  </Button>
                  <Button onClick={handleCreateWorkspace} disabled={createWorkspace.isPending || !newWorkspaceName.trim()}>
                    {createWorkspace.isPending ? "Creating..." : "Create"}
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
            <h3 className="text-xl font-semibold mb-2">Create your first knowledge base</h3>
            <p className="text-muted-foreground text-center max-w-sm mb-6">
              Knowledge bases store your documents and enable AI-powered search across them.
              Link them to any project as a data source.
            </p>
            <Button onClick={() => setShowNewWorkspace(true)} size="lg">
              <Plus className="w-4 h-4 mr-2" />
              New Knowledge Base
            </Button>
          </div>
        ) : (
          <>
            {/* Legacy workspaces (no owner — from before auth) */}
            {legacyWorkspaces.length > 0 && renderSection(
              "Shared (Legacy)",
              <Globe className="w-4 h-4 text-muted-foreground" />,
              legacyWorkspaces,
            )}

            {renderSection(
              "Public Workspaces",
              <Globe className="w-4 h-4 text-blue-500" />,
              publicWorkspaces,
              "No public workspaces",
            )}

            {renderSection(
              "Organization",
              <Building2 className="w-4 h-4 text-amber-500" />,
              tenantWorkspaces,
            )}

            {renderSection(
              "My Personal Workspaces",
              <User className="w-4 h-4 text-green-500" />,
              personalWorkspaces,
              "No personal workspaces yet",
            )}
          </>
        )}
      </div>

      {/* Delete confirmation */}
      <ConfirmDialog
        open={deleteConfirm !== null}
        onConfirm={() => deleteConfirm !== null && handleDeleteWorkspace(deleteConfirm)}
        onCancel={() => setDeleteConfirm(null)}
        title="Delete Knowledge Base"
        message="Are you sure? All documents, indexed data, and knowledge graph data will be permanently removed."
        confirmLabel="Delete"
        variant="danger"
      />
    </div>
  );
}
