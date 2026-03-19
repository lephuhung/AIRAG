import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";
import {
  Building2,
  Plus,
  Pencil,
  Users,
  Clock,
  X,
  Loader2,
  Power,
  ExternalLink,
  Link as LinkIcon,
  Copy,
  Trash2,
  Mail,
  Shield,
  UserPlus,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { ConfirmDialog } from "@/components/ui/confirm-dialog";
import { cn } from "@/lib/utils";
import {
  useAdminTenants,
  useCreateTenant,
  useUpdateTenant,
  useDeactivateTenant,
} from "@/hooks/useAdminTenants";
import {
  useCreateInvite,
  useTenantInvites,
  useRevokeInvite,
} from "@/hooks/useInvites";
import type { Tenant, InviteLink } from "@/types";

// ── Dialog Types ──────────────────────────────────────────────────────────

interface TenantFormData {
  name: string;
  slug: string;
  domain: string;
}

interface InviteFormData {
  email: string;
  role: string;
  expires_in_days: number;
  max_uses: string; // "" for unlimited, number string otherwise
}

const emptyForm: TenantFormData = { name: "", slug: "", domain: "" };
const emptyInviteForm: InviteFormData = {
  email: "",
  role: "member",
  expires_in_days: 7,
  max_uses: "",
};

export function AdminTenantsPage() {
  const navigate = useNavigate();
  const { data: tenants, isLoading } = useAdminTenants();
  const createTenant = useCreateTenant();
  const updateTenant = useUpdateTenant();
  const deactivateTenant = useDeactivateTenant();
  const createInvite = useCreateInvite();
  const revokeInvite = useRevokeInvite();

  // Dialog state
  const [showDialog, setShowDialog] = useState(false);
  const [editingTenant, setEditingTenant] = useState<Tenant | null>(null);
  const [form, setForm] = useState<TenantFormData>(emptyForm);
  const [confirmDeactivate, setConfirmDeactivate] = useState<Tenant | null>(null);

  // Invite dialog state
  const [inviteTenant, setInviteTenant] = useState<Tenant | null>(null);
  const [inviteForm, setInviteForm] = useState<InviteFormData>(emptyInviteForm);
  const [generatedInvite, setGeneratedInvite] = useState<InviteLink | null>(null);

  const { data: activeInvites } = useTenantInvites(inviteTenant?.id ?? null);

  const openCreate = () => {
    setEditingTenant(null);
    setForm(emptyForm);
    setShowDialog(true);
  };

  const openEdit = (tenant: Tenant) => {
    setEditingTenant(tenant);
    setForm({
      name: tenant.name,
      slug: tenant.slug,
      domain: tenant.domain || "",
    });
    setShowDialog(true);
  };

  const openInviteDialog = (tenant: Tenant) => {
    setInviteTenant(tenant);
    setInviteForm(emptyInviteForm);
    setGeneratedInvite(null);
  };

  const handleAutoSlug = (name: string) => {
    if (!editingTenant) {
      const slug = name
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "-")
        .replace(/^-|-$/g, "");
      setForm((f) => ({ ...f, name, slug }));
    } else {
      setForm((f) => ({ ...f, name }));
    }
  };

  const handleSubmit = async () => {
    if (!form.name.trim() || !form.slug.trim()) {
      toast.error("Name and slug are required");
      return;
    }

    try {
      if (editingTenant) {
        await updateTenant.mutateAsync({
          tenantId: editingTenant.id,
          data: {
            name: form.name,
            slug: form.slug,
            domain: form.domain || undefined,
          },
        });
        toast.success("Tenant updated");
      } else {
        await createTenant.mutateAsync({
          name: form.name,
          slug: form.slug,
          domain: form.domain || undefined,
        });
        toast.success("Tenant created");
      }
      setShowDialog(false);
    } catch (err: any) {
      toast.error(err.message || "Failed to save tenant");
    }
  };

  const handleDeactivate = async () => {
    if (!confirmDeactivate) return;
    try {
      await deactivateTenant.mutateAsync(confirmDeactivate.id);
      toast.success("Tenant deactivated");
      setConfirmDeactivate(null);
    } catch (err: any) {
      toast.error(err.message || "Failed to deactivate tenant");
    }
  };

  const handleCreateInvite = async () => {
    if (!inviteTenant) return;
    try {
      const result = await createInvite.mutateAsync({
        tenantId: inviteTenant.id,
        data: {
          email: inviteForm.email.trim() || undefined,
          role: inviteForm.role,
          expires_in_days: inviteForm.expires_in_days,
          max_uses: inviteForm.max_uses ? parseInt(inviteForm.max_uses) : undefined,
        },
      });
      setGeneratedInvite(result);
      toast.success("Invite link created");
    } catch (err: any) {
      toast.error(err.message || "Failed to create invite");
    }
  };

  const handleRevokeInvite = async (inviteId: number) => {
    if (!inviteTenant) return;
    try {
      await revokeInvite.mutateAsync({
        tenantId: inviteTenant.id,
        inviteId,
      });
      toast.success("Invite revoked");
      if (generatedInvite?.id === inviteId) {
        setGeneratedInvite(null);
      }
    } catch (err: any) {
      toast.error(err.message || "Failed to revoke invite");
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    toast.success("Copied to clipboard");
  };

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-5xl mx-auto px-6 py-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-amber-500/10 flex items-center justify-center">
              <Building2 className="w-5 h-5 text-amber-500" />
            </div>
            <div>
              <h1 className="text-xl font-bold">Tenant Management</h1>
              <p className="text-sm text-muted-foreground">
                Manage organizations and their members
              </p>
            </div>
          </div>
          <Button size="sm" onClick={openCreate}>
            <Plus className="w-4 h-4 mr-1" />
            Create Tenant
          </Button>
        </div>

        {/* Tenant Grid */}
        {isLoading ? (
          <div className="flex items-center justify-center py-20">
            <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
          </div>
        ) : !tenants || tenants.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-20 text-muted-foreground">
            <Building2 className="w-10 h-10 mb-3 opacity-30" />
            <p className="text-sm">No tenants yet</p>
            <Button
              size="sm"
              variant="ghost"
              className="mt-2"
              onClick={openCreate}
            >
              <Plus className="w-4 h-4 mr-1" />
              Create your first tenant
            </Button>
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {tenants.map((t) => (
              <div
                key={t.id}
                className={cn(
                  "rounded-xl border bg-card p-5 transition-colors hover:border-primary/30",
                  !t.is_active && "opacity-60",
                )}
              >
                {/* Title + status */}
                <div className="flex items-start justify-between mb-3">
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <h3 className="font-semibold truncate">{t.name}</h3>
                      {!t.is_active && (
                        <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-destructive/15 text-destructive font-medium">
                          Inactive
                        </span>
                      )}
                    </div>
                    <p className="text-xs text-muted-foreground mt-0.5">
                      /{t.slug}
                      {t.domain && (
                        <span className="ml-2 text-muted-foreground/60">
                          {t.domain}
                        </span>
                      )}
                    </p>
                  </div>
                </div>

                {/* Stats */}
                <div className="flex items-center gap-4 mb-4">
                  <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                    <Users className="w-3.5 h-3.5" />
                    <span>
                      <span className="font-medium text-foreground">
                        {t.member_count}
                      </span>{" "}
                      members
                    </span>
                  </div>
                  {t.pending_count > 0 && (
                    <div className="flex items-center gap-1.5 text-xs text-amber-500">
                      <Clock className="w-3.5 h-3.5" />
                      <span>
                        <span className="font-medium">{t.pending_count}</span>{" "}
                        pending
                      </span>
                    </div>
                  )}
                </div>

                {/* Actions */}
                <div className="flex items-center gap-2 border-t pt-3">
                  <Button
                    size="sm"
                    variant="ghost"
                    className="h-7 px-2 text-xs"
                    onClick={() => navigate(`/tenants/${t.id}`)}
                    title="Manage members"
                  >
                    <ExternalLink className="w-3.5 h-3.5 mr-1" />
                    Members
                  </Button>
                  <Button
                    size="sm"
                    variant="ghost"
                    className="h-7 px-2 text-xs"
                    onClick={() => openInviteDialog(t)}
                    title="Create invite link"
                  >
                    <LinkIcon className="w-3.5 h-3.5 mr-1" />
                    Invite
                  </Button>
                  <Button
                    size="sm"
                    variant="ghost"
                    className="h-7 px-2 text-xs"
                    onClick={() => openEdit(t)}
                    title="Edit tenant"
                  >
                    <Pencil className="w-3.5 h-3.5 mr-1" />
                    Edit
                  </Button>
                  {t.is_active && (
                    <Button
                      size="sm"
                      variant="ghost"
                      className="h-7 px-2 text-xs text-destructive hover:bg-destructive/10 ml-auto"
                      onClick={() => setConfirmDeactivate(t)}
                      title="Deactivate tenant"
                    >
                      <Power className="w-3.5 h-3.5" />
                    </Button>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Create / Edit Dialog */}
      {showDialog && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div
            className="absolute inset-0 bg-black/60 backdrop-blur-sm animate-in fade-in duration-200"
            onClick={() => setShowDialog(false)}
          />
          <div className="relative z-10 w-full max-w-md mx-4 rounded-xl bg-card border shadow-2xl animate-in zoom-in-95 fade-in duration-200">
            <div className="flex items-center justify-between p-5 border-b">
              <h3 className="font-semibold">
                {editingTenant ? "Edit Tenant" : "Create Tenant"}
              </h3>
              <button
                onClick={() => setShowDialog(false)}
                className="w-7 h-7 flex items-center justify-center rounded-lg text-muted-foreground hover:bg-muted transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            <div className="p-5 space-y-4">
              <div>
                <label className="text-xs font-medium text-muted-foreground block mb-1.5">
                  Name
                </label>
                <input
                  type="text"
                  value={form.name}
                  onChange={(e) => handleAutoSlug(e.target.value)}
                  placeholder="Engineering Team"
                  className="w-full px-3 py-2 text-sm rounded-lg border bg-background focus:outline-none focus:ring-2 focus:ring-primary/30"
                />
              </div>
              <div>
                <label className="text-xs font-medium text-muted-foreground block mb-1.5">
                  Slug
                </label>
                <input
                  type="text"
                  value={form.slug}
                  onChange={(e) =>
                    setForm((f) => ({ ...f, slug: e.target.value }))
                  }
                  placeholder="engineering-team"
                  className="w-full px-3 py-2 text-sm rounded-lg border bg-background focus:outline-none focus:ring-2 focus:ring-primary/30"
                />
              </div>
              <div>
                <label className="text-xs font-medium text-muted-foreground block mb-1.5">
                  Domain{" "}
                  <span className="text-muted-foreground/60">(optional)</span>
                </label>
                <input
                  type="text"
                  value={form.domain}
                  onChange={(e) =>
                    setForm((f) => ({ ...f, domain: e.target.value }))
                  }
                  placeholder="example.com"
                  className="w-full px-3 py-2 text-sm rounded-lg border bg-background focus:outline-none focus:ring-2 focus:ring-primary/30"
                />
              </div>
            </div>
            <div className="flex justify-end gap-2 p-5 border-t bg-muted/20 rounded-b-xl">
              <Button
                size="sm"
                variant="ghost"
                onClick={() => setShowDialog(false)}
              >
                Cancel
              </Button>
              <Button
                size="sm"
                onClick={handleSubmit}
                disabled={
                  createTenant.isPending || updateTenant.isPending
                }
              >
                {createTenant.isPending || updateTenant.isPending ? (
                  <Loader2 className="w-4 h-4 animate-spin mr-1" />
                ) : null}
                {editingTenant ? "Save" : "Create"}
              </Button>
            </div>
          </div>
        </div>
      )}

      {/* Invite Dialog */}
      {inviteTenant && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div
            className="absolute inset-0 bg-black/60 backdrop-blur-sm animate-in fade-in duration-200"
            onClick={() => setInviteTenant(null)}
          />
          <div className="relative z-10 w-full max-w-lg mx-4 rounded-xl bg-card border shadow-2xl animate-in zoom-in-95 fade-in duration-200 max-h-[90vh] flex flex-col">
            <div className="flex items-center justify-between p-5 border-b">
              <div>
                <h3 className="font-semibold flex items-center gap-2">
                  <UserPlus className="w-4 h-4" />
                  Invite to {inviteTenant.name}
                </h3>
                <p className="text-xs text-muted-foreground mt-0.5">
                  Create an invite link for new members
                </p>
              </div>
              <button
                onClick={() => setInviteTenant(null)}
                className="w-7 h-7 flex items-center justify-center rounded-lg text-muted-foreground hover:bg-muted transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="p-5 space-y-4 overflow-y-auto flex-1">
              {/* Create invite form */}
              <div className="space-y-3">
                <div>
                  <label className="text-xs font-medium text-muted-foreground block mb-1.5">
                    Email{" "}
                    <span className="text-muted-foreground/60">(optional — lock to specific email)</span>
                  </label>
                  <input
                    type="email"
                    value={inviteForm.email}
                    onChange={(e) =>
                      setInviteForm((f) => ({ ...f, email: e.target.value }))
                    }
                    placeholder="user@example.com"
                    className="w-full px-3 py-2 text-sm rounded-lg border bg-background focus:outline-none focus:ring-2 focus:ring-primary/30"
                  />
                </div>

                <div className="grid grid-cols-3 gap-3">
                  <div>
                    <label className="text-xs font-medium text-muted-foreground block mb-1.5">
                      Role
                    </label>
                    <select
                      value={inviteForm.role}
                      onChange={(e) =>
                        setInviteForm((f) => ({ ...f, role: e.target.value }))
                      }
                      className="w-full px-3 py-2 text-sm rounded-lg border bg-background focus:outline-none focus:ring-2 focus:ring-primary/30"
                    >
                      <option value="member">Member</option>
                      <option value="admin">Admin</option>
                    </select>
                  </div>
                  <div>
                    <label className="text-xs font-medium text-muted-foreground block mb-1.5">
                      Expires (days)
                    </label>
                    <input
                      type="number"
                      min={1}
                      max={90}
                      value={inviteForm.expires_in_days}
                      onChange={(e) =>
                        setInviteForm((f) => ({
                          ...f,
                          expires_in_days: parseInt(e.target.value) || 7,
                        }))
                      }
                      className="w-full px-3 py-2 text-sm rounded-lg border bg-background focus:outline-none focus:ring-2 focus:ring-primary/30"
                    />
                  </div>
                  <div>
                    <label className="text-xs font-medium text-muted-foreground block mb-1.5">
                      Max uses
                    </label>
                    <input
                      type="number"
                      min={1}
                      value={inviteForm.max_uses}
                      onChange={(e) =>
                        setInviteForm((f) => ({
                          ...f,
                          max_uses: e.target.value,
                        }))
                      }
                      placeholder="Unlimited"
                      className="w-full px-3 py-2 text-sm rounded-lg border bg-background focus:outline-none focus:ring-2 focus:ring-primary/30"
                    />
                  </div>
                </div>

                <Button
                  size="sm"
                  onClick={handleCreateInvite}
                  disabled={createInvite.isPending}
                  className="w-full"
                >
                  {createInvite.isPending ? (
                    <Loader2 className="w-4 h-4 animate-spin mr-1" />
                  ) : (
                    <LinkIcon className="w-4 h-4 mr-1" />
                  )}
                  Generate Invite Link
                </Button>
              </div>

              {/* Generated invite link */}
              {generatedInvite && (
                <div className="p-3 rounded-lg bg-emerald-500/10 border border-emerald-500/20">
                  <p className="text-xs font-medium text-emerald-700 dark:text-emerald-400 mb-2">
                    Invite link created!
                  </p>
                  <div className="flex items-center gap-2">
                    <input
                      type="text"
                      readOnly
                      value={generatedInvite.invite_url}
                      className="flex-1 px-3 py-1.5 text-xs rounded-md border bg-background font-mono truncate"
                    />
                    <Button
                      size="sm"
                      variant="ghost"
                      className="h-8 px-2 flex-shrink-0"
                      onClick={() => copyToClipboard(generatedInvite.invite_url)}
                    >
                      <Copy className="w-3.5 h-3.5" />
                    </Button>
                  </div>
                </div>
              )}

              {/* Active invites list */}
              {activeInvites && activeInvites.length > 0 && (
                <div>
                  <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">
                    Active Invites ({activeInvites.length})
                  </h4>
                  <div className="space-y-2">
                    {activeInvites.map((inv) => {
                      const expired = new Date(inv.expires_at) < new Date();
                      const maxedOut =
                        inv.max_uses !== null && inv.use_count >= inv.max_uses;
                      return (
                        <div
                          key={inv.id}
                          className={cn(
                            "flex items-center justify-between p-2.5 rounded-lg border text-xs",
                            (expired || maxedOut) && "opacity-60",
                          )}
                        >
                          <div className="min-w-0 flex-1">
                            <div className="flex items-center gap-2 mb-0.5">
                              {inv.email ? (
                                <span className="flex items-center gap-1 text-muted-foreground">
                                  <Mail className="w-3 h-3" />
                                  {inv.email}
                                </span>
                              ) : (
                                <span className="text-muted-foreground">
                                  Anyone
                                </span>
                              )}
                              <span className="flex items-center gap-0.5 text-muted-foreground/60">
                                <Shield className="w-3 h-3" />
                                {inv.role}
                              </span>
                            </div>
                            <div className="flex items-center gap-3 text-muted-foreground/60">
                              <span>
                                Uses: {inv.use_count}
                                {inv.max_uses !== null ? `/${inv.max_uses}` : ""}
                              </span>
                              <span>
                                Expires:{" "}
                                {new Date(inv.expires_at).toLocaleDateString()}
                              </span>
                              {expired && (
                                <span className="text-destructive font-medium">
                                  Expired
                                </span>
                              )}
                              {maxedOut && (
                                <span className="text-amber-500 font-medium">
                                  Max reached
                                </span>
                              )}
                            </div>
                          </div>
                          <div className="flex items-center gap-1 flex-shrink-0">
                            <Button
                              size="sm"
                              variant="ghost"
                              className="h-7 w-7 p-0"
                              onClick={() =>
                                copyToClipboard(inv.invite_url)
                              }
                              title="Copy link"
                            >
                              <Copy className="w-3.5 h-3.5" />
                            </Button>
                            <Button
                              size="sm"
                              variant="ghost"
                              className="h-7 w-7 p-0 text-destructive hover:bg-destructive/10"
                              onClick={() => handleRevokeInvite(inv.id)}
                              title="Revoke invite"
                            >
                              <Trash2 className="w-3.5 h-3.5" />
                            </Button>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>

            <div className="flex justify-end p-5 border-t bg-muted/20 rounded-b-xl">
              <Button
                size="sm"
                variant="ghost"
                onClick={() => setInviteTenant(null)}
              >
                Close
              </Button>
            </div>
          </div>
        </div>
      )}

      {/* Deactivate confirmation */}
      <ConfirmDialog
        open={!!confirmDeactivate}
        onCancel={() => setConfirmDeactivate(null)}
        onConfirm={handleDeactivate}
        title="Deactivate Tenant"
        message={`Are you sure you want to deactivate "${confirmDeactivate?.name}"? Members will lose access to tenant workspaces.`}
        confirmLabel="Deactivate"
        variant="danger"
      />
    </div>
  );
}
