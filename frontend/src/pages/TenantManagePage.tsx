import { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { toast } from "sonner";
import { api } from "@/lib/api";
import { useAuthStore } from "@/stores/authStore";
import { useTranslation } from "@/hooks/useTranslation";
import {
  useCreateInvite,
  useTenantInvites,
  useRevokeInvite,
} from "@/hooks/useInvites";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import {
  Building2,
  Check,
  X,
  Trash2,
  Shield,
  User,
  ArrowLeft,
  UserPlus,
  Link as LinkIcon,
  Copy,
  Mail,
  Loader2,
} from "lucide-react";
import { cn } from "@/lib/utils";
import type { Tenant, TenantUser } from "@/types";

interface InviteFormData {
  email: string;
  role: string;
  expires_in_days: number;
  max_uses: string;
}

const emptyInviteForm: InviteFormData = {
  email: "",
  role: "member",
  expires_in_days: 7,
  max_uses: "",
};

export function TenantManagePage() {
  const { t } = useTranslation();
  const { tenantId } = useParams<{ tenantId: string }>();
  const navigate = useNavigate();
  const user = useAuthStore((s) => s.user);
  const [tenant, setTenant] = useState<Tenant | null>(null);
  const [members, setMembers] = useState<TenantUser[]>([]);
  const [loading, setLoading] = useState(true);

  // Invite state
  const [showInviteDialog, setShowInviteDialog] = useState(false);
  const [inviteForm, setInviteForm] = useState<InviteFormData>(emptyInviteForm);
  const createInvite = useCreateInvite();
  const revokeInvite = useRevokeInvite();

  const { data: activeInvites } = useTenantInvites(showInviteDialog ? tenantId ?? null : null);

  useEffect(() => {
    if (!tenantId) return;
    loadData();
  }, [tenantId]);

  const loadData = async () => {
    if (!tenantId) return;
    try {
      const [tenantData, memberList] = await Promise.all([
        api.get<Tenant>(`/tenants/${tenantId}`),
        api.get<TenantUser[]>(`/tenants/${tenantId}/users`),
      ]);
      setTenant(tenantData);
      setMembers(memberList);
    } catch (err: any) {
      toast.error(err.message || t("admin.tenants.manage.loading"));
    } finally {
      setLoading(false);
    }
  };

  const handleApprove = async (userId: number) => {
    try {
      await api.post(`/tenants/${tenantId}/users/${userId}/approve`);
      toast.success(t("admin.tenants.manage.approve_success"));
      loadData();
    } catch (err: any) {
      toast.error(err.message || t("common.error"));
    }
  };

  const handleReject = async (userId: number) => {
    try {
      await api.post(`/tenants/${tenantId}/users/${userId}/reject`);
      toast.success(t("admin.tenants.manage.reject_success"));
      loadData();
    } catch (err: any) {
      toast.error(err.message || t("common.error"));
    }
  };

  const handleRemove = async (userId: number) => {
    try {
      await api.delete(`/tenants/${tenantId}/users/${userId}`);
      toast.success(t("admin.tenants.manage.remove_success"));
      loadData();
    } catch (err: any) {
      toast.error(err.message || t("common.error"));
    }
  };

  const handleToggleRole = async (userId: number, currentRole: string) => {
    const newRole = currentRole === "admin" ? "member" : "admin";
    try {
      await api.put(`/tenants/${tenantId}/users/${userId}/role`, { role: newRole });
      toast.success(t("admin.tenants.manage.role_changed", { role: newRole }));
      loadData();
    } catch (err: any) {
      toast.error(err.message || t("common.error"));
    }
  };

  const handleCreateInvite = async () => {
    try {
      const result = await createInvite.mutateAsync({
        tenantId: tenantId ?? "",
        data: {
          email: inviteForm.email.trim() || undefined,
          role: inviteForm.role,
          expires_in_days: inviteForm.expires_in_days,
          max_uses: inviteForm.max_uses ? parseInt(inviteForm.max_uses) : undefined,
        },
      });
      navigator.clipboard.writeText(result.invite_url);
      toast.success(t("admin.tenants.toast.invite_created"));
      setInviteForm(emptyInviteForm);
    } catch (err: any) {
      toast.error(err.message || t("admin.tenants.toast.invite_failed"));
    }
  };

  const handleRevokeInvite = async (inviteId: number) => {
    try {
      await revokeInvite.mutateAsync({ tenantId: tenantId ?? "", inviteId });
      toast.success(t("admin.tenants.toast.invite_revoked"));
    } catch (err: any) {
      toast.error(err.message || t("admin.tenants.toast.invite_revoke_failed"));
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    toast.success(t("admin.tenants.toast.copied"));
  };

  if (loading) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="animate-pulse text-muted-foreground">{t("admin.tenants.manage.loading")}</div>
      </div>
    );
  }

  const pendingMembers = members.filter((m) => !m.is_approved);
  const approvedMembers = members.filter((m) => m.is_approved);

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-3xl mx-auto px-6 py-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <button
              onClick={() => navigate("/admin/tenants")}
              className="w-8 h-8 flex items-center justify-center rounded-lg text-muted-foreground hover:bg-muted transition-colors"
            >
              <ArrowLeft className="w-4 h-4" />
            </button>
            <div className="flex items-center gap-2">
              <Building2 className="w-5 h-5 text-amber-500" />
              <h2 className="text-lg font-semibold">
                {tenant?.name || `Tenant ${tenantId ?? ""}`}
              </h2>
            </div>
          </div>
          <Button
            size="sm"
            onClick={() => {
              setInviteForm(emptyInviteForm);
              setShowInviteDialog(true);
            }}
          >
            <UserPlus className="w-4 h-4 mr-1" />
            {t("admin.tenants.manage.invite_btn")}
          </Button>
        </div>

        {/* Pending approvals */}
        {pendingMembers.length > 0 && (
          <div className="mb-8">
            <h3 className="text-sm font-semibold text-amber-500 uppercase tracking-wider mb-3">
              {t("admin.tenants.manage.pending")} ({pendingMembers.length})
            </h3>
            <div className="space-y-2">
              {pendingMembers.map((m) => (
                <Card key={m.id} className="border-amber-500/20">
                  <CardContent className="py-3 flex items-center justify-between">
                    <div className="min-w-0">
                      <p className="text-sm font-medium truncate">{m.full_name || t("admin.tenants.manage.unknown_user")}</p>
                      <p className="text-xs text-muted-foreground truncate">{m.email}</p>
                    </div>
                    <div className="flex items-center gap-2">
                      <Button
                        size="sm"
                        variant="ghost"
                        className="text-green-600 hover:bg-green-500/10"
                        onClick={() => handleApprove(m.user_id)}
                      >
                        <Check className="w-4 h-4 mr-1" />
                        {t("admin.tenants.manage.approve")}
                      </Button>
                      <Button
                        size="sm"
                        variant="ghost"
                        className="text-destructive hover:bg-destructive/10"
                        onClick={() => handleReject(m.user_id)}
                      >
                        <X className="w-4 h-4 mr-1" />
                        {t("admin.tenants.manage.reject")}
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        )}

        {/* Approved members */}
        <div>
          <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-3">
            {t("admin.tenants.manage.members")} ({approvedMembers.length})
          </h3>
          {approvedMembers.length === 0 ? (
            <p className="text-sm text-muted-foreground">{t("admin.tenants.manage.no_members")}</p>
          ) : (
            <div className="space-y-2">
              {approvedMembers.map((m) => (
                <Card key={m.id}>
                  <CardContent className="py-3 flex items-center justify-between">
                    <div className="flex items-center gap-3 min-w-0">
                      <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center text-xs font-semibold text-primary flex-shrink-0">
                        {(m.full_name || "?")[0].toUpperCase()}
                      </div>
                      <div className="min-w-0">
                        <div className="flex items-center gap-1.5">
                          <p className="text-sm font-medium truncate">{m.full_name || t("admin.tenants.manage.unknown_user")}</p>
                          {m.role === "admin" && (
                            <span title={t("admin.tenants.admin_role")}>
                              <Shield className="w-3.5 h-3.5 text-amber-500 flex-shrink-0" />
                            </span>
                          )}
                        </div>
                        <p className="text-xs text-muted-foreground truncate">{m.email}</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      {/* Don't allow removing yourself */}
                      {m.user_id !== user?.id && (
                        <>
                          <Button
                            size="sm"
                            variant="ghost"
                            className="text-xs"
                            onClick={() => handleToggleRole(m.user_id, m.role)}
                            title={m.role === "admin" ? t("admin.tenants.manage.demote") : t("admin.tenants.manage.promote")}
                          >
                            {m.role === "admin" ? (
                              <>
                                <User className="w-3.5 h-3.5 mr-1" />
                                {t("admin.tenants.manage.demote")}
                              </>
                            ) : (
                              <>
                                <Shield className="w-3.5 h-3.5 mr-1" />
                                {t("admin.tenants.manage.promote")}
                              </>
                            )}
                          </Button>
                          <Button
                            size="sm"
                            variant="ghost"
                            className="text-destructive hover:bg-destructive/10"
                            onClick={() => handleRemove(m.user_id)}
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </Button>
                        </>
                      )}
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Invite Dialog */}
      {showInviteDialog && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div
            className="absolute inset-0 bg-black/60 backdrop-blur-sm animate-in fade-in duration-200"
            onClick={() => setShowInviteDialog(false)}
          />
          <div className="relative z-10 w-full max-w-lg mx-4 rounded-xl bg-card border shadow-2xl animate-in zoom-in-95 fade-in duration-200 max-h-[90vh] flex flex-col">
            <div className="flex items-center justify-between p-5 border-b">
              <div>
                <h3 className="font-semibold flex items-center gap-2">
                  <UserPlus className="w-4 h-4" />
                  {t("admin.tenants.invite_dialog.title", { name: tenant?.name })}
                </h3>
                <p className="text-xs text-muted-foreground mt-0.5">
                  {t("admin.tenants.invite_dialog.subtitle")}
                </p>
              </div>
              <button
                onClick={() => setShowInviteDialog(false)}
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
                    {t("common.email")}{" "}
                    <span className="text-muted-foreground/60">({t("admin.tenants.invite_dialog.email_lock")})</span>
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
                      {t("admin.tenants.invite_dialog.role")}
                    </label>
                    <select
                      value={inviteForm.role}
                      onChange={(e) =>
                        setInviteForm((f) => ({ ...f, role: e.target.value }))
                      }
                      className="w-full px-3 py-2 text-sm rounded-lg border bg-background focus:outline-none focus:ring-2 focus:ring-primary/30"
                    >
                      <option value="member">{t("admin.tenants.member_role")}</option>
                      <option value="admin">{t("admin.tenants.admin_role")}</option>
                    </select>
                  </div>
                  <div>
                    <label className="text-xs font-medium text-muted-foreground block mb-1.5">
                      {t("admin.tenants.invite_dialog.expires_days")}
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
                      {t("admin.tenants.invite_dialog.max_uses")}
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
                      placeholder={t("admin.tenants.invite_dialog.unlimited")}
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
                  {t("admin.tenants.invite_dialog.generate_btn")}
                </Button>
              </div>

              {/* Active invites list */}
              {activeInvites && activeInvites.length > 0 && (
                <div>
                  <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">
                    {t("admin.tenants.invite_dialog.active_invites", { count: activeInvites.length })}
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
                                  {t("admin.tenants.invite_dialog.anyone")}
                                </span>
                              )}
                              <span className="flex items-center gap-0.5 text-muted-foreground/60">
                                <Shield className="w-3 h-3" />
                                {inv.role}
                              </span>
                            </div>
                            <div className="flex items-center gap-3 text-muted-foreground/60">
                              <span>
                                {t("admin.tenants.invite_dialog.uses", { count: inv.use_count })}
                                {inv.max_uses !== null ? `/${inv.max_uses}` : ""}
                              </span>
                              <span>
                                {t("admin.tenants.invite_dialog.expires", { date: new Date(inv.expires_at).toLocaleDateString() })}
                              </span>
                              {expired && (
                                <span className="text-destructive font-medium">
                                  {t("admin.tenants.invite_dialog.expired_badge")}
                                </span>
                              )}
                              {maxedOut && (
                                <span className="text-amber-500 font-medium">
                                  {t("admin.tenants.invite_dialog.max_reached_badge")}
                                </span>
                              )}
                            </div>
                          </div>
                          <div className="flex items-center gap-1 flex-shrink-0">
                            <Button
                              size="sm"
                              variant="ghost"
                              className="h-7 w-7 p-0"
                              onClick={() => copyToClipboard(inv.invite_url)}
                              title={t("admin.tenants.toast.copied")}
                            >
                              <Copy className="w-3.5 h-3.5" />
                            </Button>
                            <Button
                              size="sm"
                              variant="ghost"
                              className="h-7 w-7 p-0 text-destructive hover:bg-destructive/10"
                              onClick={() => handleRevokeInvite(inv.id)}
                              title={t("admin.tenants.invite_dialog.revoke_title")}
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
                onClick={() => setShowInviteDialog(false)}
              >
                {t("common.close")}
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
