import { useState } from "react";
import { toast } from "sonner";
import {
  Users,
  Shield,
  Search,
  ChevronLeft,
  ChevronRight,
  Trash2,
  UserCheck,
  UserX,
  Crown,
  Loader2,
  KeyRound,
  Eye,
  EyeOff,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { ConfirmDialog } from "@/components/ui/confirm-dialog";
import { cn } from "@/lib/utils";
import { useAuthStore } from "@/stores/authStore";
import {
  useAdminUsers,
  useAdminStats,
  useUpdateUser,
  useDeleteUser,
  useResetUserPassword,
  useUpdateTenantMemberRole,
} from "@/hooks/useAdminUsers";
import { useAdminTenants } from "@/hooks/useAdminTenants";
import type { AdminUserDetail } from "@/types";

type FilterStatus = "all" | "active" | "inactive";

export function AdminUsersPage() {
  const currentUser = useAuthStore((s) => s.user);
  const [search, setSearch] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useState("");
  const [filterStatus, setFilterStatus] = useState<FilterStatus>("all");
  const [filterTenant, setFilterTenant] = useState<number | "all">("all");
  const [page, setPage] = useState(1);
  const perPage = 20;

  // Debounce search
  const [searchTimeout, setSearchTimeout] = useState<ReturnType<typeof setTimeout> | null>(null);
  const handleSearchChange = (value: string) => {
    setSearch(value);
    if (searchTimeout) clearTimeout(searchTimeout);
    const timeout = setTimeout(() => {
      setDebouncedSearch(value);
      setPage(1);
    }, 300);
    setSearchTimeout(timeout);
  };

  const isActiveFilter = filterStatus === "all" ? null : filterStatus === "active";
  const tenantIdFilter = filterTenant === "all" ? null : filterTenant;

  const { data, isLoading } = useAdminUsers(
    debouncedSearch || undefined,
    isActiveFilter,
    tenantIdFilter,
    page,
    perPage,
  );
  const { data: stats } = useAdminStats();
  const { data: tenants } = useAdminTenants();
  const updateUser = useUpdateUser();
  const deleteUser = useDeleteUser();
  const resetPassword = useResetUserPassword();
  const updateRole = useUpdateTenantMemberRole();

  const [confirmDelete, setConfirmDelete] = useState<AdminUserDetail | null>(null);
  const [resetTarget, setResetTarget] = useState<AdminUserDetail | null>(null);
  const [resetNewPassword, setResetNewPassword] = useState("");
  const [resetConfirm, setResetConfirm] = useState("");
  const [showResetPw, setShowResetPw] = useState(false);

  const totalPages = data ? Math.ceil(data.total / perPage) : 1;

  const handleToggleActive = async (user: AdminUserDetail) => {
    try {
      await updateUser.mutateAsync({
        userId: user.id,
        data: { is_active: !user.is_active },
      });
      toast.success(user.is_active ? "User deactivated" : "User activated");
    } catch (err: any) {
      toast.error(err.message || "Failed to update user");
    }
  };

  const handleToggleSuperadmin = async (user: AdminUserDetail) => {
    try {
      await updateUser.mutateAsync({
        userId: user.id,
        data: { is_superadmin: !user.is_superadmin },
      });
      toast.success(
        user.is_superadmin ? "Superadmin removed" : "Superadmin granted",
      );
    } catch (err: any) {
      toast.error(err.message || "Failed to update user");
    }
  };

  const handleConfirmDelete = async () => {
    if (!confirmDelete) return;
    try {
      await deleteUser.mutateAsync(confirmDelete.id);
      toast.success("User deleted");
      setConfirmDelete(null);
    } catch (err: any) {
      toast.error(err.message || "Failed to delete user");
    }
  };

  const handleToggleTenantRole = async (userId: number, tenantId: number, currentRole: string) => {
    if (isSelf(userId)) {
      toast.error("Cannot change your own tenant role here");
      return;
    }
    const newRole = currentRole === "admin" ? "member" : "admin";
    try {
      await updateRole.mutateAsync({ tenantId, userId, role: newRole as "admin" | "member" });
      toast.success(`Role changed to ${newRole}`);
    } catch (err: any) {
      toast.error(err.message || "Failed to change role");
    }
  };

  const handleResetPassword = async () => {
    if (!resetTarget) return;
    if (!resetNewPassword || resetNewPassword.length < 6) {
      toast.error("New password must be at least 6 characters");
      return;
    }
    if (resetNewPassword !== resetConfirm) {
      toast.error("Passwords do not match");
      return;
    }
    try {
      await resetPassword.mutateAsync({
        userId: resetTarget.id,
        newPassword: resetNewPassword,
      });
      toast.success(`Password reset for ${resetTarget.full_name}`);
      setResetTarget(null);
      setResetNewPassword("");
      setResetConfirm("");
    } catch (err: any) {
      toast.error(err.message || "Failed to reset password");
    }
  };

  const isSelf = (userId: number) => userId === currentUser?.id;

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-5xl mx-auto px-6 py-8">
        {/* Header */}
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center">
            <Users className="w-5 h-5 text-primary" />
          </div>
          <div>
            <h1 className="text-xl font-bold">User Management</h1>
            <p className="text-sm text-muted-foreground">
              Manage all system users
            </p>
          </div>
        </div>

        {/* Stat cards */}
        {stats && (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-6">
            <StatCard label="Total Users" value={stats.total_users} />
            <StatCard
              label="Active"
              value={stats.active_users}
              color="text-green-500"
            />
            <StatCard
              label="Pending"
              value={stats.pending_users}
              color="text-amber-500"
            />
            <StatCard
              label="Tenants"
              value={stats.total_tenants}
              color="text-blue-500"
            />
          </div>
        )}

        {/* Search + filter */}
        <div className="flex flex-col sm:flex-row items-start sm:items-center gap-3 mb-4">
          <div className="relative flex-1 w-full">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <input
              type="text"
              value={search}
              onChange={(e) => handleSearchChange(e.target.value)}
              placeholder="Search by name or email..."
              className="w-full pl-9 pr-4 py-2 text-sm rounded-lg border bg-card focus:outline-none focus:ring-2 focus:ring-primary/30"
            />
          </div>
          <div className="flex gap-2">
            <select
              value={filterTenant}
              onChange={(e) => {
                const val = e.target.value;
                setFilterTenant(val === "all" ? "all" : Number(val));
                setPage(1);
              }}
              className="px-3 py-1.5 text-xs font-medium rounded-lg border bg-card text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/30 min-w-[140px]"
            >
              <option value="all">All Tenants</option>
              {tenants?.map((t) => (
                <option key={t.id} value={t.id}>
                  {t.name}
                </option>
              ))}
            </select>
            <div className="flex gap-1 rounded-lg border bg-card p-0.5">
              {(["all", "active", "inactive"] as FilterStatus[]).map((f) => (
                <button
                  key={f}
                  onClick={() => {
                    setFilterStatus(f);
                    setPage(1);
                  }}
                  className={cn(
                    "px-3 py-1.5 text-xs font-medium rounded-md transition-colors capitalize",
                    filterStatus === f
                      ? "bg-primary text-primary-foreground"
                      : "text-muted-foreground hover:text-foreground",
                  )}
                >
                  {f}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Table */}
        {isLoading ? (
          <div className="flex items-center justify-center py-20">
            <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
          </div>
        ) : !data || data.users.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-20 text-muted-foreground">
            <Users className="w-10 h-10 mb-3 opacity-30" />
            <p className="text-sm">No users found</p>
          </div>
        ) : (
          <div className="border rounded-xl overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b bg-muted/30">
                    <th className="px-4 py-3 text-left font-medium text-muted-foreground">
                      User
                    </th>
                    <th className="px-4 py-3 text-left font-medium text-muted-foreground">
                      Status
                    </th>
                    <th className="px-4 py-3 text-left font-medium text-muted-foreground">
                      Role
                    </th>
                    <th className="px-4 py-3 text-left font-medium text-muted-foreground">
                      Tenants
                    </th>
                    <th className="px-4 py-3 text-right font-medium text-muted-foreground">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {data.users.map((u) => (
                    <tr
                      key={u.id}
                      className={cn(
                        "border-b last:border-0 transition-colors",
                        isSelf(u.id)
                          ? "bg-primary/5"
                          : "hover:bg-muted/20",
                      )}
                    >
                      {/* Avatar + Info */}
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-3">
                          <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center text-xs font-semibold text-primary flex-shrink-0">
                            {u.full_name[0]?.toUpperCase() || "?"}
                          </div>
                          <div className="min-w-0">
                            <div className="flex items-center gap-1.5">
                              <p className="font-medium truncate">
                                {u.full_name}
                              </p>
                              {isSelf(u.id) && (
                                <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-primary/15 text-primary font-medium">
                                  You
                                </span>
                              )}
                            </div>
                            <p className="text-xs text-muted-foreground truncate">
                              {u.email}
                            </p>
                          </div>
                        </div>
                      </td>

                      {/* Status */}
                      <td className="px-4 py-3">
                        <span
                          className={cn(
                            "inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium",
                            u.is_active
                              ? "bg-green-500/15 text-green-600"
                              : "bg-amber-500/15 text-amber-600",
                          )}
                        >
                          <span
                            className={cn(
                              "w-1.5 h-1.5 rounded-full",
                              u.is_active ? "bg-green-500" : "bg-amber-500",
                            )}
                          />
                          {u.is_active ? "Active" : "Pending"}
                        </span>
                      </td>

                      {/* Role */}
                      <td className="px-4 py-3">
                        {u.is_superadmin ? (
                          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-amber-500/15 text-amber-600">
                            <Crown className="w-3 h-3" />
                            SuperAdmin
                          </span>
                        ) : (
                          <span className="text-xs text-muted-foreground">
                            User
                          </span>
                        )}
                      </td>

                      {/* Tenants */}
                      <td className="px-4 py-3">
                        {u.tenant_memberships.length > 0 ? (
                          <div className="flex flex-col gap-1">
                            {u.tenant_memberships.map((m) => (
                              <div key={m.id} className="flex items-center gap-1.5">
                                <span className="text-xs text-foreground truncate max-w-[120px]" title={m.tenant_name ?? undefined}>
                                  {m.tenant_name ?? `Tenant #${m.tenant_id}`}
                                </span>
                                <button
                                  onClick={() => handleToggleTenantRole(u.id, m.tenant_id, m.role)}
                                  disabled={updateRole.isPending}
                                  title="Click to toggle Admin/Member role"
                                  className={cn(
                                  "inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium flex-shrink-0 cursor-pointer transition-colors",
                                  m.role === "admin"
                                    ? "bg-blue-500/15 text-blue-600 hover:bg-blue-500/25"
                                    : "bg-muted text-muted-foreground hover:bg-muted/80",
                                )}>
                                  {m.role}
                                </button>
                                {!m.is_approved && (
                                  <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium bg-amber-500/15 text-amber-600 flex-shrink-0">
                                    pending
                                  </span>
                                )}
                              </div>
                            ))}
                          </div>
                        ) : (
                          <span className="text-xs text-muted-foreground/50">
                            None
                          </span>
                        )}
                      </td>

                      {/* Actions */}
                      <td className="px-4 py-3">
                        <div className="flex items-center justify-end gap-1">
                          {!isSelf(u.id) && (
                            <>
                              <Button
                                size="sm"
                                variant="ghost"
                                className={cn(
                                  "h-7 px-2 text-xs",
                                  u.is_active
                                    ? "text-amber-600 hover:bg-amber-500/10"
                                    : "text-green-600 hover:bg-green-500/10",
                                )}
                                onClick={() => handleToggleActive(u)}
                                title={
                                  u.is_active ? "Deactivate" : "Activate"
                                }
                              >
                                {u.is_active ? (
                                  <>
                                    <UserX className="w-3.5 h-3.5 mr-1" />
                                    Deactivate
                                  </>
                                ) : (
                                  <>
                                    <UserCheck className="w-3.5 h-3.5 mr-1" />
                                    Activate
                                  </>
                                )}
                              </Button>

                              <Button
                                size="sm"
                                variant="ghost"
                                className={cn(
                                  "h-7 px-2 text-xs",
                                  u.is_superadmin
                                    ? "text-muted-foreground hover:bg-muted"
                                    : "text-amber-600 hover:bg-amber-500/10",
                                )}
                                onClick={() => handleToggleSuperadmin(u)}
                                title={
                                  u.is_superadmin
                                    ? "Remove SuperAdmin"
                                    : "Grant SuperAdmin"
                                }
                              >
                                <Shield className="w-3.5 h-3.5 mr-1" />
                                {u.is_superadmin ? "Revoke" : "Promote"}
                              </Button>

                              <Button
                                size="sm"
                                variant="ghost"
                                className="h-7 px-2 text-xs text-destructive hover:bg-destructive/10"
                                onClick={() => setConfirmDelete(u)}
                                title="Delete user"
                              >
                                <Trash2 className="w-3.5 h-3.5" />
                              </Button>

                              <Button
                                size="sm"
                                variant="ghost"
                                className="h-7 px-2 text-xs text-muted-foreground hover:bg-muted"
                                onClick={() => {
                                  setResetTarget(u);
                                  setResetNewPassword("");
                                  setResetConfirm("");
                                  setShowResetPw(false);
                                }}
                                title="Reset password"
                              >
                                <KeyRound className="w-3.5 h-3.5" />
                              </Button>
                            </>
                          )}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Pagination */}
        {data && totalPages > 1 && (
          <div className="flex items-center justify-between mt-4">
            <p className="text-xs text-muted-foreground">
              Showing {(page - 1) * perPage + 1}–
              {Math.min(page * perPage, data.total)} of {data.total}
            </p>
            <div className="flex items-center gap-1">
              <Button
                size="sm"
                variant="ghost"
                className="h-7 w-7 p-0"
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={page <= 1}
              >
                <ChevronLeft className="w-4 h-4" />
              </Button>
              <span className="text-xs px-2 text-muted-foreground">
                {page} / {totalPages}
              </span>
              <Button
                size="sm"
                variant="ghost"
                className="h-7 w-7 p-0"
                onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                disabled={page >= totalPages}
              >
                <ChevronRight className="w-4 h-4" />
              </Button>
            </div>
          </div>
        )}
      </div>

      {/* Delete confirmation */}
      <ConfirmDialog
        open={!!confirmDelete}
        onCancel={() => setConfirmDelete(null)}
        onConfirm={handleConfirmDelete}
        title="Delete User"
        message={`Are you sure you want to delete "${confirmDelete?.full_name}" (${confirmDelete?.email})? This action cannot be undone.`}
        confirmLabel="Delete"
        variant="danger"
      />

      {/* Reset password dialog */}
      {resetTarget && (
        <>
          <div
            className="fixed inset-0 z-50 bg-black/50 backdrop-blur-sm"
            onClick={() => setResetTarget(null)}
          />
          <div className="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 z-50 w-full max-w-sm bg-card border rounded-2xl shadow-2xl p-5 space-y-4">
            <div>
              <h3 className="text-sm font-semibold">Reset Password</h3>
              <p className="text-xs text-muted-foreground mt-1">
                Set a new password for{" "}
                <span className="font-medium">{resetTarget.full_name}</span>.
              </p>
            </div>

            <div className="space-y-3">
              {/* New password */}
              <div className="space-y-1.5">
                <label className="text-xs font-medium text-muted-foreground">
                  New Password
                </label>
                <div className="relative">
                  <input
                    type={showResetPw ? "text" : "password"}
                    value={resetNewPassword}
                    onChange={(e) => setResetNewPassword(e.target.value)}
                    placeholder="Min 6 characters"
                    className="w-full px-3 py-2 pr-9 text-sm rounded-lg border bg-background focus:outline-none focus:ring-2 focus:ring-primary/30"
                  />
                  <button
                    type="button"
                    onClick={() => setShowResetPw((v) => !v)}
                    className="absolute right-2.5 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                  >
                    {showResetPw ? (
                      <EyeOff className="w-3.5 h-3.5" />
                    ) : (
                      <Eye className="w-3.5 h-3.5" />
                    )}
                  </button>
                </div>
              </div>

              {/* Confirm */}
              <div className="space-y-1.5">
                <label className="text-xs font-medium text-muted-foreground">
                  Confirm New Password
                </label>
                <input
                  type={showResetPw ? "text" : "password"}
                  value={resetConfirm}
                  onChange={(e) => setResetConfirm(e.target.value)}
                  placeholder="Repeat new password"
                  className="w-full px-3 py-2 text-sm rounded-lg border bg-background focus:outline-none focus:ring-2 focus:ring-primary/30"
                />
              </div>
            </div>

            <div className="flex items-center justify-end gap-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setResetTarget(null)}
              >
                Cancel
              </Button>
              <Button
                size="sm"
                onClick={handleResetPassword}
                disabled={resetPassword.isPending || !resetNewPassword || !resetConfirm}
              >
                {resetPassword.isPending && (
                  <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />
                )}
                Reset Password
              </Button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

// ── Stat Card ──────────────────────────────────────────────────────────────

function StatCard({
  label,
  value,
  color,
}: {
  label: string;
  value: number;
  color?: string;
}) {
  return (
    <div className="rounded-xl border bg-card px-4 py-3">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className={cn("text-2xl font-bold tabular-nums", color)}>{value}</p>
    </div>
  );
}
