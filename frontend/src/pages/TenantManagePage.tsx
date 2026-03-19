import { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { toast } from "sonner";
import { api } from "@/lib/api";
import { useAuthStore } from "@/stores/authStore";
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
} from "lucide-react";
import { cn } from "@/lib/utils";
import type { Tenant, TenantUser } from "@/types";

export function TenantManagePage() {
  const { tenantId } = useParams<{ tenantId: string }>();
  const navigate = useNavigate();
  const user = useAuthStore((s) => s.user);
  const [tenant, setTenant] = useState<Tenant | null>(null);
  const [members, setMembers] = useState<TenantUser[]>([]);
  const [loading, setLoading] = useState(true);

  const tid = Number(tenantId);

  useEffect(() => {
    if (!tid) return;
    loadData();
  }, [tid]);

  const loadData = async () => {
    try {
      const [tenantList, memberList] = await Promise.all([
        api.get<Tenant[]>("/tenants/my"),
        api.get<TenantUser[]>(`/tenants/${tid}/users`),
      ]);
      const found = tenantList.find((t) => t.id === tid);
      setTenant(found ?? null);
      setMembers(memberList);
    } catch (err: any) {
      toast.error(err.message || "Failed to load tenant data");
    } finally {
      setLoading(false);
    }
  };

  const handleApprove = async (userId: number) => {
    try {
      await api.post(`/tenants/${tid}/users/${userId}/approve`);
      toast.success("User approved");
      loadData();
    } catch (err: any) {
      toast.error(err.message || "Failed to approve user");
    }
  };

  const handleReject = async (userId: number) => {
    try {
      await api.post(`/tenants/${tid}/users/${userId}/reject`);
      toast.success("User rejected");
      loadData();
    } catch (err: any) {
      toast.error(err.message || "Failed to reject user");
    }
  };

  const handleRemove = async (userId: number) => {
    try {
      await api.delete(`/tenants/${tid}/users/${userId}`);
      toast.success("User removed");
      loadData();
    } catch (err: any) {
      toast.error(err.message || "Failed to remove user");
    }
  };

  const handleToggleRole = async (userId: number, currentRole: string) => {
    const newRole = currentRole === "admin" ? "member" : "admin";
    try {
      await api.put(`/tenants/${tid}/users/${userId}/role`, { role: newRole });
      toast.success(`Role changed to ${newRole}`);
      loadData();
    } catch (err: any) {
      toast.error(err.message || "Failed to change role");
    }
  };

  if (loading) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="animate-pulse text-muted-foreground">Loading...</div>
      </div>
    );
  }

  const pendingMembers = members.filter((m) => !m.is_approved);
  const approvedMembers = members.filter((m) => m.is_approved);

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-3xl mx-auto px-6 py-8">
        {/* Header */}
        <div className="flex items-center gap-3 mb-6">
          <button
            onClick={() => navigate("/")}
            className="w-8 h-8 flex items-center justify-center rounded-lg text-muted-foreground hover:bg-muted transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
          </button>
          <div className="flex items-center gap-2">
            <Building2 className="w-5 h-5 text-amber-500" />
            <h2 className="text-lg font-semibold">
              {tenant?.name || `Tenant #${tid}`}
            </h2>
          </div>
        </div>

        {/* Pending approvals */}
        {pendingMembers.length > 0 && (
          <div className="mb-8">
            <h3 className="text-sm font-semibold text-amber-500 uppercase tracking-wider mb-3">
              Pending Approvals ({pendingMembers.length})
            </h3>
            <div className="space-y-2">
              {pendingMembers.map((m) => (
                <Card key={m.id} className="border-amber-500/20">
                  <CardContent className="py-3 flex items-center justify-between">
                    <div className="min-w-0">
                      <p className="text-sm font-medium truncate">{m.full_name || "Unknown"}</p>
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
                        Approve
                      </Button>
                      <Button
                        size="sm"
                        variant="ghost"
                        className="text-destructive hover:bg-destructive/10"
                        onClick={() => handleReject(m.user_id)}
                      >
                        <X className="w-4 h-4 mr-1" />
                        Reject
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
            Members ({approvedMembers.length})
          </h3>
          {approvedMembers.length === 0 ? (
            <p className="text-sm text-muted-foreground">No approved members yet.</p>
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
                          <p className="text-sm font-medium truncate">{m.full_name || "Unknown"}</p>
                          {m.role === "admin" && (
                            <Shield className="w-3.5 h-3.5 text-amber-500 flex-shrink-0" title="Admin" />
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
                            title={m.role === "admin" ? "Demote to member" : "Promote to admin"}
                          >
                            {m.role === "admin" ? (
                              <>
                                <User className="w-3.5 h-3.5 mr-1" />
                                Demote
                              </>
                            ) : (
                              <>
                                <Shield className="w-3.5 h-3.5 mr-1" />
                                Promote
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
    </div>
  );
}
