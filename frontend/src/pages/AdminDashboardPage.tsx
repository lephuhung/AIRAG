import { useAdminStats } from "@/hooks/useAdminUsers";
import { Users, Building2, Database, FileText, Activity } from "lucide-react";
import { cn } from "@/lib/utils";

export function AdminDashboardPage() {
  const { data: stats, isLoading } = useAdminStats();

  if (isLoading) {
    return (
      <div className="flex h-full items-center justify-center p-20">
        <div className="w-6 h-6 border-2 border-primary border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  if (!stats) return null;

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-5xl mx-auto px-6 py-8">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center shadow-inner">
            <Activity className="w-5 h-5 text-primary" />
          </div>
          <div>
            <h1 className="text-2xl font-bold tracking-tight">System Dashboard</h1>
            <p className="text-sm text-muted-foreground">
              Overview of system usage and parameters
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
          <StatCard
            title="Total Users"
            value={stats.total_users}
            icon={<Users className="w-5 h-5 text-blue-500" />}
            color="text-blue-500"
            bg="bg-blue-500/10"
          />
          <StatCard
            title="Active Users"
            value={stats.active_users}
            icon={<Users className="w-5 h-5 text-green-500" />}
            color="text-green-500"
            bg="bg-green-500/10"
          />
          <StatCard
            title="Pending Users"
            value={stats.pending_users}
            icon={<Users className="w-5 h-5 text-amber-500" />}
            color="text-amber-500"
            bg="bg-amber-500/10"
          />
          <StatCard
            title="Tenants / Organizations"
            value={stats.total_tenants}
            icon={<Building2 className="w-5 h-5 text-indigo-500" />}
            color="text-indigo-500"
            bg="bg-indigo-500/10"
          />
          <StatCard
            title="Knowledge Bases"
            value={stats.total_knowledge_bases}
            icon={<Database className="w-5 h-5 text-purple-500" />}
            color="text-purple-500"
            bg="bg-purple-500/10"
          />
          <StatCard
            title="System Documents"
            value={stats.total_documents}
            icon={<FileText className="w-5 h-5 text-teal-500" />}
            color="text-teal-500"
            bg="bg-teal-500/10"
          />
        </div>
      </div>
    </div>
  );
}

function StatCard({
  title,
  value,
  icon,
  color,
  bg,
}: {
  title: string;
  value: number;
  icon: React.ReactNode;
  color?: string;
  bg?: string;
}) {
  return (
    <div className="relative overflow-hidden rounded-2xl border bg-card p-6 shadow-sm hover:shadow-md transition-shadow group">
      <div className="flex items-center gap-4 relative z-10">
        <div className={cn("p-3 rounded-xl shrink-0 transition-colors", bg)}>
          {icon}
        </div>
        <div>
          <p className="text-sm text-muted-foreground font-medium">{title}</p>
          <p className={cn("text-3xl font-black mt-1 tracking-tight", color)}>{value.toLocaleString()}</p>
        </div>
      </div>
      <div className={cn("absolute -bottom-4 -right-4 w-24 h-24 rounded-full opacity-10 blur-2xl group-hover:blur-3xl transition-all", bg)} />
    </div>
  );
}
