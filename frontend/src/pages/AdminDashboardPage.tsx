import { useAdminStats } from "@/hooks/useAdminUsers";
import { Users, Building2, Database, FileText, Activity, AlertCircle, Clock } from "lucide-react";
import { cn } from "@/lib/utils";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import dayjs from "dayjs";

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
            title="Total Uploaded Documents"
            value={stats.total_documents}
            icon={<FileText className="w-5 h-5 text-teal-500" />}
            color="text-teal-500"
            bg="bg-teal-500/10"
          />
        </div>

        {/* Advanced Metrics Area */}
        <div className="mt-8 grid grid-cols-1 lg:grid-cols-2 gap-6">
          
          {/* Growth Chart */}
          <div className="bg-card border rounded-2xl p-6 shadow-sm">
            <h2 className="text-lg font-semibold tracking-tight mb-4">System Growth (Last 30 Days)</h2>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={stats.users_growth}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                  <XAxis 
                    dataKey="date" 
                    tickFormatter={(val) => dayjs(val).format("MMM DD")} 
                    fontSize={12} 
                    tickMargin={10} 
                  />
                  <YAxis fontSize={12} allowDecimals={false} />
                  <Tooltip 
                    labelFormatter={(label) => dayjs(label).format("MMM DD, YYYY")}
                    contentStyle={{ borderRadius: '8px', border: '1px solid #e2e8f0', fontSize: '14px' }}
                  />
                  <Line type="monotone" dataKey="count" name="New Users" stroke="#3b82f6" strokeWidth={3} dot={{ r: 3 }} activeDot={{ r: 6 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Document Status */}
          <div className="bg-card border rounded-2xl p-6 shadow-sm">
            <h2 className="text-lg font-semibold tracking-tight mb-4">Document Status Breakdown</h2>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Tooltip contentStyle={{ borderRadius: '8px', border: '1px solid #e2e8f0', fontSize: '14px' }} />
                  <Legend verticalAlign="bottom" height={36} iconType="circle" wrapperStyle={{ fontSize: '12px' }} />
                  <Pie
                    data={stats.document_status_breakdown}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={90}
                    paddingAngle={2}
                    dataKey="count"
                    nameKey="status"
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                    labelLine={false}
                  >
                    {stats.document_status_breakdown.map((entry, index) => {
                      const colors = {
                        indexed: "#10b981", // green
                        failed: "#ef4444",   // red
                        pending: "#f59e0b",  // amber
                        chunking: "#3b82f6", // blue
                        parsing: "#8b5cf6",  // violet
                      };
                      return <Cell key={`cell-${index}`} fill={colors[entry.status as keyof typeof colors] || "#94a3b8"} />;
                    })}
                  </Pie>
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Top Workspaces */}
          <div className="bg-card border rounded-2xl p-6 shadow-sm">
            <h2 className="text-lg font-semibold tracking-tight mb-4">Top Workspaces (Storage)</h2>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={stats.top_workspaces} layout="vertical" margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} opacity={0.3} />
                  <XAxis type="number" tickFormatter={(val) => `${(val / (1024 * 1024)).toFixed(1)} MB`} fontSize={12} />
                  <YAxis dataKey="name" type="category" width={150} fontSize={12} tick={{ fill: 'currentColor' }} />
                  <Tooltip 
                    formatter={(value: number) => [`${(value / (1024 * 1024)).toFixed(2)} MB`, "Total Size"]}
                    contentStyle={{ borderRadius: '8px', border: '1px solid #e2e8f0', fontSize: '14px' }}
                  />
                  <Bar dataKey="total_size" fill="#8b5cf6" radius={[0, 4, 4, 0]} maxBarSize={40}>
                    {stats.top_workspaces.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={["#8b5cf6", "#ec4899", "#f43f5e", "#f97316", "#eab308"][index % 5]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Document Types */}
          <div className="bg-card border rounded-2xl p-6 shadow-sm">
            <h2 className="text-lg font-semibold tracking-tight mb-4">Document Types Breakdown</h2>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={stats.document_type_breakdown} margin={{ top: 5, right: 30, left: 10, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} opacity={0.3} />
                  <XAxis dataKey="name" fontSize={12} tickMargin={10} />
                  <YAxis fontSize={12} allowDecimals={false} />
                  <Tooltip 
                    formatter={(value: number) => [value, "Count"]}
                    contentStyle={{ borderRadius: '8px', border: '1px solid #e2e8f0', fontSize: '14px' }}
                  />
                  <Bar dataKey="count" fill="#10b981" radius={[4, 4, 0, 0]} maxBarSize={40}>
                    {stats.document_type_breakdown.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={["#3b82f6", "#10b981", "#8b5cf6", "#f59e0b", "#ec4899", "#14b8a6", "#6366f1"][index % 7]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Actionable Tables */}
        <div className="mt-8 grid grid-cols-1 lg:grid-cols-2 gap-6 pb-12">
          
          {/* Failed Documents Alert */}
          <div className="bg-card border-red-500/30 border-2 rounded-2xl overflow-hidden shadow-sm">
            <div className="bg-red-500/10 px-4 py-3 border-b flex items-center gap-2">
              <AlertCircle className="w-5 h-5 text-red-500" />
              <h2 className="text-sm font-bold text-red-600">Recent Failed Documents</h2>
            </div>
            <div className="p-0">
              {stats.recent_failed_docs.length === 0 ? (
                <p className="text-sm text-muted-foreground p-6 text-center">No failed documents recently. System is healthy!</p>
              ) : (
                <ul className="divide-y">
                  {stats.recent_failed_docs.map(doc => (
                    <li key={doc.id} className="p-4 hover:bg-muted/30 transition-colors">
                      <p className="text-sm font-semibold truncate" title={doc.filename}>{doc.filename}</p>
                      <p className="text-xs text-muted-foreground mt-1">Workspace: {doc.workspace_name}</p>
                      <p className="text-xs text-red-500/80 mt-2 bg-red-500/10 p-2 rounded truncate" title={doc.error_message || ""}>
                        {doc.error_message || "Unknown error"}
                      </p>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>

          {/* Pending Approvals */}
          <div className="bg-card border-amber-500/30 border-2 rounded-2xl overflow-hidden shadow-sm">
            <div className="bg-amber-500/10 px-4 py-3 border-b flex items-center gap-2">
              <Clock className="w-5 h-5 text-amber-500" />
              <h2 className="text-sm font-bold text-amber-600">Pending Approvals</h2>
            </div>
            <div className="p-0">
              {stats.pending_approvals.length === 0 ? (
                <p className="text-sm text-muted-foreground p-6 text-center">No pending users waiting for approval.</p>
              ) : (
                <ul className="divide-y">
                  {stats.pending_approvals.map(user => (
                    <li key={user.user_id} className="p-4 flex items-center justify-between hover:bg-muted/30 transition-colors">
                      <div className="overflow-hidden">
                        <p className="text-sm font-semibold truncate" title={user.email}>{user.email}</p>
                        <p className="text-xs text-muted-foreground mt-1 truncate">Tenant: {user.tenant_name} • Role: {user.role}</p>
                      </div>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>
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
