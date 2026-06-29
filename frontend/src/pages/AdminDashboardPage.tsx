import { useAdminStats } from "@/hooks/useAdminUsers";
import { useTranslation } from "@/hooks/useTranslation";
import { Users, Building2, Database, FileText, Activity, AlertCircle, Clock, RefreshCw } from "lucide-react";
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
  const { data: stats, isLoading, isFetching } = useAdminStats();
  const { t } = useTranslation();

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
          <div className="flex-1">
            <div className="flex items-center gap-3">
              <h1 className="text-2xl font-bold tracking-tight">{t("admin.dashboard.title")}</h1>
              <div className="flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-primary/5 border border-primary/10 text-[10px] font-medium text-primary">
                <div className={cn("w-1.5 h-1.5 rounded-full bg-primary", !isFetching && "animate-pulse")} />
                {isFetching ? (
                  <RefreshCw className="w-2.5 h-2.5 animate-spin" />
                ) : (
                   "LIVE"
                )}
              </div>
            </div>
            <p className="text-sm text-muted-foreground">
              {t("admin.dashboard.subtitle")} • {t("common.auto_refresh_15s")}
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
          <StatCard
            title={t("admin.dashboard.stat_total_users")}
            value={stats.total_users}
            icon={<Users className="w-5 h-5 text-blue-500" />}
            color="text-blue-500"
            bg="bg-blue-500/10"
          />
          <StatCard
            title={t("admin.dashboard.stat_active_users")}
            value={stats.active_users}
            icon={<Users className="w-5 h-5 text-green-500" />}
            color="text-green-500"
            bg="bg-green-500/10"
          />
          <StatCard
            title={t("admin.dashboard.stat_pending_users")}
            value={stats.pending_users}
            icon={<Users className="w-5 h-5 text-amber-500" />}
            color="text-amber-500"
            bg="bg-amber-500/10"
          />
          <StatCard
            title={t("admin.dashboard.stat_tenants")}
            value={stats.total_tenants}
            icon={<Building2 className="w-5 h-5 text-indigo-500" />}
            color="text-indigo-500"
            bg="bg-indigo-500/10"
          />
          <StatCard
            title={t("admin.dashboard.stat_kb")}
            value={stats.total_knowledge_bases}
            icon={<Database className="w-5 h-5 text-purple-500" />}
            color="text-purple-500"
            bg="bg-purple-500/10"
          />
          <StatCard
            title={t("admin.dashboard.stat_docs")}
            value={stats.total_documents}
            icon={<FileText className="w-5 h-5 text-teal-500" />}
            color="text-teal-500"
            bg="bg-teal-500/10"
          />
        </div>

        {/* Advanced Metrics Area */}
        <div className="mt-8 grid grid-cols-1 lg:grid-cols-2 gap-6">

          {/* Q&A Activity — questions vs answers per day */}
          <div className="bg-card border rounded-2xl p-6 shadow-sm lg:col-span-2">
            <h2 className="text-lg font-semibold tracking-tight mb-4">{t("admin.dashboard.qa_activity_title")}</h2>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={stats.messages_growth}>
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
                  <Legend verticalAlign="top" height={28} iconType="circle" wrapperStyle={{ fontSize: '12px' }} />
                  {/* questions ≥ answers always; draw questions solid underneath and
                      answers dashed on top so the blue line stays visible when the two
                      coincide (every run answered → identical series). */}
                  <Line type="monotone" dataKey="questions" name={t("admin.dashboard.qa_questions")} stroke="#0ea5e9" strokeWidth={3} dot={false} activeDot={{ r: 5 }} />
                  <Line type="monotone" dataKey="answers" name={t("admin.dashboard.qa_answers")} stroke="#10b981" strokeWidth={2} strokeDasharray="6 4" dot={false} activeDot={{ r: 5 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Growth Chart */}
          <div className="bg-card border rounded-2xl p-6 shadow-sm">
            <h2 className="text-lg font-semibold tracking-tight mb-4">{t("admin.dashboard.growth_title")}</h2>
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
                  <Line type="monotone" dataKey="count" name={t("admin.dashboard.growth_new_users")} stroke="#3b82f6" strokeWidth={3} dot={{ r: 3 }} activeDot={{ r: 6 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Document Status */}
          <div className="bg-card border rounded-2xl p-6 shadow-sm">
            <h2 className="text-lg font-semibold tracking-tight mb-4">{t("admin.dashboard.status_breakdown")}</h2>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Tooltip
                    formatter={(value: any, name: any) => [value, t("workers.status." + name)] as [any, any]}
                    contentStyle={{ borderRadius: '8px', border: '1px solid #e2e8f0', fontSize: '14px' }}
                  />
                  <Legend
                    verticalAlign="bottom"
                    height={36}
                    iconType="circle"
                    formatter={(value) => t("workers.status." + value)}
                    wrapperStyle={{ fontSize: '12px' }}
                  />
                  <Pie
                    data={stats.document_status_breakdown}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={90}
                    paddingAngle={2}
                    dataKey="count"
                    nameKey="status"
                    labelLine={false}
                    label={({ cx, cy, midAngle, innerRadius, outerRadius, percent }) => {
                      // Hide labels for tiny slices to avoid overlap; draw the rest
                      // centered INSIDE the donut band so they never collide.
                      if (!percent || percent < 0.06) return null;
                      const RADIAN = Math.PI / 180;
                      const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
                      const x = cx + radius * Math.cos(-midAngle * RADIAN);
                      const y = cy + radius * Math.sin(-midAngle * RADIAN);
                      return (
                        <text
                          x={x}
                          y={y}
                          fill="#fff"
                          textAnchor="middle"
                          dominantBaseline="central"
                          fontSize={11}
                          fontWeight={600}
                        >
                          {`${(percent * 100).toFixed(0)}%`}
                        </text>
                      );
                    }}
                  >
                    {stats.document_status_breakdown.map((_entry, index) => {
                      const colors = {
                        indexed: "#10b981", // green
                        failed: "#ef4444",   // red
                        pending: "#f59e0b",  // amber
                        chunking: "#3b82f6", // blue
                        parsing: "#8b5cf6",  // violet
                      };
                      return <Cell key={`cell-${index}`} fill={colors[_entry.status as keyof typeof colors] || "#94a3b8"} />;
                    })}
                  </Pie>
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Top Workspaces — ranking by storage size → horizontal bars */}
          <div className="bg-card border rounded-2xl p-6 shadow-sm">
            <h2 className="text-lg font-semibold tracking-tight mb-4">{t("admin.dashboard.top_workspaces")}</h2>
            <div className="h-96">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={stats.top_workspaces}
                  layout="vertical"
                  margin={{ top: 5, right: 30, left: 10, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" opacity={0.3} horizontal={false} />
                  <XAxis
                    type="number"
                    tickFormatter={(val) => `${(val / (1024 * 1024)).toFixed(0)}MB`}
                    fontSize={12}
                  />
                  <YAxis
                    type="category"
                    dataKey="name"
                    fontSize={12}
                    width={120}
                    tickMargin={6}
                  />
                  <Tooltip
                    cursor={{ fill: "rgba(139, 92, 246, 0.06)" }}
                    formatter={(value: any) => [`${(Number(value) / (1024 * 1024)).toFixed(1)} MB`, t("admin.dashboard.total_size")] as [any, any]}
                    contentStyle={{ borderRadius: '8px', border: '1px solid #e2e8f0', fontSize: '14px' }}
                  />
                  <Bar dataKey="total_size" fill="#8b5cf6" radius={[0, 6, 6, 0]} barSize={22} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Document Types — counts per category → vertical bars */}
          <div className="bg-card border rounded-2xl p-6 shadow-sm">
            <h2 className="text-lg font-semibold tracking-tight mb-4">{t("admin.dashboard.type_breakdown")}</h2>
            <div className="h-96">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={stats.document_type_breakdown} margin={{ top: 5, right: 30, left: 10, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.3} vertical={false} />
                  <XAxis dataKey="name" fontSize={12} tickMargin={10} interval={0} angle={-15} textAnchor="end" />
                  <YAxis fontSize={12} allowDecimals={false} />
                  <Tooltip
                    cursor={{ fill: "rgba(16, 185, 129, 0.06)" }}
                    formatter={(value: any) => [value, t("admin.dashboard.count")] as [any, any]}
                    contentStyle={{ borderRadius: '8px', border: '1px solid #e2e8f0', fontSize: '14px' }}
                  />
                  <Bar dataKey="count" fill="#10b981" radius={[6, 6, 0, 0]} barSize={36} />
                </BarChart>
              </ResponsiveContainer>
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
