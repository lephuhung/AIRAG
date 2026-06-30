import { useState, useMemo, type ReactNode } from "react";
import {
  User,
  DatabaseZap,
  FileText,
  BookOpen,
  FileSearch,
  Mic,
  LayoutGrid,
  Settings2,
  Zap,
  Sparkles,
  GraduationCap,
  Share2,
  ClipboardCheck,
  Copy,
  Loader2,
} from "lucide-react";
import { cn } from "@/lib/utils";
import type { PeopleRecord } from "@/types";

/** Field display config: maps schema-agnostic keys to display labels and icon */
const FIELD_CONFIG: Record<string, { label: string; icon: ReactNode }> = {
  hoTen: { label: "Họ tên", icon: <User className="w-3 h-3" /> },
  HO_TEN: { label: "Họ tên", icon: <User className="w-3 h-3" /> },
  TenHoiVien: { label: "Họ tên", icon: <User className="w-3 h-3" /> },
  ho_ten: { label: "Họ tên", icon: <User className="w-3 h-3" /> },
  fullName: { label: "Họ tên", icon: <User className="w-3 h-3" /> },
  tenKhachHang: { label: "Họ tên", icon: <User className="w-3 h-3" /> },
  maSoBhxh: { label: "Mã BHXH", icon: <DatabaseZap className="w-3 h-3" /> },
  soTheBhyt: { label: "Số thẻ BHYT", icon: <FileText className="w-3 h-3" /> },
  ngaySinhHienThi: { label: "Ngày sinh", icon: <BookOpen className="w-3 h-3" /> },
  NGAY_SINH: { label: "Ngày sinh", icon: <BookOpen className="w-3 h-3" /> },
  NgaySinh: { label: "Ngày sinh", icon: <BookOpen className="w-3 h-3" /> },
  namsinh: { label: "Năm sinh", icon: <BookOpen className="w-3 h-3" /> },
  fullNam: { label: "Năm sinh", icon: <BookOpen className="w-3 h-3" /> },
  soCmnd: { label: "Số CMND", icon: <FileSearch className="w-3 h-3" /> },
  cmnd: { label: "Số CMND", icon: <FileSearch className="w-3 h-3" /> },
  SoDinhDanh: { label: "Số định danh", icon: <FileSearch className="w-3 h-3" /> },
  MA_DOI_TUONG: { label: "Mã định danh", icon: <FileSearch className="w-3 h-3" /> },
  dienThoai: { label: "Điện thoại", icon: <Mic className="w-3 h-3" /> },
  SoDienThoai: { label: "Điện thoại", icon: <Mic className="w-3 h-3" /> },
  DIEN_THOAI_ME: { label: "Điện thoại mẹ", icon: <Mic className="w-3 h-3" /> },
  mobile: { label: "Điện thoại", icon: <Mic className="w-3 h-3" /> },
  so_dien_thoai: { label: "Điện thoại", icon: <Mic className="w-3 h-3" /> },
  diaChi: { label: "Địa chỉ", icon: <LayoutGrid className="w-3 h-3" /> },
  DiaChi: { label: "Địa chỉ", icon: <LayoutGrid className="w-3 h-3" /> },
  diaChiCapDien: { label: "Địa chỉ", icon: <LayoutGrid className="w-3 h-3" /> },
  coSoKCB: { label: "CS KCB", icon: <Settings2 className="w-3 h-3" /> },
  trangThaiThe: { label: "Trạng thái", icon: <Zap className="w-3 h-3" /> },
  tyLeBhyt: { label: "Tỷ lệ BHYT", icon: <Sparkles className="w-3 h-3" /> },
  tuNgay: { label: "Từ ngày", icon: <BookOpen className="w-3 h-3" /> },
  denNgay: { label: "Đến ngày", icon: <BookOpen className="w-3 h-3" /> },
  ngayDangKy: { label: "Ngày đăng ký", icon: <BookOpen className="w-3 h-3" /> },
  TenHangHoiVien: { label: "Hạng hội viên", icon: <GraduationCap className="w-3 h-3" /> },
  SoTheHoiVien: { label: "Số thẻ hội viên", icon: <FileText className="w-3 h-3" /> },
  TEN_ME: { label: "Tên mẹ", icon: <User className="w-3 h-3" /> },
  GIOI_TINH: { label: "Giới tính", icon: <User className="w-3 h-3" /> },
  PID: { label: "PID", icon: <FileSearch className="w-3 h-3" /> },
  uid: { label: "Facebook UID", icon: <Share2 className="w-3 h-3" /> },
  uids: { label: "Facebook UID", icon: <Share2 className="w-3 h-3" /> },
};

/** Fields to exclude from display */
const SKIP_FIELDS = new Set(["_id", "_source_schema", "_person_group", "lookup_type", "found", "persons", "display"]);

/** Get name field from a people record (tries multiple possible field names) */
function getNameField(record: Record<string, unknown>): string {
  for (const key of ["hoTen", "HO_TEN", "TenHoiVien", "ho_ten", "fullName", "tenKhachHang"]) {
    if (record[key] && typeof record[key] === "string") {
      return record[key] as string;
    }
  }
  return "(Không có tên)";
}

/** Schema display name mapping */
const SCHEMA_LABELS: Record<string, string> = {
  bhxh: "BHXH",
  lg: "LG Hội viên",
  vacxin: "Tiêm chủng",
  evn: "Điện lực",
  cv19: "Covid 19",
  uids: "UIDS",
  vnvc: "VNVC",
};

/** A person consolidated from one or more source records (across schemas) */
interface MergedPerson {
  groupKey: string;
  name: string;
  sources: string[];               // distinct schema keys, in first-seen order
  fields: [string, string[]][];    // [fieldKey, distinct values], ordered by FIELD_CONFIG
}

/**
 * Group flat people records into consolidated persons.
 * Records sharing `_person_group` (set by the backend after cross-schema
 * consolidation) become ONE card. Old data without `_person_group` falls back
 * to one card per record (previous behaviour).
 */
function mergePeople(people: PeopleRecord[]): MergedPerson[] {
  const fieldOrder = Object.keys(FIELD_CONFIG);
  const groups = new Map<string, PeopleRecord[]>();
  const order: string[] = [];

  people.forEach((p, i) => {
    const g = p._person_group;
    const key = g !== undefined && g !== null ? `g${g}` : `i${i}`;
    if (!groups.has(key)) {
      groups.set(key, []);
      order.push(key);
    }
    groups.get(key)!.push(p);
  });

  return order.map((key) => {
    const recs = groups.get(key)!;
    const sources: string[] = [];
    const fieldMap = new Map<string, string[]>();
    let name = "";

    for (const r of recs) {
      const schema = (r._source_schema as string) || "";
      if (schema && !sources.includes(schema)) sources.push(schema);
      if (name === "") {
        const n = getNameField(r);
        if (n !== "(Không có tên)") name = n;
      }
      for (const [k, v] of Object.entries(r)) {
        if (SKIP_FIELDS.has(k) || v === undefined || v === null || v === "") continue;
        const val = String(v).trim();
        if (!val) continue;
        if (!fieldMap.has(k)) fieldMap.set(k, []);
        const arr = fieldMap.get(k)!;
        if (!arr.includes(val)) arr.push(val);
      }
    }

    // Fallback header khi không có tên (vd hồ sơ chỉ có UID/SĐT như uids)
    if (!name) {
      const phoneVal = ["soDienThoai", "SoDienThoai", "dienThoai", "mobile", "so_dien_thoai", "DIEN_THOAI_ME", "phone"]
        .map((k) => fieldMap.get(k)?.[0])
        .find(Boolean);
      const uidVal = fieldMap.get("uid")?.[0] || fieldMap.get("uids")?.[0];
      name = phoneVal || (uidVal ? `UID ${uidVal}` : "");
    }

    const fields = Array.from(fieldMap.entries())
      .filter(([k]) => FIELD_CONFIG[k])
      .sort(([a], [b]) => fieldOrder.indexOf(a) - fieldOrder.indexOf(b));

    return { groupKey: key, name: name || "(Không có tên)", sources, fields };
  });
}

export function PeopleCard({ people, isLoadingMore }: { people: PeopleRecord[], isLoadingMore?: boolean }) {
  const [copiedKey, setCopiedKey] = useState<string | null>(null);

  const merged = useMemo(() => mergePeople(people), [people]);

  if (!people || people.length === 0) return null;

  const schemaLabel = (s: string) => SCHEMA_LABELS[s] || s;

  const handleCopyCard = (person: MergedPerson) => {
    const sources = person.sources.map(schemaLabel).join(", ") || "Unknown";
    const lines = [`Thông tin: ${person.name}`, `Nguồn dữ liệu: ${sources}`, ""];
    for (const [key, vals] of person.fields) {
      const label = FIELD_CONFIG[key]?.label || key;
      lines.push(`- ${label}: ${vals.join(" · ")}`);
    }
    navigator.clipboard.writeText(lines.join("\n")).then(() => {
      setCopiedKey(person.groupKey);
      setTimeout(() => setCopiedKey(null), 2000);
    });
  };

  return (
    <div className="my-3 space-y-2">
      {merged.map((person) => {
        const isMulti = person.sources.length > 1;
        return (
          <div
            key={person.groupKey}
            className="relative rounded-xl border border-border/40 bg-zinc-50/50 dark:bg-zinc-900/40 p-4 hover:shadow-md transition-all duration-300"
          >
            {/* Copy button — top right corner */}
            <button
              onClick={() => handleCopyCard(person)}
              className={cn(
                "absolute top-3 right-3 p-2 rounded-lg text-xs transition-all",
                copiedKey === person.groupKey
                  ? "bg-emerald-500/10 text-emerald-600"
                  : "bg-muted/30 hover:bg-muted text-muted-foreground hover:text-foreground"
              )}
              title="Copy"
            >
              {copiedKey === person.groupKey ? (
                <ClipboardCheck className="w-4 h-4" />
              ) : (
                <Copy className="w-4 h-4" />
              )}
            </button>

            {/* Header: name + source badge(s) */}
            <div className="flex items-center gap-3 mb-4 pr-10">
              <div className="w-9 h-9 rounded-xl bg-primary/10 border border-primary/20 flex items-center justify-center text-primary text-sm font-bold shadow-sm">
                {person.name.charAt(0).toUpperCase()}
              </div>
              <div className="flex-1 min-w-0">
                <p className="font-bold text-[15px] text-foreground tracking-tight truncate">{person.name}</p>
                {person.sources.length > 0 && (
                  <div className="flex items-center flex-wrap gap-1.5 mt-1">
                    {isMulti && (
                      <span className="inline-flex items-center gap-1 px-1.5 h-[18px] rounded-full bg-emerald-500/12 text-emerald-600 dark:text-emerald-400 text-[9px] font-bold uppercase tracking-wider">
                        <LayoutGrid className="w-2.5 h-2.5" />
                        {person.sources.length} nguồn
                      </span>
                    )}
                    {person.sources.map((s) => (
                      <span
                        key={s}
                        className="inline-flex items-center gap-1 px-1.5 h-[18px] rounded-full bg-primary/10 text-primary text-[9px] font-bold uppercase tracking-wider"
                      >
                        <span className="w-1.5 h-1.5 rounded-full bg-primary/60" />
                        {schemaLabel(s)}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* Fields grid (merged across sources) */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-2.5">
              {person.fields.map(([key, vals]) => {
                const config = FIELD_CONFIG[key];
                if (!config) return null;
                return (
                  <div key={key} className="flex items-start gap-2.5">
                    <div className="flex-shrink-0 w-5 h-5 mt-0.5 rounded-md bg-muted/40 flex items-center justify-center text-primary/70 scale-90">
                      {config.icon}
                    </div>
                    <div className="min-w-0">
                      <p className="text-[10px] text-muted-foreground font-semibold uppercase tracking-tight leading-none mb-0.5">{config.label}</p>
                      {key === "uid" || key === "uids" ? (
                        <div className="flex flex-col gap-0.5">
                          {vals.map((v) => (
                            <a
                              key={v}
                              href={`https://facebook.com/${v}`}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-[12px] font-medium text-blue-600 dark:text-blue-400 hover:underline truncate"
                            >
                              {v}
                            </a>
                          ))}
                        </div>
                      ) : (
                        <p className="text-[12px] font-medium text-foreground/90 break-words">{vals.join(" · ")}</p>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        );
      })}

      {isLoadingMore && (
        <div className="flex items-center justify-center gap-2 p-3 text-muted-foreground bg-muted/10 rounded-lg border border-dashed">
          <Loader2 className="w-4 h-4 animate-spin" />
          <span className="text-xs font-medium">Đang tìm kiếm thêm cơ sở dữ liệu...</span>
        </div>
      )}
    </div>
  );
}
