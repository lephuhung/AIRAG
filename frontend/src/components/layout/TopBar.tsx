import { memo, useEffect, useState } from "react";
import { useLocation } from "react-router-dom";
import { ChevronRight, Cpu, Languages, Menu } from "lucide-react";
import { cn } from "@/lib/utils";
import { api } from "@/lib/api";
import { UserMenu } from "@/components/layout/UserMenu";
import { useTranslation } from "@/hooks/useTranslation";
import logo from "@/assets/logo.png";

interface ConfigStatus {
  llm_provider: string;
  llm_model: string;
  kg_embedding_provider: string;
  kg_embedding_model: string;
  kg_embedding_dimension: number;
  hrag_embedding_model: string;
  hrag_reranker_model: string;
}

interface TopBarProps {
  actions?: React.ReactNode;
  className?: string;
  onToggle?: () => void;
  isNarrow?: boolean;
}

export const TopBar = memo(function TopBar({ actions, className, onToggle, isNarrow }: TopBarProps) {
  const location = useLocation();
  const [config, setConfig] = useState<ConfigStatus | null>(null);
  const { t, language, setLanguage } = useTranslation();

  useEffect(() => {
    api.get<ConfigStatus>("/config/status").then(setConfig).catch(() => {});
  }, []);

  const segments: { label: string; active: boolean }[] = [
    { label: t("app.name"), active: false },
  ];

  if (location.pathname === "/") {
    segments.push({ label: t("nav.knowledge_bases"), active: true });
  } else if (location.pathname.startsWith("/knowledge-bases/")) {
    segments.push({ label: t("common.workspace"), active: true });
  }

  return (
    <div
      className={cn(
        "h-12 flex items-center justify-between px-4 border-b border-border flex-shrink-0 bg-background",
        className
      )}
    >
      {/* Breadcrumbs */}
      <div className="flex items-center gap-3 text-sm min-w-0">
        {isNarrow && (
          <button
            onClick={onToggle}
            className="p-1.5 rounded-md hover:bg-muted transition-colors -ml-1"
          >
            <Menu className="w-4 h-4 text-muted-foreground" />
          </button>
        )}
        <div className="flex items-center gap-1.5 flex-shrink-0">
           <img src={logo} alt="Logo" className="w-6 h-6 object-contain" />
        </div>
        <div className="flex items-center gap-1.5 min-w-0">
          {segments.map((seg, i) => (
            <div key={i} className="flex items-center gap-1.5 min-w-0">
              {i > 0 && <ChevronRight className="w-3.5 h-3.5 text-muted-foreground flex-shrink-0" />}
              <span
                className={cn(
                  "truncate",
                  seg.active ? "font-medium text-foreground" : "text-muted-foreground"
                )}
              >
                {seg.label}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Right-side: model badges + actions */}
      <div className="flex items-center gap-2 flex-shrink-0">
        {config && (
          <div className="flex items-center gap-1.5 mr-2">
            <div
              className={cn(
                "flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium",
                config.llm_provider === "ollama"
                  ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400"
                  : "bg-blue-500/10 text-blue-600 dark:text-blue-400"
              )}
              title={`LLM: ${config.llm_provider} / ${config.llm_model}`}
            >
              <Cpu className="w-3 h-3" />
              <span>{config.llm_model}</span>
            </div>
          </div>
        )}

        {/* Language Switcher */}
        <button
          onClick={() => setLanguage(language === "en" ? "vi" : "en")}
          className="flex items-center gap-1.5 px-2 py-1 rounded-md border bg-muted/30 hover:bg-muted/60 transition-colors text-xs font-medium text-muted-foreground hover:text-foreground"
          title={t("common.language")}
        >
          <Languages className="w-3.5 h-3.5" />
          <span className="uppercase">{language}</span>
        </button>

        {actions}
        <UserMenu />
      </div>
    </div>
  );
});
