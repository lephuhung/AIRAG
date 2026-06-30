import { Image as ImageIcon, Music, GraduationCap, Pencil } from "lucide-react";
import { useTranslation } from "@/hooks/useTranslation";

// Suggestion chips (empty state)
export function SuggestionChips({ onSelect }: { onSelect: (text: string) => void }) {
  const { t } = useTranslation();

  const suggestions = [
    { text: t("chat.suggestion_topics"), icon: <ImageIcon className="w-3.5 h-3.5 text-orange-400" /> },
    { text: t("chat.suggestion_entities"), icon: <Music className="w-3.5 h-3.5 text-pink-400" /> },
    { text: t("chat.suggestion_methodology"), icon: <GraduationCap className="w-3.5 h-3.5 text-blue-400" /> },
    { text: t("chat.suggestion_any"), icon: <Pencil className="w-3.5 h-3.5 text-gray-400" /> },
  ];

  return (
    <div className="flex flex-wrap gap-2.5 justify-center max-w-[800px] mt-8 animate-in fade-in slide-in-from-bottom-4 duration-700 delay-300 fill-mode-both px-4">
      {suggestions.map((s) => (
        <button
          key={s.text}
          onClick={() => onSelect(s.text)}
          className="flex items-center gap-2.5 text-[13px] px-5 py-2.5 rounded-full bg-secondary/30 hover:bg-secondary/60 border border-transparent hover:border-secondary transition-all duration-300 text-muted-foreground hover:text-foreground font-medium shadow-sm active:scale-95 whitespace-nowrap"
        >
          {s.icon}
          <span>{s.text}</span>
        </button>
      ))}
    </div>
  );
}
