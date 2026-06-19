/**
 * TTSTab
 * ======
 * Per-user text-to-speech preferences: voice, speaking speed, pitch. Saved into
 * `users.settings.tts` via PUT /auth/me and used as the default whenever an
 * answer is read aloud in the chat. Includes a short preview button.
 */
import { useEffect, useRef, useState } from "react";
import { Loader2, Volume2, Play, Square } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";
import { useAuthStore } from "@/stores/authStore";
import type { TTSVoice, TTSVoicesResponse, User } from "@/types";

const PREVIEW_TEXT = "Xin chào, đây là giọng đọc thử cho hệ thống AIRAG.";

export function TTSTab() {
  const user = useAuthStore((s) => s.user)!;
  const updateUser = useAuthStore((s) => s.updateUser);

  const current = user.settings?.tts ?? {};
  const [voice, setVoice] = useState<string>(current.voice ?? "");
  const [speed, setSpeed] = useState<number>(current.speed ?? 1.0);
  const [pitch, setPitch] = useState<number>(current.pitch ?? 1.0);

  const [voices, setVoices] = useState<TTSVoice[]>([]);
  const [enabled, setEnabled] = useState(true);
  const [loadingVoices, setLoadingVoices] = useState(true);
  const [saving, setSaving] = useState(false);
  const [previewing, setPreviewing] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .get<TTSVoicesResponse>("/tts/voices")
      .then((res) => {
        if (cancelled) return;
        setVoices(res.voices);
        setEnabled(res.enabled);
      })
      .catch(() => setEnabled(false))
      .finally(() => !cancelled && setLoadingVoices(false));
    return () => {
      cancelled = true;
      audioRef.current?.pause();
    };
  }, []);

  const handlePreview = async () => {
    if (previewing) {
      audioRef.current?.pause();
      setPreviewing(false);
      return;
    }
    setPreviewing(true);
    try {
      const blob = await api.synthesizeSpeech({ text: PREVIEW_TEXT, voice, speed, pitch });
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      audioRef.current = audio;
      audio.onended = () => {
        setPreviewing(false);
        URL.revokeObjectURL(url);
      };
      audio.onerror = () => {
        setPreviewing(false);
        URL.revokeObjectURL(url);
      };
      await audio.play();
    } catch (err) {
      setPreviewing(false);
      toast.error(err instanceof Error ? err.message : "Không thể phát thử");
    }
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      const updated = await api.put<User>("/auth/me", {
        settings: { tts: { voice, speed, pitch } },
      });
      updateUser(updated);
      toast.success("Đã lưu giọng đọc");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Lưu thất bại");
    } finally {
      setSaving(false);
    }
  };

  if (loadingVoices) {
    return (
      <div className="flex items-center justify-center py-8 text-muted-foreground">
        <Loader2 className="w-4 h-4 animate-spin mr-2" /> Đang tải...
      </div>
    );
  }

  if (!enabled) {
    return (
      <div className="flex flex-col items-center gap-2 py-8 text-center text-muted-foreground">
        <Volume2 className="w-6 h-6" />
        <p className="text-sm">Dịch vụ đọc văn bản hiện chưa được bật.</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Voice */}
      <div className="space-y-1.5">
        <label className="text-xs font-medium text-muted-foreground">Giọng đọc</label>
        <select
          value={voice}
          onChange={(e) => setVoice(e.target.value)}
          className="w-full px-3 py-2 text-sm rounded-lg border bg-background focus:outline-none focus:ring-2 focus:ring-primary/30"
        >
          {voices.map((v) => (
            <option key={v.id || "default"} value={v.id}>
              {v.label}
            </option>
          ))}
        </select>
      </div>

      {/* Speed */}
      <div className="space-y-1.5">
        <label className="flex items-center justify-between text-xs font-medium text-muted-foreground">
          <span>Tốc độ</span>
          <span className="tabular-nums">{speed.toFixed(2)}×</span>
        </label>
        <input
          type="range"
          min={0.5}
          max={2}
          step={0.05}
          value={speed}
          onChange={(e) => setSpeed(parseFloat(e.target.value))}
          className="w-full accent-primary"
        />
      </div>

      {/* Pitch */}
      <div className="space-y-1.5">
        <label className="flex items-center justify-between text-xs font-medium text-muted-foreground">
          <span>Cao độ</span>
          <span className="tabular-nums">{pitch.toFixed(2)}×</span>
        </label>
        <input
          type="range"
          min={0.5}
          max={2}
          step={0.05}
          value={pitch}
          onChange={(e) => setPitch(parseFloat(e.target.value))}
          className="w-full accent-primary"
        />
      </div>

      <div className="flex items-center gap-2 pt-1">
        <Button variant="outline" size="sm" onClick={handlePreview}>
          {previewing ? (
            <Square className="w-3.5 h-3.5 mr-1.5" />
          ) : (
            <Play className="w-3.5 h-3.5 mr-1.5" />
          )}
          {previewing ? "Dừng" : "Nghe thử"}
        </Button>
        <Button size="sm" onClick={handleSave} disabled={saving}>
          {saving && <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />}
          Lưu
        </Button>
      </div>
    </div>
  );
}
