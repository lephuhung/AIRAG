/**
 * useTTS
 * ======
 *
 * Read assistant answers aloud via the backend `/tts/synthesize` endpoint.
 *
 * A single module-level <audio> element is shared across all chat messages, so
 * starting playback on one message automatically stops any other. Components
 * subscribe through the Zustand store to reflect idle/loading/playing state.
 */
import { create } from "zustand";
import { toast } from "sonner";
import { api } from "@/lib/api";

type TTSStatus = "idle" | "loading" | "playing";

interface TTSState {
  /** id of the message currently loading or playing, or null. */
  activeId: string | null;
  status: TTSStatus;
  _audio: HTMLAudioElement | null;
  _objectUrl: string | null;
  play: (id: string, text: string, voice?: string) => Promise<void>;
  stop: () => void;
}

export const useTTSStore = create<TTSState>((set, get) => ({
  activeId: null,
  status: "idle",
  _audio: null,
  _objectUrl: null,

  stop: () => {
    const { _audio, _objectUrl } = get();
    if (_audio) {
      _audio.pause();
      _audio.src = "";
    }
    if (_objectUrl) URL.revokeObjectURL(_objectUrl);
    set({ activeId: null, status: "idle", _audio: null, _objectUrl: null });
  },

  play: async (id, text, voice) => {
    const { activeId, status } = get();
    // Toggle: clicking the active message stops it.
    if (activeId === id && status !== "idle") {
      get().stop();
      return;
    }
    // Switching messages — tear down the previous one first.
    get().stop();

    const clean = text.trim();
    if (!clean) return;

    set({ activeId: id, status: "loading" });
    try {
      const blob = await api.synthesizeSpeech({ text: clean, voice });
      // Aborted/replaced while the request was in flight.
      if (get().activeId !== id) return;

      const objectUrl = URL.createObjectURL(blob);
      const audio = new Audio(objectUrl);
      audio.onended = () => {
        if (get().activeId === id) get().stop();
      };
      audio.onerror = () => {
        if (get().activeId === id) get().stop();
      };
      set({ _audio: audio, _objectUrl: objectUrl, status: "playing" });
      await audio.play();
    } catch (err) {
      if (get().activeId === id) set({ activeId: null, status: "idle" });
      toast.error(err instanceof Error ? err.message : "Không thể đọc văn bản");
    }
  },
}));
