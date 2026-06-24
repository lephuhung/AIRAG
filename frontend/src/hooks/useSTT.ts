/**
 * useSTT
 * ======
 *
 * Voice-to-text for the chat input. Records the microphone with MediaRecorder,
 * posts the audio blob to the backend `/stt/transcribe` (Whisper) endpoint, and
 * hands the transcript back to the caller.
 *
 * Mirrors the spirit of `handy` (local Whisper dictation) but server-side:
 * press mic → speak → press again → transcript is appended to the chat box for
 * the user to review before sending.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { api } from "@/lib/api";

// Prefer Opus-in-webm; fall back to whatever the browser offers.
function pickMimeType(): string {
  const candidates = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/mp4"];
  const MR = (window as any).MediaRecorder;
  if (MR && typeof MR.isTypeSupported === "function") {
    for (const c of candidates) {
      if (MR.isTypeSupported(c)) return c;
    }
  }
  return "";
}

interface UseSTTOptions {
  /** Called with the transcribed text once a recording finishes. */
  onTranscript: (text: string) => void;
  /** Translation function for user-facing messages. */
  t?: (key: string) => string;
}

export function useSTT({ onTranscript, t }: UseSTTOptions) {
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);

  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const tr = useCallback((key: string, fallback: string) => (t ? t(key) || fallback : fallback), [t]);

  const cleanupStream = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    recorderRef.current = null;
  }, []);

  const startRecording = useCallback(async () => {
    // getUserMedia/MediaRecorder are only exposed in a "secure context"
    // (HTTPS, or localhost/127.0.0.1). On a plain-HTTP origin like
    // http://rag.hatinh.local the browser leaves navigator.mediaDevices
    // undefined — surface that as the real cause, not "unsupported".
    if (typeof window !== "undefined" && window.isSecureContext === false) {
      toast.error(tr("chat.stt_insecure", "Cần HTTPS hoặc localhost để dùng micro (trang đang chạy HTTP)"));
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia || !(window as any).MediaRecorder) {
      toast.error(tr("chat.stt_unsupported", "Trình duyệt không hỗ trợ ghi âm"));
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      chunksRef.current = [];

      const mimeType = pickMimeType();
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      recorderRef.current = recorder;

      recorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.onstop = async () => {
        const type = recorder.mimeType || "audio/webm";
        const blob = new Blob(chunksRef.current, { type });
        chunksRef.current = [];
        cleanupStream();
        setIsRecording(false);

        if (blob.size === 0) return;
        const ext = type.includes("ogg") ? "ogg" : type.includes("mp4") ? "mp4" : "webm";
        setIsTranscribing(true);
        try {
          const res = await api.transcribeAudio(blob, `recording.${ext}`);
          const text = (res?.text || "").trim();
          if (text) {
            onTranscript(text);
          } else {
            toast.info(tr("chat.stt_empty", "Không nhận được giọng nói"));
          }
        } catch (err) {
          toast.error(err instanceof Error ? err.message : tr("chat.stt_failed", "Không thể chuyển giọng nói thành văn bản"));
        } finally {
          setIsTranscribing(false);
        }
      };

      recorder.start();
      setIsRecording(true);
    } catch (err) {
      cleanupStream();
      setIsRecording(false);
      const name = (err as DOMException)?.name;
      if (name === "NotAllowedError" || name === "SecurityError") {
        toast.error(tr("chat.mic_permission_denied", "Bạn cần cấp quyền micro để dùng tính năng này"));
      } else {
        toast.error(tr("chat.stt_failed", "Không thể chuyển giọng nói thành văn bản"));
      }
    }
  }, [cleanupStream, onTranscript, tr]);

  const stopRecording = useCallback(() => {
    const recorder = recorderRef.current;
    if (recorder && recorder.state !== "inactive") {
      recorder.stop();
    } else {
      cleanupStream();
      setIsRecording(false);
    }
  }, [cleanupStream]);

  const toggleRecording = useCallback(() => {
    if (isTranscribing) return;
    if (isRecording) {
      stopRecording();
    } else {
      void startRecording();
    }
  }, [isRecording, isTranscribing, startRecording, stopRecording]);

  // Tear down the mic stream if the component unmounts mid-recording.
  useEffect(() => () => cleanupStream(), [cleanupStream]);

  return { isRecording, isTranscribing, toggleRecording, startRecording, stopRecording };
}
