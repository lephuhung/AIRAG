/**
 * ToolsPage
 * =========
 *
 * A self-service utility page exposing two on-demand conversions that reuse the
 * models already running in the stack:
 *
 *   1. PDF → text  (OCR)  — `POST /ocr/extract`  (HunyuanOCR)
 *   2. Voice → text (STT) — `POST /stt/transcribe` (Whisper)
 *
 * Both feed a single, shared output area. OCR replaces it with the document
 * text; voice transcription appends to it. A status bar reports the live phase
 * (uploading / processing / recording / transcribing). Nothing is stored: the
 * upload/recording is processed in-process and the text handed back to the user.
 * Available to every authenticated user (including the plain `user` role).
 */
import { useCallback, useRef, useState } from "react";
import { toast } from "sonner";
import {
  FileText,
  Mic,
  Square,
  Upload,
  Loader2,
  Copy,
  Check,
  Download,
  ScanText,
  AudioLines,
} from "lucide-react";
import { api } from "@/lib/api";
import { useSTT } from "@/hooks/useSTT";
import { useTranslation } from "@/hooks/useTranslation";
import { cn } from "@/lib/utils";

type Source = "ocr" | "stt" | null;

export function ToolsPage() {
  const { t } = useTranslation();

  // ---- Shared output ----------------------------------------------------
  const [output, setOutput] = useState<string>("");
  const [source, setSource] = useState<Source>(null);
  const [pdfName, setPdfName] = useState<string>("");
  const [pdfPages, setPdfPages] = useState<number>(0);
  const [copied, setCopied] = useState(false);

  // ---- PDF → text (OCR) phases -----------------------------------------
  const [pdfBusy, setPdfBusy] = useState(false);
  const [phase, setPhase] = useState<"uploading" | "processing" | null>(null);
  const [uploadPct, setUploadPct] = useState(0);
  const pdfInputRef = useRef<HTMLInputElement>(null);

  // ---- Voice → text (STT) ----------------------------------------------
  const [audioBusy, setAudioBusy] = useState(false);
  const audioInputRef = useRef<HTMLInputElement>(null);

  const appendTranscript = useCallback((text: string) => {
    setOutput((prev) => (prev ? `${prev}\n${text}` : text));
    setSource("stt");
  }, []);

  const { isRecording, isTranscribing, toggleRecording } = useSTT({
    onTranscript: appendTranscript,
    t,
  });

  const busy = pdfBusy || isRecording || isTranscribing || audioBusy;

  const handlePdf = useCallback(
    async (file: File | undefined) => {
      if (!file) return;
      if (!file.name.toLowerCase().endsWith(".pdf") && file.type !== "application/pdf") {
        toast.error(t("tools.ocr.invalid"));
        return;
      }
      setPdfBusy(true);
      setPhase("uploading");
      setUploadPct(0);
      setPdfName(file.name);
      setPdfPages(0);
      try {
        const res = await api.extractPdfText(file, {
          onUploadProgress: (p) => setUploadPct(p),
          onProcessing: () => setPhase("processing"),
        });
        setOutput(res.text || "");
        setPdfPages(res.pages || 0);
        setSource("ocr");
        if (!res.text?.trim()) toast.info(t("tools.ocr.empty"));
      } catch (err) {
        toast.error(err instanceof Error ? err.message : t("tools.ocr.failed"));
      } finally {
        setPdfBusy(false);
        setPhase(null);
      }
    },
    [t]
  );

  const handleAudioFile = useCallback(
    async (file: File | undefined) => {
      if (!file) return;
      setAudioBusy(true);
      try {
        const res = await api.transcribeAudio(file, file.name);
        const text = (res?.text || "").trim();
        if (text) appendTranscript(text);
        else toast.info(t("tools.stt.empty"));
      } catch (err) {
        toast.error(err instanceof Error ? err.message : t("tools.stt.failed"));
      } finally {
        setAudioBusy(false);
      }
    },
    [appendTranscript, t]
  );

  // ---- Output actions ---------------------------------------------------
  const onCopy = useCallback(async () => {
    if (!output) return;
    try {
      await navigator.clipboard.writeText(output);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      toast.error(t("tools.copy_failed"));
    }
  }, [output, t]);

  const onDownload = useCallback(() => {
    if (!output) return;
    const name =
      source === "ocr"
        ? `${pdfName ? pdfName.replace(/\.pdf$/i, "") : "ocr"}.txt`
        : source === "stt"
        ? "transcript.txt"
        : "output.txt";
    const blob = new Blob([output], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }, [output, source, pdfName]);

  // ---- Live status ------------------------------------------------------
  let status: { text: string; tone: "busy" | "rec" } | null = null;
  if (pdfBusy && phase === "uploading")
    status = { text: t("tools.status.uploading", { percent: uploadPct }), tone: "busy" };
  else if (pdfBusy && phase === "processing")
    status = { text: t("tools.status.processing"), tone: "busy" };
  else if (isRecording) status = { text: t("tools.status.recording"), tone: "rec" };
  else if (isTranscribing) status = { text: t("tools.status.transcribing"), tone: "busy" };
  else if (audioBusy) status = { text: t("tools.status.uploading_audio"), tone: "busy" };

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="flex-shrink-0 border-b border-border px-6 py-4">
        <h1 className="text-lg font-bold flex items-center gap-2">
          <ScanText className="w-5 h-5 text-primary" />
          {t("tools.title")}
        </h1>
        <p className="text-sm text-muted-foreground mt-0.5">{t("tools.subtitle")}</p>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-6">
        <div className="max-w-5xl mx-auto flex flex-col gap-5">
          {/* Input panels */}
          <div className="grid gap-4 md:grid-cols-2">
            {/* OCR panel */}
            <section className="rounded-xl border border-border bg-card p-5 flex flex-col gap-3">
              <div>
                <h2 className="font-semibold flex items-center gap-2">
                  <FileText className="w-4 h-4 text-primary" />
                  {t("tools.ocr.title")}
                </h2>
                <p className="text-xs text-muted-foreground mt-0.5">{t("tools.ocr.hint")}</p>
              </div>

              <input
                ref={pdfInputRef}
                type="file"
                accept="application/pdf,.pdf"
                className="hidden"
                onChange={(e) => {
                  handlePdf(e.target.files?.[0]);
                  e.target.value = "";
                }}
              />
              <button
                type="button"
                onClick={() => pdfInputRef.current?.click()}
                disabled={busy}
                className={cn(
                  "w-full flex items-center justify-center gap-2 px-4 py-3 rounded-lg text-sm font-medium border-2 border-dashed transition-colors",
                  pdfBusy
                    ? "border-border text-muted-foreground cursor-wait"
                    : "border-primary/40 text-primary hover:bg-primary/5",
                  busy && !pdfBusy && "opacity-60 cursor-not-allowed"
                )}
              >
                {pdfBusy ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    {phase === "uploading"
                      ? t("tools.status.uploading", { percent: uploadPct })
                      : t("tools.ocr.processing")}
                  </>
                ) : (
                  <>
                    <Upload className="w-4 h-4" />
                    {t("tools.ocr.upload")}
                  </>
                )}
              </button>

              {/* Upload progress bar */}
              {pdfBusy && phase === "uploading" && (
                <div className="h-1.5 w-full rounded-full bg-muted overflow-hidden">
                  <div
                    className="h-full bg-primary transition-all"
                    style={{ width: `${uploadPct}%` }}
                  />
                </div>
              )}

              {pdfName && (
                <p className="text-xs text-muted-foreground truncate">
                  {pdfName}
                  {pdfPages > 0 && ` · ${t("tools.ocr.pages", { count: pdfPages })}`}
                </p>
              )}
            </section>

            {/* STT panel */}
            <section className="rounded-xl border border-border bg-card p-5 flex flex-col gap-3">
              <div>
                <h2 className="font-semibold flex items-center gap-2">
                  <AudioLines className="w-4 h-4 text-primary" />
                  {t("tools.stt.title")}
                </h2>
                <p className="text-xs text-muted-foreground mt-0.5">{t("tools.stt.hint")}</p>
              </div>

              <input
                ref={audioInputRef}
                type="file"
                accept="audio/*"
                className="hidden"
                onChange={(e) => {
                  handleAudioFile(e.target.files?.[0]);
                  e.target.value = "";
                }}
              />
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={toggleRecording}
                  disabled={isTranscribing || audioBusy || pdfBusy}
                  className={cn(
                    "flex-1 flex items-center justify-center gap-2 px-4 py-3 rounded-lg text-sm font-medium transition-colors",
                    isRecording
                      ? "bg-red-500/10 text-red-500 hover:bg-red-500/20"
                      : "bg-primary/10 text-primary hover:bg-primary/20",
                    (isTranscribing || audioBusy || pdfBusy) && "opacity-60 cursor-not-allowed"
                  )}
                >
                  {isTranscribing ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      {t("tools.stt.transcribing")}
                    </>
                  ) : isRecording ? (
                    <>
                      <Square className="w-4 h-4 fill-current" />
                      {t("tools.stt.stop")}
                    </>
                  ) : (
                    <>
                      <Mic className="w-4 h-4" />
                      {t("tools.stt.record")}
                    </>
                  )}
                </button>
                <button
                  type="button"
                  onClick={() => audioInputRef.current?.click()}
                  disabled={busy}
                  className="flex items-center justify-center gap-2 px-4 py-3 rounded-lg text-sm font-medium border border-border text-muted-foreground hover:bg-muted/60 hover:text-foreground transition-colors disabled:opacity-60 disabled:cursor-not-allowed"
                  title={t("tools.stt.upload")}
                >
                  {audioBusy ? <Loader2 className="w-4 h-4 animate-spin" /> : <Upload className="w-4 h-4" />}
                  {t("tools.stt.upload")}
                </button>
              </div>
            </section>
          </div>

          {/* Shared output */}
          <section className="rounded-xl border border-border bg-card overflow-hidden flex flex-col">
            <div className="px-5 py-3 border-b border-border flex items-center justify-between gap-3">
              <div className="flex items-center gap-3 min-w-0">
                <span className="text-sm font-semibold">{t("tools.output")}</span>
                {status && (
                  <span
                    className={cn(
                      "inline-flex items-center gap-1.5 text-xs font-medium",
                      status.tone === "rec" ? "text-red-500" : "text-primary"
                    )}
                  >
                    {status.tone === "rec" ? (
                      <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                    ) : (
                      <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    )}
                    <span className="truncate">{status.text}</span>
                  </span>
                )}
              </div>
              <div className="flex items-center gap-2 flex-shrink-0">
                <button
                  type="button"
                  onClick={onDownload}
                  disabled={!output}
                  className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs font-medium border border-border text-muted-foreground hover:bg-muted/60 hover:text-foreground transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  <Download className="w-3.5 h-3.5" />
                  {t("tools.download")}
                </button>
                <button
                  type="button"
                  onClick={onCopy}
                  disabled={!output}
                  className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs font-medium border border-border text-muted-foreground hover:bg-muted/60 hover:text-foreground transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  {copied ? <Check className="w-3.5 h-3.5 text-green-500" /> : <Copy className="w-3.5 h-3.5" />}
                  {t("tools.copy")}
                </button>
              </div>
            </div>
            <textarea
              value={output}
              onChange={(e) => setOutput(e.target.value)}
              placeholder={t("tools.placeholder")}
              className="w-full resize-y min-h-[420px] bg-background px-4 py-3 text-sm font-mono leading-relaxed focus:outline-none"
            />
          </section>
        </div>
      </div>
    </div>
  );
}
