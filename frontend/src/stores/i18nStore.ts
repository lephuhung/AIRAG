import { create } from "zustand";
import { persist } from "zustand/middleware";

export type Language = "en" | "vi";

interface I18nState {
  language: Language;
  setLanguage: (lang: Language) => void;
}

export const useI18nStore = create<I18nState>()(
  persist(
    (set) => ({
      language: "vi", // Default language
      setLanguage: (language) => set({ language }),
    }),
    {
      name: "hrag-language",
    }
  )
);
