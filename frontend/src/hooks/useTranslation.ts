import { useCallback } from "react";
import { useI18nStore, type Language } from "@/stores/i18nStore";
import en from "@/lib/translations/en.json";
import vi from "@/lib/translations/vi.json";

const translations: Record<Language, any> = { en, vi };

export const translate = (key: string, language: Language, params?: Record<string, any>): string => {
  const keys = key.split(".");
  let result = translations[language];
  
  for (const k of keys) {
    if (result && typeof result === "object" && k in result) {
      result = result[k];
    } else {
      return key; // Fallback to key itself
    }
  }
  
  if (typeof result !== "string") return key;

  if (params) {
    Object.entries(params).forEach(([k, v]) => {
      result = (result as string).replace(new RegExp(`{{${k}}}`, "g"), String(v));
    });
  }
  
  return result;
};

export const useTranslation = () => {
  const { language, setLanguage } = useI18nStore();

  const t = useCallback(
    (key: string, params?: Record<string, any>): string => {
      return translate(key, language, params);
    },
    [language]
  );

  return { t, language, setLanguage };
};
