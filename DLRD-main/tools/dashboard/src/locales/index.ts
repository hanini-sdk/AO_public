import en from "./en";

// The dashboard UI chrome is English only. The "output language" setting
// (English or French) affects the LLM-generated graph content (summaries,
// tours) — which is data, not chrome — so it never changes these UI strings.
export type LocaleKey = "en";
export type Locale = typeof en;

export const locales: Record<LocaleKey, Locale> = { en };

export function getLocale(_key: LocaleKey): Locale {
  return en;
}

export function resolveLocaleKey(_lang: string | undefined): LocaleKey {
  return "en";
}

export { en };
