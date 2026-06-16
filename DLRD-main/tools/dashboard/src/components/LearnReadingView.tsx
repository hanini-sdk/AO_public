import { useEffect, useState } from "react";
import type { ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import { useDashboardStore } from "../store";
import type { Story } from "../store";
import { useI18n } from "../contexts/I18nContext";

// Markdown renderers for the story body — mirrors the LearnPanel tour map, with
// heading/list overrides suited to a longer reading document.
const mdComponents = {
  p: ({ children }: { children?: ReactNode }) => (
    <p className="mb-3 leading-relaxed">{children}</p>
  ),
  strong: ({ children }: { children?: ReactNode }) => (
    <strong className="font-semibold text-text-primary">{children}</strong>
  ),
  em: ({ children }: { children?: ReactNode }) => <em className="italic">{children}</em>,
  h1: ({ children }: { children?: ReactNode }) => (
    <h3 className="text-base font-heading text-text-primary mt-4 mb-2">{children}</h3>
  ),
  h2: ({ children }: { children?: ReactNode }) => (
    <h4 className="text-sm font-semibold text-accent uppercase tracking-wider mt-4 mb-2">{children}</h4>
  ),
  h3: ({ children }: { children?: ReactNode }) => (
    <h4 className="text-sm font-semibold text-accent mt-3 mb-1.5">{children}</h4>
  ),
  ul: ({ children }: { children?: ReactNode }) => (
    <ul className="list-disc list-inside mb-3 space-y-1">{children}</ul>
  ),
  ol: ({ children }: { children?: ReactNode }) => (
    <ol className="list-decimal list-inside mb-3 space-y-1">{children}</ol>
  ),
  li: ({ children }: { children?: ReactNode }) => (
    <li className="text-text-secondary">{children}</li>
  ),
  code: ({ children }: { children?: ReactNode }) => (
    <code className="bg-elevated rounded px-1 py-0.5 text-[12px] font-mono">{children}</code>
  ),
};

/**
 * Full reading view for the project story. The story is loaded from /story.json
 * on startup; if none is cached yet (or the user forces a refresh), it asks the
 * backend to (re)generate it via the existing internal LLM service. Pure reader —
 * the click-a-section-to-highlight tour is a later feature (section node ids are
 * already stored for it).
 */
export default function LearnReadingView() {
  const story = useDashboardStore((s) => s.story);
  const setStory = useDashboardStore((s) => s.setStory);
  const closeLearnView = useDashboardStore((s) => s.closeLearnView);
  const { t } = useI18n();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const generate = async (force: boolean) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`/api/regenerate-story${force ? "?force=true" : ""}`, {
        method: "POST",
      });
      if (!res.ok) {
        const detail = await res.json().catch(() => null);
        throw new Error(detail?.detail || `Request failed (${res.status})`);
      }
      const data = await res.json();
      if (data?.story) setStory(data.story as Story);
      else throw new Error("Empty response");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  // First open with no cached story -> generate it (loading state). If the story
  // was already fetched on startup, this no-ops and the reader renders at once.
  useEffect(() => {
    if (!story && !loading) void generate(false);
    // run once on mount
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div
      className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50"
      onClick={closeLearnView}
    >
      <div
        className="glass rounded-lg shadow-2xl max-w-3xl w-full max-h-[85vh] flex flex-col m-4"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between gap-3 px-5 py-3 border-b border-border-subtle shrink-0">
          <h2 className="text-lg font-heading text-text-primary truncate">
            {story?.title ?? t.learnPanel.storyTitle}
          </h2>
          <div className="flex items-center gap-2 shrink-0">
            <button
              type="button"
              onClick={() => generate(true)}
              disabled={loading}
              title={t.learnPanel.regenerateTitle}
              className="text-[10px] font-semibold uppercase tracking-wider px-2.5 py-1 rounded border border-border-subtle text-text-muted hover:text-gold hover:border-gold/30 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {t.learnPanel.regenerate}
            </button>
            <button
              type="button"
              onClick={closeLearnView}
              className="text-text-muted hover:text-text-primary transition-colors"
              title={t.learnPanel.closeStory}
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto px-6 py-5 min-h-0">
          {loading && (
            <div className="h-full flex items-center justify-center text-sm text-text-muted">
              {t.learnPanel.generating}
            </div>
          )}
          {!loading && error && (
            <div className="text-sm">
              <p className="text-[#c97070] mb-2">{t.learnPanel.storyError}</p>
              <p className="text-text-muted text-xs font-mono break-words">{error}</p>
              <button
                type="button"
                onClick={() => generate(true)}
                className="mt-3 text-xs px-3 py-1.5 rounded bg-elevated border border-border-subtle text-text-secondary hover:text-gold transition-colors"
              >
                {t.learnPanel.retry}
              </button>
            </div>
          )}
          {!loading && !error && story && story.sections.length > 0 && (
            <article className="text-sm text-text-secondary">
              {story.sections.map((sec) => (
                <section key={sec.id} className="mb-6 last:mb-0">
                  <h3 className="text-base font-heading text-text-primary border-b border-border-subtle/60 pb-1 mb-3">
                    {sec.title}
                  </h3>
                  <ReactMarkdown components={mdComponents}>{sec.body}</ReactMarkdown>
                </section>
              ))}
            </article>
          )}
          {!loading && !error && story && story.sections.length === 0 && (
            <div className="text-sm text-text-muted">{t.learnPanel.storyEmpty}</div>
          )}
        </div>
      </div>
    </div>
  );
}
