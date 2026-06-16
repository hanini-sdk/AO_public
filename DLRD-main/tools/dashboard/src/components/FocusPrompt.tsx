import { useDashboardStore } from "../store";
import { useI18n } from "../contexts/I18nContext";

/**
 * Shown instead of the graph canvas when the project is too large to render
 * (oversized) and no lineage trace is active. The user searches for a node and
 * picks one to enter its exclusive lineage view — the only renderable path while
 * the full graph is over the size threshold.
 */
export default function FocusPrompt() {
  const graph = useDashboardStore((s) => s.graph);
  const searchQuery = useDashboardStore((s) => s.searchQuery);
  const searchResults = useDashboardStore((s) => s.searchResults);
  const setSearchQuery = useDashboardStore((s) => s.setSearchQuery);
  const focusLineageOn = useDashboardStore((s) => s.focusLineageOn);
  const nodesById = useDashboardStore((s) => s.nodesById);
  const { t } = useI18n();

  const count = graph?.nodes.length ?? 0;
  const results = searchResults.slice(0, 20);

  return (
    <div className="h-full w-full flex items-center justify-center bg-root rounded-lg p-6">
      <div className="w-full max-w-md flex flex-col gap-4">
        <div className="text-center">
          <h2 className="text-lg font-heading text-text-primary">{t.focusPrompt.largeTitle}</h2>
          <p className="text-sm text-text-secondary mt-1 tabular-nums">
            {count.toLocaleString()} {t.focusPrompt.nodesSuffix}
          </p>
          <p className="text-sm text-text-secondary mt-2">{t.focusPrompt.instruction}</p>
        </div>
        <input
          autoFocus
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder={t.focusPrompt.searchPlaceholder}
          className="w-full px-3 py-2 rounded-lg bg-elevated border border-border-subtle text-text-primary text-sm placeholder:text-text-muted focus:outline-none focus:border-gold/50"
        />
        {searchQuery.trim() && (
          <div className="max-h-80 overflow-y-auto rounded-lg border border-border-subtle bg-surface divide-y divide-border-subtle">
            {results.length === 0 ? (
              <div className="px-3 py-2 text-sm text-text-muted">{t.focusPrompt.noMatches}</div>
            ) : (
              results.map((r) => {
                const node = nodesById.get(r.nodeId);
                if (!node) return null;
                return (
                  <button
                    key={r.nodeId}
                    type="button"
                    onClick={() => focusLineageOn(r.nodeId)}
                    className="w-full text-left px-3 py-2 hover:bg-elevated/60 transition-colors flex items-center justify-between gap-2"
                  >
                    <span className="text-sm text-text-primary truncate">{node.name}</span>
                    <span className="text-[10px] uppercase tracking-wider text-text-muted shrink-0">
                      {node.type}
                    </span>
                  </button>
                );
              })
            )}
          </div>
        )}
      </div>
    </div>
  );
}
