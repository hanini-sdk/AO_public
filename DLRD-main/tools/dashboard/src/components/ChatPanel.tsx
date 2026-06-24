import { useState, useRef, useEffect, useCallback, useMemo, Children } from "react";
import type { ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import type { Components } from "react-markdown";
import { useDashboardStore } from "../store";

interface Source {
  id: string;
  name: string;
  type: string;
  filePath: string | null;
  summary: string;
}

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  error?: boolean;
}

const TYPE_ICONS: Record<string, string> = {
  file: "📄",
  function: "⚙️",
  class: "🔷",
  table: "🗄️",
  column: "📊",
  missing: "❓",
};

const SUGGESTED_QUESTIONS = [
  "How does this project work overall?",
  "Which files handle the API layer?",
  "What are the main data models?",
  "Which functions are most complex?",
];

// ── Deterministic inline entity linking ──────────────────────────────────
// We turn a mention in the chat answer into a clickable link ONLY when the
// token EXACTLY matches an existing graph node's id or name. Nothing is
// inferred from the model's wording: the match is against the real node set the
// dashboard already holds (no extra network call). A name shared by more than
// one node is ambiguous and left as plain text — we never risk navigating to
// the wrong node.

interface LinkData {
  // token (exact node id OR unambiguous node name) -> node id to navigate to
  index: Map<string, string>;
  // precompiled alternation of all tokens, longest-first, for prose scanning
  regex: RegExp | null;
}

const EMPTY_LINK_DATA: LinkData = { index: new Map(), regex: null };
const MAX_LINK_TOKENS = 5000; // bound the prose regex on very large graphs

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function buildLinkData(
  nodes: ReadonlyArray<{ id: string; name?: string | null }> | undefined,
): LinkData {
  if (!nodes || nodes.length === 0) return EMPTY_LINK_DATA;
  const index = new Map<string, string>();
  const nameCount = new Map<string, number>();
  const nameFirst = new Map<string, string>();
  for (const n of nodes) {
    index.set(n.id, n.id); // exact node ids (e.g. the [[id]] citations) are unique
    const nm = (n.name ?? "").trim();
    if (nm.length >= 2) {
      nameCount.set(nm, (nameCount.get(nm) ?? 0) + 1);
      if (!nameFirst.has(nm)) nameFirst.set(nm, n.id);
    }
  }
  // Only names that resolve to a single node become links (ambiguous -> skip).
  for (const [nm, count] of nameCount) {
    if (count === 1 && !index.has(nm)) index.set(nm, nameFirst.get(nm)!);
  }
  const keys = [...index.keys()].sort((a, b) => b.length - a.length);
  let regex: RegExp | null = null;
  if (keys.length > 0) {
    const alt = keys.slice(0, MAX_LINK_TOKENS).map(escapeRegExp).join("|");
    try {
      regex = new RegExp(alt, "g");
    } catch {
      regex = null;
    }
  }
  return { index, regex };
}

function NodeRef({ label, onClick }: { label: string; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={`Reveal ${label} in the graph`}
      className="inline text-accent underline decoration-dotted underline-offset-2 hover:text-accent-bright hover:decoration-solid transition-colors"
    >
      {label}
    </button>
  );
}

// Split a plain string into text + NodeRef links for every exact, unambiguous,
// whole-token match. Boundaries reject matches that sit inside a larger
// identifier (so "orders" is not linked inside "t_customer_orders").
function linkifyText(
  text: string,
  data: LinkData,
  onNavigate: (id: string) => void,
): ReactNode[] {
  if (!data.regex) return [text];
  const out: ReactNode[] = [];
  let last = 0;
  let key = 0;
  data.regex.lastIndex = 0;
  let m: RegExpExecArray | null;
  while ((m = data.regex.exec(text)) !== null) {
    const matched = m[0];
    const start = m.index;
    const end = start + matched.length;
    const before = start > 0 ? text[start - 1] : "";
    const after = end < text.length ? text[end] : "";
    const isWholeToken = !/[\w.:/-]/.test(before) && !/[\w.:/-]/.test(after);
    const id = data.index.get(matched);
    if (isWholeToken && id) {
      if (start > last) out.push(text.slice(last, start));
      const nodeId = id;
      out.push(<NodeRef key={`l${key++}`} label={matched} onClick={() => onNavigate(nodeId)} />);
      last = end;
    }
    if (data.regex.lastIndex === start) data.regex.lastIndex++; // guard against empty match
  }
  if (last < text.length) out.push(text.slice(last));
  return out.length > 0 ? out : [text];
}

// Linkify only the direct string children, leaving inline elements (code,
// strong, …) for their own renderers to handle.
function renderWithLinks(
  children: ReactNode,
  data: LinkData,
  onNavigate: (id: string) => void,
): ReactNode {
  return Children.map(children, (child) =>
    typeof child === "string" ? linkifyText(child, data, onNavigate) : child,
  );
}

function SourceChip({
  source,
  onClick,
}: {
  source: Source;
  onClick: () => void;
}) {
  const icon = TYPE_ICONS[source.type] ?? "📎";
  return (
    <button
      type="button"
      onClick={onClick}
      title={source.summary || source.name}
      className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-medium bg-elevated border border-border-medium text-text-secondary hover:text-accent hover:border-accent/40 transition-colors max-w-[160px]"
    >
      <span className="shrink-0">{icon}</span>
      <span className="truncate">{source.name}</span>
    </button>
  );
}

function AssistantBubble({ msg, linkData }: { msg: ChatMessage; linkData: LinkData }) {
  const navigateToNode = useDashboardStore((s) => s.navigateToNode);

  // Convert [[node-id]] citation markers to inline code; the code renderer below
  // then turns any that exactly match a node into clickable links (the source
  // chips still navigate too — this adds inline linking on top).
  const cleanedContent = msg.content.replace(/\[\[([^\]]+)\]\]/g, "`$1`");

  // Markdown renderers that linkify exact graph-entity mentions. Inline code is
  // linked on a full exact match (covers the [[id]] citations and back-ticked
  // names); prose/list/emphasis text is scanned for whole-token name matches.
  const components = useMemo<Components>(() => {
    const inline = (children: ReactNode) =>
      renderWithLinks(children, linkData, navigateToNode);
    return {
      code({ className, children, ...rest }) {
        const raw = Array.isArray(children) ? children.join("") : String(children ?? "");
        // Only inline code is a candidate: fenced blocks carry a language-
        // class or span multiple lines and must render verbatim.
        if (!className && !raw.includes("\n")) {
          const id = linkData.index.get(raw.trim());
          if (id) return <NodeRef label={raw} onClick={() => navigateToNode(id)} />;
        }
        return (
          <code className={className} {...rest}>
            {children}
          </code>
        );
      },
      p: ({ children }) => <p>{inline(children)}</p>,
      li: ({ children }) => <li>{inline(children)}</li>,
      strong: ({ children }) => <strong>{inline(children)}</strong>,
      em: ({ children }) => <em>{inline(children)}</em>,
    };
  }, [linkData, navigateToNode]);

  return (
    <div className="flex flex-col gap-1.5 items-start">
      <div
        className={`max-w-full rounded-lg px-3 py-2 text-sm leading-relaxed mr-2 ${
          msg.error
            ? "bg-red-900/20 border border-red-700/30 text-red-300"
            : "bg-elevated border border-border-subtle text-text-primary"
        }`}
      >
        <div className="prose-sm max-w-none text-inherit [&_p]:mb-1.5 [&_p:last-child]:mb-0 [&_ul]:pl-4 [&_li]:mb-0.5 [&_code]:text-accent [&_code]:bg-accent/10 [&_code]:px-1 [&_code]:rounded [&_strong]:text-text-primary [&_h1]:text-sm [&_h2]:text-sm [&_h3]:text-sm [&_h1]:font-semibold [&_h2]:font-semibold [&_h3]:font-semibold">
          <ReactMarkdown components={components}>{cleanedContent}</ReactMarkdown>
        </div>
      </div>
      {msg.sources && msg.sources.length > 0 && (
        <div className="flex flex-wrap gap-1 mr-2 max-w-full">
          {msg.sources.map((src) => (
            <SourceChip
              key={src.id}
              source={src}
              onClick={() => navigateToNode(src.id)}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function UserBubble({ content }: { content: string }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-[90%] rounded-lg px-3 py-2 text-sm leading-relaxed ml-4 bg-accent/10 border border-accent/20 text-text-primary">
        <span className="whitespace-pre-wrap">{content}</span>
      </div>
    </div>
  );
}

export default function ChatPanel() {
  const graph = useDashboardStore((s) => s.graph);

  // Deterministic entity-link index, rebuilt only when the graph changes. Used
  // to turn exact node mentions in answers into clickable links.
  const linkData = useMemo(() => buildLinkData(graph?.nodes), [graph]);

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const send = useCallback(async (override?: string) => {
    const question = (override ?? input).trim();
    if (!question || loading) return;

    // History sent to the API (no error messages, no sources metadata)
    const history = messages
      .filter((m) => !m.error)
      .map((m) => ({ role: m.role, content: m.content }));

    setMessages((prev) => [...prev, { role: "user", content: question }]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, history }),
      });

      const data = await res.json();

      if (!res.ok) {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: data.detail ?? "An error occurred. Please try again.",
            error: true,
          },
        ]);
      } else {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: data.answer ?? "",
            sources: data.sources ?? [],
          },
        ]);
      }
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "Network error — is the server running?",
          error: true,
        },
      ]);
    } finally {
      setLoading(false);
      requestAnimationFrame(() => inputRef.current?.focus());
    }
  }, [input, loading, messages]);

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }

  function handleSuggestion(q: string) {
    // Route a suggested question through the same send path as a typed message,
    // so its question and answer are appended to the persistent message list
    // (and survive every later turn) exactly like a typed message.
    send(q);
  }

  if (!graph) {
    return (
      <div className="h-full flex items-center justify-center p-6">
        <p className="text-text-muted text-sm text-center">
          Run an analysis first to enable the chat.
        </p>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col min-h-0">
      {/* ── Header ────────────────────────────────────────────── */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-border-subtle shrink-0">
        <div className="flex items-center gap-1.5">
          <span className="text-xs font-semibold uppercase tracking-wider text-text-muted">
            Chat
          </span>
        </div>
        {messages.length > 0 && (
          <button
            type="button"
            onClick={() => setMessages([])}
            className="text-[10px] text-text-muted hover:text-text-secondary transition-colors"
          >
            Clear
          </button>
        )}
      </div>

      {/* ── Message list ──────────────────────────────────────── */}
      <div className="flex-1 overflow-y-auto min-h-0 p-3 space-y-4">
        {/* Empty state */}
        {messages.length === 0 && (
          <div className="py-4 space-y-4">
            <div className="text-center space-y-1">
              <p className="text-2xl">💬</p>
              <p className="text-sm text-text-secondary">
                Ask anything about{" "}
                <span className="font-medium text-text-primary">
                  {graph.project.name}
                </span>
              </p>
              <p className="text-xs text-text-muted/70">
                Answers are grounded in the knowledge graph. Click a source chip
                to navigate to that node.
              </p>
            </div>
            <div className="space-y-1.5">
              {SUGGESTED_QUESTIONS.map((q) => (
                <button
                  key={q}
                  type="button"
                  onClick={() => handleSuggestion(q)}
                  className="w-full text-left text-xs px-3 py-2 rounded-lg bg-elevated border border-border-medium text-text-secondary hover:text-text-primary hover:border-border-strong transition-colors"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Messages */}
        {messages.map((msg, i) =>
          msg.role === "user" ? (
            <UserBubble key={i} content={msg.content} />
          ) : (
            <AssistantBubble key={i} msg={msg} linkData={linkData} />
          ),
        )}

        {/* Loading indicator */}
        {loading && (
          <div className="flex items-start">
            <div className="bg-elevated border border-border-subtle rounded-lg px-3 py-2">
              <span className="text-text-muted text-sm">
                <span className="animate-pulse">Thinking</span>
                <span className="animate-pulse delay-75">.</span>
                <span className="animate-pulse delay-150">.</span>
                <span className="animate-pulse delay-300">.</span>
              </span>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* ── Input ─────────────────────────────────────────────── */}
      <div className="shrink-0 border-t border-border-subtle p-2 space-y-1">
        <div className="flex items-end gap-1.5">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question… (Enter to send)"
            rows={2}
            disabled={loading}
            className="flex-1 resize-none rounded-lg bg-elevated border border-border-medium text-sm text-text-primary placeholder:text-text-muted/50 px-2.5 py-1.5 focus:outline-none focus:border-accent/50 disabled:opacity-50 transition-colors"
            style={{ minHeight: "52px" }}
          />
          <button
            type="button"
            onClick={() => send()}
            disabled={!input.trim() || loading}
            title="Send (Enter)"
            className="shrink-0 p-2 rounded-lg bg-accent/10 border border-accent/30 text-accent hover:bg-accent/20 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            <svg
              className="w-4 h-4"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
              />
            </svg>
          </button>
        </div>
        <p className="text-[10px] text-text-muted/40 pl-0.5">
          Shift+Enter for newline · sources are clickable graph nodes
        </p>
      </div>
    </div>
  );
}
