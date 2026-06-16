import { useState, useRef, useEffect, useCallback } from "react";
import ReactMarkdown from "react-markdown";
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

function AssistantBubble({ msg }: { msg: ChatMessage }) {
  const navigateToNode = useDashboardStore((s) => s.navigateToNode);

  // Strip [[node-id]] markers from rendered markdown (they become chips instead)
  const cleanedContent = msg.content.replace(/\[\[([^\]]+)\]\]/g, "`$1`");

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
          <ReactMarkdown>{cleanedContent}</ReactMarkdown>
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

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const submit = useCallback(async () => {
    const question = input.trim();
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
      submit();
    }
  }

  function handleSuggestion(q: string) {
    setInput(q);
    inputRef.current?.focus();
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
          <span className="text-[10px] px-1.5 py-0.5 rounded bg-accent/10 text-accent border border-accent/20 font-medium">
            RAG
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
            <AssistantBubble key={i} msg={msg} />
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
            onClick={submit}
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
