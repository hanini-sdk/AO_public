import { useEffect, useRef } from "react";
import { useDashboardStore } from "../store";
import type { Story } from "../store";
import { useI18n } from "../contexts/I18nContext";
import type { KnowledgeGraph } from "@core/types";
import { filterNodes, filterEdges } from "../utils/filters";

function escapeXml(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// --- Story -> print-friendly HTML (browser "Save as PDF"; no new dependency) --
// Minimal inline markdown: escape, then **bold**, *italic*, `code`.
function mdInline(s: string): string {
  let out = escapeXml(s);
  out = out.replace(/`([^`]+)`/g, "<code>$1</code>");
  out = out.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  out = out.replace(/(^|[^*])\*([^*]+)\*/g, "$1<em>$2</em>");
  return out;
}

// Minimal block markdown: paragraphs, "- " bullet lists, "#" headings.
function mdBlockToHtml(body: string): string {
  const html: string[] = [];
  let list: string[] = [];
  let para: string[] = [];
  const flushList = () => {
    if (list.length) { html.push("<ul>" + list.map((li) => `<li>${mdInline(li)}</li>`).join("") + "</ul>"); list = []; }
  };
  const flushPara = () => {
    if (para.length) { html.push(`<p>${mdInline(para.join(" "))}</p>`); para = []; }
  };
  for (const raw of body.split("\n")) {
    const line = raw.replace(/\s+$/, "");
    const bullet = line.match(/^\s*[-*]\s+(.*)$/);
    const heading = line.match(/^(#{1,6})\s+(.*)$/);
    if (bullet) { flushPara(); list.push(bullet[1]); }
    else if (heading) { flushPara(); flushList(); const lvl = Math.min(heading[1].length + 1, 4); html.push(`<h${lvl}>${mdInline(heading[2])}</h${lvl}>`); }
    else if (line.trim() === "") { flushPara(); flushList(); }
    else { flushList(); para.push(line.trim()); }
  }
  flushPara();
  flushList();
  return html.join("\n");
}

function openStoryPrintWindow(story: Story): boolean {
  const w = window.open("", "_blank");
  if (!w) return false;
  const sections = story.sections
    .map((sec) => `<section><h2>${escapeXml(sec.title)}</h2>${mdBlockToHtml(sec.body)}</section>`)
    .join("\n");
  // No inline <script> (the app CSP forbids inline scripts); the opener triggers
  // print() below. Inline <style> is allowed by the CSP's style-src.
  const docHtml = `<!doctype html><html><head><meta charset="utf-8"><title>${escapeXml(story.title)}</title>
<style>
  @page { margin: 18mm; }
  body { font-family: Georgia, "Times New Roman", serif; color: #1a1a1a; line-height: 1.5; max-width: 720px; margin: 0 auto; padding: 24px; }
  h1 { font-size: 22px; border-bottom: 2px solid #888; padding-bottom: 6px; }
  h2 { font-size: 16px; margin-top: 26px; border-bottom: 1px solid #ccc; padding-bottom: 4px; page-break-after: avoid; }
  h3, h4 { font-size: 13px; margin-top: 14px; }
  p { margin: 0 0 10px; }
  ul { margin: 0 0 10px 18px; padding: 0; }
  li { margin: 2px 0; }
  code { background: #f0f0f0; padding: 1px 4px; border-radius: 3px; font-family: "Courier New", monospace; font-size: 12px; }
  section { page-break-inside: avoid; }
  .meta { color: #777; font-size: 11px; margin-bottom: 18px; }
</style></head>
<body>
  <h1>${escapeXml(story.title)}</h1>
  <div class="meta">${escapeXml(story.generatedAt || "")}</div>
  ${sections}
</body></html>`;
  w.document.open();
  w.document.write(docHtml);
  w.document.close();
  // Give the new window a moment to lay out, then print from the opener (avoids
  // a CSP-blocked inline script in the print document).
  window.setTimeout(() => {
    try { w.focus(); w.print(); } catch { /* user can print manually */ }
  }, 300);
  return true;
}

export default function ExportMenu() {
  const graph = useDashboardStore((s) => s.graph);
  const nodeIdToLayerIds = useDashboardStore((s) => s.nodeIdToLayerIds);
  const filters = useDashboardStore((s) => s.filters);
  const exportMenuOpen = useDashboardStore((s) => s.exportMenuOpen);
  const toggleExportMenu = useDashboardStore((s) => s.toggleExportMenu);
  const reactFlowInstance = useDashboardStore((s) => s.reactFlowInstance);
  const persona = useDashboardStore((s) => s.persona);
  const story = useDashboardStore((s) => s.story);
  const setStory = useDashboardStore((s) => s.setStory);
  const { t } = useI18n();

  const containerRef = useRef<HTMLDivElement>(null);

  // Close dropdown on outside click
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        if (exportMenuOpen) {
          toggleExportMenu();
        }
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [exportMenuOpen, toggleExportMenu]);

  const buildCleanSvg = () => {
    if (!reactFlowInstance) return null;

    const nodes = reactFlowInstance.getNodes();
    const edges = reactFlowInstance.getEdges();
    if (nodes.length === 0) return null;

    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    nodes.forEach((node) => {
      const x = node.position.x;
      const y = node.position.y;
      const width = (node.width ?? 200);
      const height = (node.height ?? 80);
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x + width);
      maxY = Math.max(maxY, y + height);
    });

    const padding = 40;
    const width = maxX - minX + padding * 2;
    const height = maxY - minY + padding * 2;
    const offsetX = -minX + padding;
    const offsetY = -minY + padding;

    let svgContent = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">`;
    svgContent += `<rect width="100%" height="100%" fill="#0a0a0a"/>`;

    edges.forEach((edge) => {
      const sourceNode = nodes.find((n) => n.id === edge.source);
      const targetNode = nodes.find((n) => n.id === edge.target);
      if (!sourceNode || !targetNode) return;

      const sx = sourceNode.position.x + (sourceNode.width ?? 200) / 2 + offsetX;
      const sy = sourceNode.position.y + (sourceNode.height ?? 80) / 2 + offsetY;
      const tx = targetNode.position.x + (targetNode.width ?? 200) / 2 + offsetX;
      const ty = targetNode.position.y + (targetNode.height ?? 80) / 2 + offsetY;

      svgContent += `<line x1="${sx}" y1="${sy}" x2="${tx}" y2="${ty}" stroke="rgba(212,165,116,0.3)" stroke-width="1.5"/>`;
    });

    nodes.forEach((node) => {
      if (node.type === "group") return;

      const x = node.position.x + offsetX;
      const y = node.position.y + offsetY;
      const w = node.width ?? 200;
      const h = node.height ?? 80;

      svgContent += `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="8" fill="#1a1a1a" stroke="rgba(212,165,116,0.2)" stroke-width="1"/>`;
      svgContent += `<text x="${x + w / 2}" y="${y + h / 2}" fill="#d4a574" text-anchor="middle" dominant-baseline="middle" font-size="12">${escapeXml(String(node.data.label ?? node.id))}</text>`;
    });

    svgContent += `</svg>`;
    return { svgContent, width, height };
  };

  const exportPNG = async () => {
    if (!reactFlowInstance) {
      alert("Graph not ready for export");
      return;
    }

    try {
      const result = buildCleanSvg();
      if (!result) {
        alert("No nodes to export");
        return;
      }

      const { svgContent, width, height } = result;
      const svgBlob = new Blob([svgContent], { type: "image/svg+xml;charset=utf-8" });
      const url = URL.createObjectURL(svgBlob);

      const img = new Image();
      img.onerror = () => {
        URL.revokeObjectURL(url);
        alert("Failed to export PNG: could not render graph as image.");
      };
      img.onload = () => {
        const canvas = document.createElement("canvas");
        canvas.width = width * 2;
        canvas.height = height * 2;
        const ctx = canvas.getContext("2d");
        if (!ctx) {
          URL.revokeObjectURL(url);
          alert("Failed to create canvas context");
          return;
        }
        ctx.drawImage(img, 0, 0, width * 2, height * 2);
        URL.revokeObjectURL(url);

        const filename = `${graph?.project.name ?? "knowledge-graph"}-export.png`;
        canvas.toBlob((blob) => {
          if (blob) {
            downloadBlob(blob, filename);
            toggleExportMenu();
          } else {
            alert("Failed to export PNG: image encoding failed.");
          }
        }, "image/png");
      };
      img.src = url;
    } catch (error) {
      console.error("PNG export failed:", error);
      alert(`Failed to export PNG: ${error instanceof Error ? error.message : String(error)}`);
    }
  };

  const exportSVG = () => {
    if (!reactFlowInstance) {
      alert("Graph not ready for export");
      return;
    }

    try {
      const result = buildCleanSvg();
      if (!result) {
        alert("No nodes to export");
        return;
      }

      const blob = new Blob([result.svgContent], { type: "image/svg+xml;charset=utf-8" });
      const filename = `${graph?.project.name ?? "knowledge-graph"}-export.svg`;
      downloadBlob(blob, filename);
      toggleExportMenu();
    } catch (error) {
      console.error("SVG export failed:", error);
      alert(`Failed to export SVG: ${error instanceof Error ? error.message : String(error)}`);
    }
  };

  const exportJSON = () => {
    if (!graph) {
      alert("No graph loaded");
      return;
    }

    try {
      // Apply persona and filters to create filtered graph
      // Non-technical persona: hide function/class sub-nodes, keep everything else
      const subFileTypes = new Set(["function", "class"]);
      let filteredGraphNodes = persona === "non-technical"
        ? graph.nodes.filter((n) => !subFileTypes.has(n.type))
        : graph.nodes;

      filteredGraphNodes = filterNodes(filteredGraphNodes, nodeIdToLayerIds, filters);
      const filteredNodeIds = new Set(filteredGraphNodes.map((n) => n.id));

      let filteredGraphEdges = graph.edges.filter(
        (e) => filteredNodeIds.has(e.source) && filteredNodeIds.has(e.target)
      );
      filteredGraphEdges = filterEdges(filteredGraphEdges, filteredNodeIds, filters);

      const filteredGraph: KnowledgeGraph = {
        ...graph,
        nodes: filteredGraphNodes,
        edges: filteredGraphEdges,
      };

      const json = JSON.stringify(filteredGraph, null, 2);
      const blob = new Blob([json], { type: "application/json" });
      const filename = `${graph.project.name ?? "knowledge-graph"}-export.json`;
      downloadBlob(blob, filename);
      toggleExportMenu();
    } catch (error) {
      console.error("JSON export failed:", error);
      alert(`Failed to export JSON: ${error instanceof Error ? error.message : String(error)}`);
    }
  };

  // Story PDF: render the ALREADY-CACHED story to PDF (a pure format conversion,
  // zero LLM calls). Only when no story is cached at all does it generate one
  // first (the single, cache-backed LLM path) before rendering.
  const exportStoryPDF = async () => {
    let s: Story | null = story;
    if (!s) {
      try {
        const res = await fetch("/api/regenerate-story", { method: "POST" });
        if (!res.ok) {
          const detail = await res.json().catch(() => null);
          throw new Error(detail?.detail || `Request failed (${res.status})`);
        }
        const data = await res.json();
        s = (data?.story as Story) ?? null;
        if (s) setStory(s);
      } catch (e) {
        alert(`Could not prepare the story for export: ${e instanceof Error ? e.message : String(e)}`);
        return;
      }
    }
    if (!s || s.sections.length === 0) {
      alert("No story is available to export yet.");
      return;
    }
    if (!openStoryPrintWindow(s)) {
      alert("Could not open the print window. Please allow pop-ups for this page.");
      return;
    }
    toggleExportMenu();
  };

  return (
    <div ref={containerRef} className="relative">
      <button
        onClick={toggleExportMenu}
        className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm bg-elevated text-text-secondary hover:text-text-primary transition-colors"
        title={t.export.title}
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
            d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
          />
        </svg>
        {t.export.label}
      </button>

      {exportMenuOpen && (
        <div className="absolute right-0 top-full mt-2 w-52 glass rounded-lg shadow-xl overflow-hidden animate-fade-slide-in z-50">
          <div className="p-2">
            <button
              onClick={exportPNG}
              disabled={!reactFlowInstance}
              className="w-full flex items-center gap-3 px-3 py-2 text-sm text-text-primary hover:bg-elevated transition-colors rounded-lg text-left disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
              <span>{t.export.asPNG}</span>
            </button>
            <button
              onClick={exportSVG}
              disabled={!reactFlowInstance}
              className="w-full flex items-center gap-3 px-3 py-2 text-sm text-text-primary hover:bg-elevated transition-colors rounded-lg text-left disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zm0 0h12a2 2 0 002-2v-4a2 2 0 00-2-2h-2.343M11 7.343l1.657-1.657a2 2 0 012.828 0l2.829 2.829a2 2 0 010 2.828l-8.486 8.485M7 17h.01" />
              </svg>
              <span>{t.export.asSVG}</span>
            </button>
            <button
              onClick={exportJSON}
              disabled={!graph}
              className="w-full flex items-center gap-3 px-3 py-2 text-sm text-text-primary hover:bg-elevated transition-colors rounded-lg text-left disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
              </svg>
              <span>{t.export.asJSON}</span>
            </button>
            <button
              onClick={exportStoryPDF}
              disabled={!graph}
              className="w-full flex items-center gap-3 px-3 py-2 text-sm text-text-primary hover:bg-elevated transition-colors rounded-lg text-left disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              <span>{t.export.asStoryPDF}</span>
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
