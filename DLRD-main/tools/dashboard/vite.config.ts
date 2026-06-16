import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "path";

// Self-contained build of the adapted dashboard.
//   * served by the local FastAPI backend under /dashboard/  (base below)
//   * @core/{schema,search,types} are aliased to the three
//     vendored source files (vendor/core/*) — pure TS that only needs zod +
//     fuse.js, so the heavy/native core package is never installed.
//   * no dev middleware: the backend serves /knowledge-graph.json etc.
export default defineConfig({
  base: "/dashboard/",
  resolve: {
    alias: {
      "@core/schema": path.resolve(__dirname, "vendor/core/schema.ts"),
      "@core/search": path.resolve(__dirname, "vendor/core/search.ts"),
      "@core/types": path.resolve(__dirname, "vendor/core/types.ts"),
    },
  },
  build: {
    outDir: path.resolve(__dirname, "../../web/dashboard"),
    emptyOutDir: true,
    chunkSizeWarningLimit: 2000,
    rollupOptions: {
      output: {
        manualChunks(id: string) {
          if (!id.includes("node_modules")) return;
          if (/[\\/]node_modules[\\/](react|react-dom|scheduler)[\\/]/.test(id)) return "react-vendor";
          if (id.includes("node_modules/@xyflow/")) return "xyflow";
          if (id.includes("node_modules/elkjs/")) return "elk";
          if (id.includes("node_modules/graphology")) return "graphology";
          if (id.includes("node_modules/@dagrejs/") || id.includes("node_modules/d3-force/")) return "graph-layout";
          if (
            id.includes("node_modules/react-markdown/") ||
            id.includes("node_modules/hast-util-to-jsx-runtime/") ||
            /[\\/]node_modules[\\/](remark|rehype|mdast|hast|unist|micromark|decode-named-character-reference|property-information|space-separated-tokens|comma-separated-tokens|html-url-attributes|devlop|bail|ccount|character-entities|is-plain-obj|trim-lines|trough|unified|vfile|zwitch)/.test(id)
          ) {
            return "markdown";
          }
        },
      },
    },
  },
  plugins: [react(), tailwindcss()],
});
