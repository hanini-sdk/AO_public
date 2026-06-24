import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
// Self-hosted fonts (replace the removed Google Fonts <link>). Bundled locally
// at build time → no CDN, no external network. Family names match index.css.
import "@fontsource/dm-serif-display/400.css";
import "@fontsource/inter/300.css";
import "@fontsource/inter/400.css";
import "@fontsource/inter/500.css";
import "@fontsource/inter/600.css";
import "@fontsource/jetbrains-mono/400.css";
import "@fontsource/jetbrains-mono/500.css";
import "./index.css";
import App from "./App";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
