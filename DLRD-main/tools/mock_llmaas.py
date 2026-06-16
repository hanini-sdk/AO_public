"""A tiny LOCAL stand-in for an OpenAI-compatible LLMAAS endpoint.

This is a development/demo helper ONLY — it lets you exercise the whole app
offline, with no real model, entirely on 127.0.0.1 (so it is consistent with the
egress guard: point apiBase at http://127.0.0.1:8900/v1). It returns canned,
deterministic enrichment JSON. It is never used on the work machine.

Run:  python tools/mock_llmaas.py --port 8900
Then in the app's config screen set:
    API base URL = http://127.0.0.1:8900/v1
    API key      = anything (not checked)
    Model        = mock-model
"""

from __future__ import annotations

import argparse
import json
import re
import time

import uvicorn
from fastapi import FastAPI, Request

app = FastAPI(title="mock-llmaas")

_LAYER_KEYS = [
    ("/api", "API"), ("route", "API"), ("controller", "API"), ("server", "API"),
    ("endpoint", "API"), ("handler", "API"),
    ("model", "Data"), ("repository", "Data"), ("repo", "Data"), ("schema", "Data"),
    ("/db", "Data"), ("/data", "Data"), ("entity", "Data"),
    (".jsx", "UI"), (".tsx", "UI"), ("component", "UI"), ("/ui", "UI"), ("view", "UI"),
    ("page", "UI"), ("widget", "UI"), ("css", "UI"),
    ("service", "Service"), ("usecase", "Service"), ("domain", "Service"),
    ("logic", "Service"), ("manager", "Service"),
    ("util", "Utility"), ("helper", "Utility"), ("config", "Utility"),
    ("common", "Utility"), ("/lib", "Utility"),
]


def guess_layer(path: str) -> str:
    p = path.lower()
    for needle, layer in _LAYER_KEYS:
        if needle in p:
            return layer
    return "Service"


@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    body = await req.json()
    messages = body.get("messages", [])
    joined = " ".join(m.get("content", "") for m in messages)
    last = messages[-1]["content"] if messages else ""

    # Mirror the backend's language clause so the mock can demonstrate that the
    # output-language setting actually drives the generated content language.
    french = "in French" in joined

    if "Reply with the single word: OK" in joined:
        content = "OK"

    elif "Return a JSON object with EXACTLY these keys" in joined:
        m = re.search(r"File:\s*(.+)", last)
        path = m.group(1).strip() if m else "file"
        layer = guess_layer(path)
        members = re.findall(r"-\s+(function|class)\s+(\S+)\s+\(lines", last)
        if french:
            kind_fr = {"function": "La fonction", "class": "La classe"}
            member_objs = []
            for kind, name in members:
                label = kind_fr.get(kind, "L'élément")
                member_objs.append({
                    "name": name,
                    "summary": f"{label} « {name} » contribue à {path}.",
                    "complexity": "simple",
                })
            payload = {
                "summary": f"[mock] {path} : implémente les responsabilités de la couche {layer} du projet exemple.",
                "layer": layer,
                "complexity": "moderate",
                "tags": ["mock", layer.lower(), "exemple"],
                "members": member_objs,
            }
        else:
            member_objs = [
                {"name": name, "summary": f"{kind.capitalize()} '{name}' contributes to {path}.",
                 "complexity": "simple"}
                for kind, name in members
            ]
            payload = {
                "summary": f"[mock] {path}: implements {layer.lower()} responsibilities for the sample project.",
                "layer": layer,
                "complexity": "moderate",
                "tags": ["mock", layer.lower(), "sample"],
                "members": member_objs,
            }
        content = json.dumps(payload, ensure_ascii=False)

    elif "KNOWLEDGE GRAPH CONTEXT:" in joined:
        # RAG chat request from /api/chat — the system prompt contains the
        # knowledge graph context. The mock extracts a few node IDs and the
        # user's question to build a varied, citation-bearing response.
        question = last.strip()

        # Extract node ids from context blocks formatted as [[node-id]] TYPE «name»
        node_refs = re.findall(r"\[\[([^\]]+)\]\]", joined)
        # Deduplicate while preserving order, take first 3 for the answer.
        seen: set = set()
        top_nodes: list = []
        for ref in node_refs:
            if ref not in seen:
                seen.add(ref)
                top_nodes.append(ref)
            if len(top_nodes) == 3:
                break

        # Also extract friendly names  «name» following each [[id]]
        name_map: dict = {}
        for m in re.finditer(r"\[\[([^\]]+)\]\]\s+\w+\s+«([^»]+)»", joined):
            name_map[m.group(1)] = m.group(2)

        # Build citation snippets
        citations = []
        for nid in top_nodes:
            name = name_map.get(nid, nid.split(":")[-1])
            citations.append(f"**{name}** ([[{nid}]])")

        if french:
            if citations:
                cit_str = ", ".join(citations)
                content = (
                    f"[mock — réponse simulée] D'après le graphe de connaissance, "
                    f"votre question « {question} » concerne notamment : {cit_str}.\n\n"
                    "Ces éléments font partie du projet exemple. "
                    "Remplacez le mock par un vrai modèle LLM pour obtenir des réponses précises."
                )
            else:
                content = (
                    f"[mock — réponse simulée] Votre question « {question} » "
                    "a bien été reçue, mais aucun nœud pertinent n'a été trouvé dans le contexte."
                )
        else:
            if citations:
                cit_str = ", ".join(citations)
                content = (
                    f"[mock response] Based on the knowledge graph, your question "
                    f'"{question}" relates to: {cit_str}.\n\n'
                    "These nodes are part of the sample project. "
                    "Replace the mock with a real LLM to get accurate answers."
                )
            else:
                content = (
                    f"[mock response] Your question \"{question}\" was received, "
                    "but no relevant nodes were found in the provided context."
                )

    elif french:
        content = "[mock] Un petit projet exemple illustrant le tableau de bord local Data-Lineage and Retro-Documentation."
    else:
        content = "[mock] A small sample project demonstrating the local Data-Lineage and Retro-Documentation dashboard."

    return {
        "id": "mock-cmpl", "object": "chat.completion", "created": int(time.time()),
        "model": body.get("model", "mock-model"),
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": "mock-model", "object": "model", "owned_by": "mock"}]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8900)
    args = ap.parse_args()
    print(f"  mock LLMAAS on http://127.0.0.1:{args.port}/v1  (model: mock-model)")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
