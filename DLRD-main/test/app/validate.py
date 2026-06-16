"""Deterministic validation: referential integrity, de-duplication, stats.

A final safety net over the assembled graph: drop duplicate node ids, drop edges
that reference missing nodes / are self-loops / are duplicates, and prune layer
and tour node-id lists to existing nodes. Mirrors the guarantees the dashboard's
own validator expects, so the graph renders with no hacks.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ValidationReport:
    issues: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def validate_graph(graph: dict) -> tuple[dict, ValidationReport]:
    report = ValidationReport()
    nodes = graph.get("nodes", []) or []
    edges = graph.get("edges", []) or []

    # --- de-duplicate nodes by id ---
    seen_ids: set[str] = set()
    clean_nodes: list[dict] = []
    dup_nodes = 0
    for n in nodes:
        nid = n.get("id")
        if not nid or nid in seen_ids:
            dup_nodes += 1
            continue
        seen_ids.add(nid)
        clean_nodes.append(n)
    if dup_nodes:
        report.issues.append(f"Removed {dup_nodes} duplicate/empty-id node(s).")

    # --- edges: referential integrity + de-dup + no self-loops ---
    clean_edges: list[dict] = []
    seen_edges: set[tuple] = set()
    dangling = self_loops = dup_edges = 0
    for e in edges:
        s, t, ty = e.get("source"), e.get("target"), e.get("type")
        if s not in seen_ids or t not in seen_ids:
            dangling += 1
            continue
        if s == t:
            self_loops += 1
            continue
        key = (s, t, ty)
        if key in seen_edges:
            dup_edges += 1
            continue
        seen_edges.add(key)
        clean_edges.append(e)
    if dangling:
        report.issues.append(f"Removed {dangling} edge(s) referencing missing nodes.")
    if self_loops:
        report.issues.append(f"Removed {self_loops} self-loop edge(s).")
    if dup_edges:
        report.issues.append(f"Removed {dup_edges} duplicate edge(s).")

    # --- layers: prune node ids to existing nodes; drop empty layers ---
    clean_layers: list[dict] = []
    for layer in graph.get("layers", []) or []:
        node_ids = [nid for nid in layer.get("nodeIds", []) if nid in seen_ids]
        if node_ids:
            clean_layers.append({**layer, "nodeIds": node_ids})

    # --- tour: prune node ids to existing nodes ---
    clean_tour = []
    for step in graph.get("tour", []) or []:
        clean_tour.append({**step, "nodeIds": [nid for nid in step.get("nodeIds", []) if nid in seen_ids]})

    out = {
        **graph,
        "nodes": clean_nodes,
        "edges": clean_edges,
        "layers": clean_layers,
        "tour": clean_tour,
    }
    report.stats = {
        "nodes": len(clean_nodes),
        "edges": len(clean_edges),
        "layers": len(clean_layers),
        "files": sum(1 for n in clean_nodes if n.get("type") == "file"),
        "functions": sum(1 for n in clean_nodes if n.get("type") == "function"),
        "classes": sum(1 for n in clean_nodes if n.get("type") == "class"),
    }
    return out, report
