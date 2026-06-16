// Transitive data-lineage tracing over the structural graph.
//
// The backend (phases A-E) emits directed edges we can walk without any schema
// change:
//   file --writes_to--> table   (the file writes or purges the table)
//   file --reads_from--> table  (the file reads the table)
//   table --reads_from--> table (provenance: source table populated from target)
//   file --depends_on--> file   ("feeds": consumer depends on producer)
//
// Upstream of a table X = where its data comes from: the files that write/purge
// X and the tables those files read, recursively. Downstream of X = where it
// propagates: the files that read X and the tables those files write,
// recursively. Traversal alternates table <-> file and is capped by a node
// budget (raise it via "show more") so a hot table can't expand the whole graph.

import type { KnowledgeGraph } from "@core/types";

export type LineageDirection = "up" | "down" | "both";

export interface LineageResult {
  nodeIds: Set<string>; // every node in the lineage subgraph, including the root
  truncated: boolean; // hit the node budget — more lineage exists
  total: number; // nodeIds.size, for the status badge
}

export interface LineageIndex {
  nodeType: Map<string, string>;
  producersOfTable: Map<string, string[]>; // table -> files that write/purge it
  readsOfFile: Map<string, string[]>; // file  -> tables it reads
  provUp: Map<string, string[]>; // table X -> tables X is populated from
  consumersOfTable: Map<string, string[]>; // table -> files that read it
  writesOfFile: Map<string, string[]>; // file  -> tables it writes
  provDown: Map<string, string[]>; // table Y -> tables populated from Y
  colUp: Map<string, string[]>; // column C -> columns C is derived from
  colDown: Map<string, string[]>; // column C -> columns derived from C
  tableOfColumn: Map<string, string>; // column -> its owning table (via contains)
}

function push(m: Map<string, string[]>, k: string, v: string): void {
  const arr = m.get(k);
  if (arr) arr.push(v);
  else m.set(k, [v]);
}

/** Pre-index the graph's lineage edges once per graph (memoize on `graph`). */
export function buildLineageIndex(graph: KnowledgeGraph): LineageIndex {
  const nodeType = new Map<string, string>();
  for (const n of graph.nodes) nodeType.set(n.id, n.type);

  const idx: LineageIndex = {
    nodeType,
    producersOfTable: new Map(),
    readsOfFile: new Map(),
    provUp: new Map(),
    consumersOfTable: new Map(),
    writesOfFile: new Map(),
    provDown: new Map(),
    colUp: new Map(),
    colDown: new Map(),
    tableOfColumn: new Map(),
  };

  for (const e of graph.edges) {
    const s = e.source;
    const t = e.target;
    if (e.type === "writes_to") {
      // file --writes_to--> table (write or purge both count as "producing"/touching X)
      push(idx.producersOfTable, t, s);
      push(idx.writesOfFile, s, t);
    } else if (e.type === "reads_from") {
      const st = nodeType.get(s);
      const tt = nodeType.get(t);
      if (st === "table" && tt === "table") {
        // table s populated from table t -> t is upstream of s
        push(idx.provUp, s, t);
        push(idx.provDown, t, s);
      } else if (st === "column" && tt === "column") {
        // column s derived from column t -> t is upstream of s
        push(idx.colUp, s, t);
        push(idx.colDown, t, s);
      } else {
        // file --reads_from--> table
        push(idx.readsOfFile, s, t);
        push(idx.consumersOfTable, t, s);
      }
    } else if (e.type === "contains") {
      // table --contains--> column: remember the owning table so a column trace
      // can also light up the tables it touches (used as bridge context below).
      if (nodeType.get(s) === "table" && nodeType.get(t) === "column") {
        idx.tableOfColumn.set(t, s);
      }
    }
    // depends_on (file->file "feeds") is intentionally not walked: the
    // file<->table chain already reaches the producer/consumer files. Such an
    // edge is still highlighted whenever both its endpoints land in the set.
  }
  return idx;
}

/**
 * Walk the lineage of `rootId` in the requested direction, up to `limit` nodes.
 * Returns the reached node ids (including the root) and whether the budget
 * clipped the result.
 */
export function traceLineage(
  index: LineageIndex,
  rootId: string,
  direction: LineageDirection,
  limit: number,
): LineageResult {
  const visited = new Set<string>([rootId]);
  let truncated = false;

  const walk = (downstream: boolean): void => {
    const queue: string[] = [rootId];
    while (queue.length > 0) {
      const cur = queue.shift() as string;
      const type = index.nodeType.get(cur);
      let nexts: string[] = [];
      if (type === "table") {
        nexts = downstream
          ? [...(index.consumersOfTable.get(cur) ?? []), ...(index.provDown.get(cur) ?? [])]
          : [...(index.producersOfTable.get(cur) ?? []), ...(index.provUp.get(cur) ?? [])];
      } else if (type === "file") {
        nexts = downstream
          ? index.writesOfFile.get(cur) ?? []
          : index.readsOfFile.get(cur) ?? [];
      } else if (type === "column") {
        // column-level lineage: walk only column<->column edges so a column
        // trace stays at column granularity (its parent table is added as
        // context after the walk, not traversed into table lineage).
        nexts = downstream
          ? index.colDown.get(cur) ?? []
          : index.colUp.get(cur) ?? [];
      }
      for (const n of nexts) {
        if (visited.has(n)) continue;
        if (visited.size >= limit) {
          truncated = true;
          return;
        }
        visited.add(n);
        queue.push(n);
      }
    }
  };

  if (direction !== "down") walk(false); // upstream
  if (direction !== "up") walk(true); // downstream

  // Bridge any traced column to its owning table so the involved tables light
  // up too (especially useful when column nodes themselves are hidden). Added
  // as context only — not walked further, so a column trace never balloons into
  // the table's full lineage.
  for (const id of [...visited]) {
    if (index.nodeType.get(id) === "column") {
      const tbl = index.tableOfColumn.get(id);
      if (tbl) visited.add(tbl);
    }
  }

  return { nodeIds: visited, truncated, total: visited.size };
}
