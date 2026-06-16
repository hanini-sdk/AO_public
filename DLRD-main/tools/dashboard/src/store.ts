import { create } from "zustand";
import { SearchEngine } from "@core/search";
import type { SearchResult } from "@core/search";
import type { GraphIssue } from "@core/schema";
import type {
  GraphNode,
  KnowledgeGraph,
  TourStep,
} from "@core/types";
import type { ReactFlowInstance } from "@xyflow/react";
import type { LineageDirection } from "./utils/lineage";

export type Persona = "non-technical" | "junior" | "experienced";

// Project story: one ordered narrative synthesized by the backend from the
// graph's node summaries + lineage. Each section records the ids of the nodes it
// describes (for a later interactive tour). Served from /story.json (optional).
export interface StorySection {
  id: string;
  title: string;
  body: string;       // markdown
  nodeIds: string[];
}
export interface Story {
  version: string;
  generatedAt: string;
  graphFingerprint?: string;
  language?: string;
  title: string;
  sections: StorySection[];
}

// Lineage trace node budget: how many nodes a single trace highlights before
// it stops and offers "show more" (which raises the budget by one step).
export const DEFAULT_LINEAGE_LIMIT = 500;
export const LINEAGE_LIMIT_STEP = 250;
// Above this node count the full graph is too large to render: the dashboard
// shows a focus prompt and renders only a chosen node's lineage subgraph. The
// lineage budget bounds the trace and serves as the render cap (a column trace
// may add a few owning-table nodes on top of the budget), keeping it far smaller
// than the full graph.
export const LARGE_GRAPH_THRESHOLD = 2000;
export type NavigationLevel = "overview" | "layer-detail";
export type NodeType = "file" | "function" | "class" | "module" | "concept" | "config" | "document" | "service" | "table" | "column" | "endpoint" | "pipeline" | "schema" | "resource" | "domain" | "flow" | "step" | "article" | "entity" | "topic" | "claim" | "source";
export type Complexity = "simple" | "moderate" | "complex";
export type EdgeCategory = "structural" | "behavioral" | "data-flow" | "dependencies" | "semantic" | "infrastructure" | "domain" | "knowledge";
export type ViewMode = "structural" | "domain" | "knowledge";
export type DetailLevel = "file" | "class";

export interface FilterState {
  nodeTypes: Set<NodeType>;
  complexities: Set<Complexity>;
  layerIds: Set<string>;
  edgeCategories: Set<EdgeCategory>;
}

export const ALL_NODE_TYPES: NodeType[] = ["file", "function", "class", "module", "concept", "config", "document", "service", "table", "column", "endpoint", "pipeline", "schema", "resource", "domain", "flow", "step", "article", "entity", "topic", "claim", "source"];
export const ALL_COMPLEXITIES: Complexity[] = ["simple", "moderate", "complex"];
export const ALL_EDGE_CATEGORIES: EdgeCategory[] = ["structural", "behavioral", "data-flow", "dependencies", "semantic", "infrastructure", "domain", "knowledge"];

export const EDGE_CATEGORY_MAP: Record<EdgeCategory, string[]> = {
  structural: ["imports", "exports", "contains", "inherits", "implements"],
  behavioral: ["calls", "subscribes", "publishes", "middleware"],
  "data-flow": ["reads_from", "writes_to", "transforms", "validates"],
  dependencies: ["depends_on", "tested_by", "configures"],
  semantic: ["related", "similar_to"],
  infrastructure: ["deploys", "serves", "provisions", "triggers", "migrates", "documents", "routes", "defines_schema"],
  domain: ["contains_flow", "flow_step", "cross_domain"],
  knowledge: ["cites", "contradicts", "builds_on", "exemplifies", "categorized_under", "authored_by"],
};

export const DOMAIN_EDGE_TYPES = EDGE_CATEGORY_MAP.domain;

const DEFAULT_FILTERS: FilterState = {
  nodeTypes: new Set<NodeType>(ALL_NODE_TYPES),
  complexities: new Set<Complexity>(ALL_COMPLEXITIES),
  layerIds: new Set<string>(),
  edgeCategories: new Set<EdgeCategory>(ALL_EDGE_CATEGORIES),
};

/** Categories used for node type filter toggles. Single source of truth for NodeCategory. */
export type NodeCategory = "code" | "config" | "docs" | "infra" | "data" | "domain" | "knowledge";

/**
 * Build the (id → node) and (id → layerId) lookup maps that the rest of
 * the dashboard reads via store selectors. Centralised so `setGraph` and
 * any future graph-replacement path stay in sync.
 *
 * Two layer indexes, intentionally distinct:
 *
 * - `nodeIdToLayerId` preserves the prior `findNodeLayer` "first matching
 *   layer wins" semantics — if a node id appears in multiple layers
 *   (rare but legal in the schema), the first occurrence in `graph.layers`
 *   order is the one we map to. Drives navigation (drillIntoLayer, tour
 *   step → layer, sidebar history) where a single canonical layer is the
 *   right answer.
 *
 * - `nodeIdToLayerIds` records *every* layer a node belongs to. Drives
 *   membership queries (filterNodes) where the prior `Layer[] +
 *   layer.nodeIds.includes` shape was any-layer-wins — a node in L1 and
 *   L2 with only L2 selected must still pass. Collapsing to first-wins
 *   for filtering would be a silent regression.
 */
function buildGraphIndexes(graph: KnowledgeGraph): {
  nodesById: Map<string, GraphNode>;
  nodeIdToLayerId: Map<string, string>;
  nodeIdToLayerIds: Map<string, Set<string>>;
} {
  const nodesById = new Map<string, GraphNode>();
  for (const node of graph.nodes) nodesById.set(node.id, node);
  const nodeIdToLayerId = new Map<string, string>();
  const nodeIdToLayerIds = new Map<string, Set<string>>();
  for (const layer of graph.layers) {
    for (const nid of layer.nodeIds) {
      if (!nodeIdToLayerId.has(nid)) nodeIdToLayerId.set(nid, layer.id);
      let set = nodeIdToLayerIds.get(nid);
      if (!set) {
        set = new Set<string>();
        nodeIdToLayerIds.set(nid, set);
      }
      set.add(layer.id);
    }
  }
  return { nodesById, nodeIdToLayerId, nodeIdToLayerIds };
}

/** Maximum number of entries in the sidebar navigation history. */
const MAX_HISTORY = 50;

interface DashboardStore {
  graph: KnowledgeGraph | null;
  /** id → node lookup, rebuilt by setGraph. Empty before any graph loads. */
  nodesById: Map<string, GraphNode>;
  /** id → layer id (first-matching-layer wins), rebuilt by setGraph. Empty before any graph loads. */
  nodeIdToLayerId: Map<string, string>;
  /** id → set of every layer the node belongs to, rebuilt by setGraph. Empty before any graph loads. */
  nodeIdToLayerIds: Map<string, Set<string>>;
  selectedNodeId: string | null;
  searchQuery: string;
  searchResults: SearchResult[];
  searchEngine: SearchEngine | null;
  searchMode: "fuzzy" | "semantic";
  setSearchMode: (mode: "fuzzy" | "semantic") => void;

  // Lens navigation
  navigationLevel: NavigationLevel;
  activeLayerId: string | null;

  codeViewerOpen: boolean;
  codeViewerNodeId: string | null;
  codeViewerExpanded: boolean;

  tourActive: boolean;
  currentTourStep: number;
  tourHighlightedNodeIds: string[];

  persona: Persona;

  diffMode: boolean;
  changedNodeIds: Set<string>;
  affectedNodeIds: Set<string>;

  // Focus mode: isolate a node's 1-hop neighborhood
  focusNodeId: string | null;

  // Lineage trace: highlight a table's full transitive upstream/downstream and
  // dim the rest. Root is the current selectedNodeId; cleared when selection
  // changes. `lineageLimit` caps the traced node count ("show more" raises it).
  lineageActive: boolean;
  lineageDirection: LineageDirection;
  lineageLimit: number;

  // True when the graph exceeds LARGE_GRAPH_THRESHOLD nodes: the full graph is
  // not rendered; the focus prompt picks a node to view its lineage instead.
  oversized: boolean;

  // Sidebar navigation history (stack of visited node IDs)
  nodeHistory: string[];

  // Filter & Export features
  filters: FilterState;
  filterPanelOpen: boolean;
  exportMenuOpen: boolean;
  pathFinderOpen: boolean;
  reactFlowInstance: ReactFlowInstance | null;

  // Node type category filters
  nodeTypeFilters: Record<NodeCategory, boolean>;
  toggleNodeTypeFilter: (category: NodeCategory) => void;

  // Detail level: "file" shows only file nodes (architecture view),
  // "class" shows files + class nodes (code structure view) with optional function expansion.
  detailLevel: DetailLevel;
  setDetailLevel: (level: DetailLevel) => void;
  showFunctionsInClassView: boolean;
  toggleShowFunctionsInClassView: () => void;

  // In-graph node clustering: group the layer's nodes into folder/community
  // container boxes. OFF by default — the graph renders flat until the user
  // opts in via the "Clusterize" toggle. (This is distinct from the separate
  // layer-overview screen, which is unaffected.)
  clusterEnabled: boolean;
  toggleClusterize: () => void;

  // Column granularity: render the used columns as nodes attached to their
  // table. OFF by default (top-bar COLUMNS toggle); columns stay hidden until on.
  columnsView: boolean;
  toggleColumnsView: () => void;

  // Missing-reference overlay: render red placeholder nodes for scripts/SQL files
  // referenced but not found in the project. OFF by default; when off the graph
  // is byte-for-byte identical to before the feature (missing nodes are layerless
  // so they never enter the overview or layer counts, and the layer-detail view
  // is gated on this flag).
  missingView: boolean;
  toggleMissingView: () => void;

  // Project story (Learn reading view). Loaded from /story.json on startup.
  story: Story | null;
  setStory: (story: Story | null) => void;
  learnViewOpen: boolean;
  openLearnView: () => void;
  closeLearnView: () => void;

  setGraph: (graph: KnowledgeGraph) => void;
  selectNode: (nodeId: string | null) => void;
  navigateToNode: (nodeId: string) => void;
  navigateToNodeInLayer: (nodeId: string) => void;
  navigateToHistoryIndex: (index: number) => void;
  goBackNode: () => void;
  drillIntoLayer: (layerId: string) => void;
  navigateToOverview: () => void;
  setFocusNode: (nodeId: string | null) => void;
  toggleLineageTrace: () => void;
  setLineageDirection: (direction: LineageDirection) => void;
  showMoreLineage: () => void;
  // Enter the exclusive lineage view rooted at nodeId (used by the focus prompt,
  // reachable on demand). Forces layer-detail; a column root also enables COL.
  focusLineageOn: (nodeId: string) => void;
  setSearchQuery: (query: string) => void;
  setPersona: (persona: Persona) => void;
  openCodeViewer: (nodeId: string) => void;
  closeCodeViewer: () => void;
  expandCodeViewer: () => void;
  collapseCodeViewer: () => void;

  setDiffOverlay: (changed: string[], affected: string[]) => void;
  toggleDiffMode: () => void;
  clearDiffOverlay: () => void;

  toggleFilterPanel: () => void;
  toggleExportMenu: () => void;
  togglePathFinder: () => void;
  setReactFlowInstance: (instance: ReactFlowInstance | null) => void;
  setFilters: (filters: Partial<FilterState>) => void;
  resetFilters: () => void;
  hasActiveFilters: () => boolean;

  startTour: () => void;
  stopTour: () => void;
  setTourStep: (step: number) => void;
  nextTourStep: () => void;
  prevTourStep: () => void;

  // View mode
  viewMode: ViewMode;
  isKnowledgeGraph: boolean;
  domainGraph: KnowledgeGraph | null;
  activeDomainId: string | null;

  setDomainGraph: (graph: KnowledgeGraph) => void;
  setViewMode: (mode: ViewMode) => void;
  setIsKnowledgeGraph: (value: boolean) => void;
  navigateToDomain: (domainId: string) => void;
  clearActiveDomain: () => void;

  // Container expand/collapse + lazy layout caches
  expandedContainers: Set<string>;
  toggleContainer: (containerId: string) => void;
  expandContainer: (containerId: string) => void;
  collapseContainer: (containerId: string) => void;
  collapseAllContainers: () => void;
  /** Container the user just manually expanded; viewport should lock onto it. Cleared by GraphView once the lock is applied. */
  pendingFocusContainer: string | null;
  setPendingFocusContainer: (containerId: string | null) => void;
  /** True while TourFitView is waiting for highlighted nodes to materialise (Stage 2 layout in progress). Drives the "Computing layout…" overlay. */
  tourFitPending: boolean;
  setTourFitPending: (pending: boolean) => void;

  containerLayoutCache: Map<
    string,
    {
      childPositions: Map<string, { x: number; y: number }>;
      actualSize: { width: number; height: number };
    }
  >;
  setContainerLayout: (
    containerId: string,
    childPositions: Map<string, { x: number; y: number }>,
    actualSize: { width: number; height: number },
  ) => void;
  clearContainerLayouts: () => void;

  containerSizeMemory: Map<string, { width: number; height: number }>;

  stage1Tick: number;
  bumpStage1Tick: () => void;

  // Layout-time issues (e.g. ELK input repair). Funneled into the
  // WarningBanner alongside graph-validation issues.
  layoutIssues: GraphIssue[];
  appendLayoutIssues: (issues: GraphIssue[]) => void;
  clearLayoutIssues: () => void;
}

function getSortedTour(graph: KnowledgeGraph): TourStep[] {
  const tour = graph.tour ?? [];
  return [...tour].sort((a, b) => a.order - b.order);
}

/** Navigate tour step to the correct layer for the first highlighted node. */
function navigateTourToLayer(
  nodeIdToLayerId: Map<string, string>,
  nodeIds: string[],
): Partial<DashboardStore> {
  if (nodeIds.length === 0) return {};
  const layerId = nodeIdToLayerId.get(nodeIds[0]);
  if (layerId) {
    return {
      navigationLevel: "layer-detail" as const,
      activeLayerId: layerId,
    };
  }
  return {};
}

/**
 * Container ids derive from per-layer state — folder names in folder-strategy
 * layers, community indices (`container:cluster-N`) in community-strategy
 * layers — and collide across layers (e.g. API Contracts and Load Testing
 * both produce `container:cluster-0`). When a tour step crosses layers we
 * must drop the previous layer's container caches so Stage 2 actually re-
 * runs for the new layer's children. Mirrors the reset block in
 * `drillIntoLayer`.
 */
function layerResetIfChanged(
  layerNav: Partial<DashboardStore>,
  prevLayerId: string | null,
): Partial<DashboardStore> {
  const next = layerNav.activeLayerId;
  if (!next || next === prevLayerId) return {};
  return {
    containerLayoutCache: new Map(),
    containerSizeMemory: new Map(),
    expandedContainers: new Set(),
    // Drop any pending focus too — its id was scoped to the previous
    // layer and would otherwise re-collide with a same-id container in
    // the new layer for the duration of the 1.2s timer.
    pendingFocusContainer: null,
  };
}

export const useDashboardStore = create<DashboardStore>()((set, get) => ({
  graph: null,
  nodesById: new Map<string, GraphNode>(),
  nodeIdToLayerId: new Map<string, string>(),
  nodeIdToLayerIds: new Map<string, Set<string>>(),
  selectedNodeId: null,
  searchQuery: "",
  searchResults: [],
  searchEngine: null,
  searchMode: "fuzzy",

  navigationLevel: "overview",
  activeLayerId: null,
  codeViewerOpen: false,
  codeViewerNodeId: null,
  codeViewerExpanded: false,

  tourActive: false,
  currentTourStep: 0,
  tourHighlightedNodeIds: [],

  // Default to "experienced" (Deep Dive): the "junior"/Learn persona's only
  // distinguishing feature is the guided tour, which is a future (v2) feature,
  // so its selector tab is shown disabled. Same node detail as Learn, minus the
  // (currently empty) tour panel.
  persona: "experienced",

  diffMode: false,
  changedNodeIds: new Set<string>(),
  affectedNodeIds: new Set<string>(),

  focusNodeId: null,
  lineageActive: false,
  lineageDirection: "both",
  lineageLimit: DEFAULT_LINEAGE_LIMIT,
  oversized: false,
  nodeHistory: [],

  filters: { ...DEFAULT_FILTERS, nodeTypes: new Set(DEFAULT_FILTERS.nodeTypes), complexities: new Set(DEFAULT_FILTERS.complexities), layerIds: new Set(DEFAULT_FILTERS.layerIds), edgeCategories: new Set(DEFAULT_FILTERS.edgeCategories) },
  filterPanelOpen: false,
  exportMenuOpen: false,
  pathFinderOpen: false,
  reactFlowInstance: null,

  nodeTypeFilters: { code: true, config: true, docs: true, infra: true, data: true, domain: true, knowledge: true },

  toggleNodeTypeFilter: (category) =>
    set((state) => ({
      nodeTypeFilters: {
        ...state.nodeTypeFilters,
        [category]: !state.nodeTypeFilters[category],
      },
      // Filter changes shift container.nodeIds; cached child positions
      // may reference filtered-out children. Drop the cache so Stage 2
      // recomputes against the current set.
      containerLayoutCache: new Map(),
      containerSizeMemory: new Map(),
      expandedContainers: new Set(),
      pendingFocusContainer: null,
    })),

  detailLevel: "file",
  setDetailLevel: (level) =>
    set({
      detailLevel: level,
      // Detail level changes which nodes are visible; cached positions stale.
      // Reset fn toggle so it doesn't resurrect when re-entering class view.
      showFunctionsInClassView: false,
      containerLayoutCache: new Map(),
      containerSizeMemory: new Map(),
      expandedContainers: new Set(),
      pendingFocusContainer: null,
    }),

  showFunctionsInClassView: false,
  toggleShowFunctionsInClassView: () =>
    set((state) => ({
      showFunctionsInClassView: !state.showFunctionsInClassView,
      containerLayoutCache: new Map(),
      containerSizeMemory: new Map(),
      expandedContainers: new Set(),
      pendingFocusContainer: null,
    })),

  clusterEnabled: false,
  toggleClusterize: () =>
    set((state) => ({
      clusterEnabled: !state.clusterEnabled,
      // Clustering changes node grouping + layout, so drop the container
      // caches like every other layout-affecting toggle. Selection, lineage
      // trace, and focus are intentionally preserved across the switch.
      containerLayoutCache: new Map(),
      containerSizeMemory: new Map(),
      expandedContainers: new Set(),
      pendingFocusContainer: null,
    })),

  columnsView: false,
  toggleColumnsView: () =>
    set((state) => ({
      columnsView: !state.columnsView,
      // Showing/hiding column nodes changes the visible node set + layout, so
      // drop the container caches like the other view toggles. Selection and
      // lineage trace are preserved.
      containerLayoutCache: new Map(),
      containerSizeMemory: new Map(),
      expandedContainers: new Set(),
      pendingFocusContainer: null,
    })),

  missingView: false,
  toggleMissingView: () =>
    set((state) => ({
      missingView: !state.missingView,
      // Showing/hiding missing nodes changes the visible node set + layout, so
      // drop the container caches like the other view toggles.
      containerLayoutCache: new Map(),
      containerSizeMemory: new Map(),
      expandedContainers: new Set(),
      pendingFocusContainer: null,
    })),

  story: null,
  setStory: (story) => set({ story }),
  learnViewOpen: false,
  openLearnView: () => set({ learnViewOpen: true }),
  closeLearnView: () => set({ learnViewOpen: false }),

  setGraph: (graph) => {
    // "missing" overlay nodes are excluded from search so the toggle-off graph
    // (and search results) stay byte-for-byte identical to before the feature.
    const searchEngine = new SearchEngine(graph.nodes.filter((n) => n.type !== "missing"));
    const query = get().searchQuery;
    const searchResults = query.trim() ? searchEngine.search(query) : [];
    const { viewMode, domainGraph, activeDomainId } = get();
    // Preserve domain view if a domain graph is already loaded
    const keepDomainView = viewMode === "domain" && domainGraph !== null;
    const { nodesById, nodeIdToLayerId, nodeIdToLayerIds } = buildGraphIndexes(graph);
    // A single-layer project (e.g. SQL-only -> one "Data" layer) has no useful
    // layer-overview: the lone brick is noise + an extra click. Skip the
    // overview and open the node graph directly, pinned to the sole layer.
    // >= 2 layers keep the layer-overview as the entry point.
    const layers = graph.layers ?? [];
    const singleLayer = layers.length <= 1;
    // Oversized: the full graph won't be rendered (the focus prompt + the
    // exclusive lineage view take over). selectedNodeId/lineageActive stay
    // null/false so the prompt is what shows; overview/layer browsing is gated.
    const oversized = graph.nodes.length > LARGE_GRAPH_THRESHOLD;
    set({
      graph,
      nodesById,
      nodeIdToLayerId,
      nodeIdToLayerIds,
      searchEngine,
      searchResults,
      navigationLevel: singleLayer ? ("layer-detail" as const) : ("overview" as const),
      activeLayerId: singleLayer ? (layers[0]?.id ?? null) : null,
      selectedNodeId: null,
      focusNodeId: null,
      lineageActive: false,
      oversized,
      nodeHistory: [],
      viewMode: keepDomainView ? "domain" as const : "structural" as const,
      activeDomainId: keepDomainView ? activeDomainId : null,
      containerLayoutCache: new Map(),
      expandedContainers: new Set(),
      pendingFocusContainer: null,
      containerSizeMemory: new Map(),
      stage1Tick: 0,
      layoutIssues: [],
    });
  },

  selectNode: (nodeId) => {
    const { selectedNodeId, nodeHistory } = get();
    // Changing (or clearing) the selection ends any active lineage trace so it
    // never lingers on a node that is no longer selected.
    const lineageReset =
      nodeId !== selectedNodeId
        ? { lineageActive: false, lineageLimit: DEFAULT_LINEAGE_LIMIT }
        : {};
    if (nodeId && selectedNodeId && nodeId !== selectedNodeId) {
      // Push current node to history before navigating away
      set({
        selectedNodeId: nodeId,
        nodeHistory: [...nodeHistory, selectedNodeId].slice(-MAX_HISTORY),
        ...lineageReset,
      });
    } else {
      set({ selectedNodeId: nodeId, ...lineageReset });
    }
  },

  navigateToNode: (nodeId) => {
    get().navigateToNodeInLayer(nodeId);
  },

  navigateToNodeInLayer: (nodeId) => {
    const { graph, selectedNodeId, nodeHistory, nodeIdToLayerId } = get();
    if (!graph) return;
    const layerId = nodeIdToLayerId.get(nodeId) ?? null;
    const newHistory =
      selectedNodeId && nodeId !== selectedNodeId
        ? [...nodeHistory, selectedNodeId].slice(-MAX_HISTORY)
        : nodeHistory;
    if (layerId) {
      set({
        navigationLevel: "layer-detail",
        activeLayerId: layerId,
        selectedNodeId: nodeId,
        focusNodeId: null,
        lineageActive: false, // re-rooting the selection ends any lineage trace
        codeViewerOpen: false,
        codeViewerNodeId: null,
        codeViewerExpanded: false,
        nodeHistory: newHistory,
      });
    } else {
      set({
        selectedNodeId: nodeId,
        lineageActive: false,
        nodeHistory: newHistory,
      });
    }
  },

  navigateToHistoryIndex: (index) => {
    const { nodeHistory, graph, nodeIdToLayerId } = get();
    if (!graph || index < 0 || index >= nodeHistory.length) return;
    const targetId = nodeHistory[index];
    const newHistory = nodeHistory.slice(0, index);
    const layerId = nodeIdToLayerId.get(targetId) ?? null;
    set({
      selectedNodeId: targetId,
      nodeHistory: newHistory,
      lineageActive: false,
      ...(layerId ? { navigationLevel: "layer-detail" as const, activeLayerId: layerId } : {}),
    });
  },

  goBackNode: () => {
    const { nodeHistory, graph, nodeIdToLayerId } = get();
    if (nodeHistory.length === 0 || !graph) return;
    const prevNodeId = nodeHistory[nodeHistory.length - 1];
    const newHistory = nodeHistory.slice(0, -1);
    const layerId = nodeIdToLayerId.get(prevNodeId) ?? null;
    if (layerId) {
      set({
        navigationLevel: "layer-detail",
        activeLayerId: layerId,
        selectedNodeId: prevNodeId,
        lineageActive: false,
        nodeHistory: newHistory,
      });
    } else {
      set({
        selectedNodeId: prevNodeId,
        lineageActive: false,
        nodeHistory: newHistory,
      });
    }
  },

  drillIntoLayer: (layerId) =>
    set({
      navigationLevel: "layer-detail",
      activeLayerId: layerId,
      selectedNodeId: null,
      focusNodeId: null,
      lineageActive: false,
      codeViewerOpen: false,
      codeViewerNodeId: null,
      codeViewerExpanded: false,
      // Container ids derive from folder names and collide across layers
      // (e.g. `container:auth` exists in many layers). Drop the cache so
      // we don't render stale positions for the new layer's children.
      containerLayoutCache: new Map(),
      containerSizeMemory: new Map(),
      expandedContainers: new Set(),
      pendingFocusContainer: null,
    }),

  navigateToOverview: () => {
    // Single-layer projects have no overview to return to — the node graph is
    // the top-level view. Stay put so navigation can't land on an empty/absent
    // overview (defensive: the UI also hides the "back" affordances).
    if ((get().graph?.layers?.length ?? 0) <= 1) return;
    set({
      navigationLevel: "overview",
      activeLayerId: null,
      selectedNodeId: null,
      focusNodeId: null,
      lineageActive: false,
      codeViewerOpen: false,
      codeViewerNodeId: null,
      codeViewerExpanded: false,
      containerLayoutCache: new Map(),
      containerSizeMemory: new Map(),
      expandedContainers: new Set(),
      pendingFocusContainer: null,
    });
  },

  setFocusNode: (nodeId) =>
    set({
      focusNodeId: nodeId,
      selectedNodeId: nodeId,
      lineageActive: false, // focus and lineage trace are mutually exclusive modes
      // Focus mode narrows filteredGraphNodes to focus + 1-hop; the
      // surviving containers have a subset of their original children,
      // and the cache must not return positions for filtered-out ids.
      containerLayoutCache: new Map(),
      containerSizeMemory: new Map(),
      expandedContainers: new Set(),
      pendingFocusContainer: null,
    }),

  // Lineage trace toggles around the current selection. Enabling clears focus
  // mode and resets the node budget; disabling restores the full view.
  toggleLineageTrace: () =>
    set((s) => {
      const next = !s.lineageActive;
      // Oversized: exiting the trace returns to the focus prompt (the full graph
      // cannot be rendered), so drop the selection too.
      if (!next && s.oversized) {
        return { lineageActive: false, selectedNodeId: null, lineageLimit: DEFAULT_LINEAGE_LIMIT };
      }
      // Turning the trace ON over a column root needs COL on, or the exclusive
      // lineage universe hides the whole column chain (mirrors focusLineageOn).
      const colRoot = next && !!s.selectedNodeId
        && s.nodesById.get(s.selectedNodeId)?.type === "column";
      return {
        lineageActive: next,
        lineageLimit: DEFAULT_LINEAGE_LIMIT,
        focusNodeId: s.lineageActive ? s.focusNodeId : null,
        ...(colRoot ? { columnsView: true } : {}),
      };
    }),
  setLineageDirection: (direction) =>
    set({ lineageDirection: direction, lineageLimit: DEFAULT_LINEAGE_LIMIT }),
  showMoreLineage: () =>
    set((s) => ({ lineageLimit: s.lineageLimit + LINEAGE_LIMIT_STEP })),

  focusLineageOn: (nodeId) =>
    set((s) => {
      const node = s.nodesById.get(nodeId);
      return {
        selectedNodeId: nodeId,
        lineageActive: true,
        lineageDirection: "both" as const,
        lineageLimit: DEFAULT_LINEAGE_LIMIT,
        navigationLevel: "layer-detail" as const,
        activeLayerId: s.nodeIdToLayerId.get(nodeId) ?? null,
        // a column root needs COL on, or the column + its column lineage hide
        ...(node?.type === "column" ? { columnsView: true } : {}),
      };
    }),

  setSearchMode: (mode) => set({ searchMode: mode }),
  setSearchQuery: (query) => {
    const engine = get().searchEngine;
    const mode = get().searchMode;
    if (!engine || !query.trim()) {
      set({ searchQuery: query, searchResults: [] });
      return;
    }
    // Currently both modes use the same fuzzy engine
    // When embeddings are available, "semantic" mode will use SemanticSearchEngine
    void mode;
    const searchResults = engine.search(query);
    set({ searchQuery: query, searchResults });
  },

  setPersona: (persona) =>
    set({
      persona,
      // Persona changes filter node types, which shifts container.nodeIds.
      containerLayoutCache: new Map(),
      containerSizeMemory: new Map(),
      expandedContainers: new Set(),
      pendingFocusContainer: null,
    }),

  openCodeViewer: (nodeId) =>
    set({ codeViewerOpen: true, codeViewerNodeId: nodeId, codeViewerExpanded: false }),
  closeCodeViewer: () =>
    set({ codeViewerOpen: false, codeViewerNodeId: null, codeViewerExpanded: false }),
  expandCodeViewer: () => set({ codeViewerExpanded: true }),
  collapseCodeViewer: () => set({ codeViewerExpanded: false }),

  setDiffOverlay: (changed, affected) =>
    set({
      diffMode: true,
      changedNodeIds: new Set(changed),
      affectedNodeIds: new Set(affected),
    }),

  toggleDiffMode: () => set((state) => ({ diffMode: !state.diffMode })),

  clearDiffOverlay: () =>
    set({
      diffMode: false,
      changedNodeIds: new Set<string>(),
      affectedNodeIds: new Set<string>(),
    }),

  toggleFilterPanel: () => set((state) => ({
    filterPanelOpen: !state.filterPanelOpen,
    exportMenuOpen: false,
  })),

  toggleExportMenu: () => set((state) => ({
    exportMenuOpen: !state.exportMenuOpen,
    filterPanelOpen: false,
  })),

  togglePathFinder: () => set((state) => ({
    pathFinderOpen: !state.pathFinderOpen,
  })),

  setReactFlowInstance: (instance) => set({ reactFlowInstance: instance }),

  setFilters: (newFilters) => set((state) => ({
    filters: { ...state.filters, ...newFilters },
  })),

  resetFilters: () => set({
    filters: {
      nodeTypes: new Set<NodeType>(ALL_NODE_TYPES),
      complexities: new Set<Complexity>(ALL_COMPLEXITIES),
      layerIds: new Set<string>(),
      edgeCategories: new Set<EdgeCategory>(ALL_EDGE_CATEGORIES),
    },
  }),

  hasActiveFilters: () => {
    const { filters } = get();
    return filters.nodeTypes.size !== ALL_NODE_TYPES.length
      || filters.complexities.size !== ALL_COMPLEXITIES.length
      || filters.layerIds.size > 0
      || filters.edgeCategories.size !== ALL_EDGE_CATEGORIES.length;
  },

  startTour: () => {
    const { graph, nodeIdToLayerId, activeLayerId } = get();
    if (!graph || !graph.tour || graph.tour.length === 0) return;
    const sorted = getSortedTour(graph);
    const layerNav = navigateTourToLayer(nodeIdToLayerId, sorted[0].nodeIds);
    set({
      tourActive: true,
      currentTourStep: 0,
      tourHighlightedNodeIds: sorted[0].nodeIds,
      selectedNodeId: null,
      ...layerNav,
      ...layerResetIfChanged(layerNav, activeLayerId),
    });
  },

  stopTour: () =>
    set({
      tourActive: false,
      currentTourStep: 0,
      tourHighlightedNodeIds: [],
    }),

  setTourStep: (step) => {
    const { graph, nodeIdToLayerId, activeLayerId } = get();
    if (!graph || !graph.tour || graph.tour.length === 0) return;
    const sorted = getSortedTour(graph);
    if (step < 0 || step >= sorted.length) return;
    const layerNav = navigateTourToLayer(nodeIdToLayerId, sorted[step].nodeIds);
    set({
      currentTourStep: step,
      tourHighlightedNodeIds: sorted[step].nodeIds,
      ...layerNav,
      ...layerResetIfChanged(layerNav, activeLayerId),
    });
  },

  nextTourStep: () => {
    const { graph, currentTourStep, nodeIdToLayerId, activeLayerId } = get();
    if (!graph || !graph.tour || graph.tour.length === 0) return;
    const sorted = getSortedTour(graph);
    if (currentTourStep < sorted.length - 1) {
      const next = currentTourStep + 1;
      const layerNav = navigateTourToLayer(nodeIdToLayerId, sorted[next].nodeIds);
      set({
        currentTourStep: next,
        tourHighlightedNodeIds: sorted[next].nodeIds,
        ...layerNav,
        ...layerResetIfChanged(layerNav, activeLayerId),
      });
    }
  },

  prevTourStep: () => {
    const { graph, currentTourStep, nodeIdToLayerId, activeLayerId } = get();
    if (!graph || !graph.tour || graph.tour.length === 0) return;
    if (currentTourStep > 0) {
      const sorted = getSortedTour(graph);
      const prev = currentTourStep - 1;
      const layerNav = navigateTourToLayer(nodeIdToLayerId, sorted[prev].nodeIds);
      set({
        currentTourStep: prev,
        tourHighlightedNodeIds: sorted[prev].nodeIds,
        ...layerNav,
        ...layerResetIfChanged(layerNav, activeLayerId),
      });
    }
  },

  viewMode: "structural",
  isKnowledgeGraph: false,
  domainGraph: null,
  activeDomainId: null,

  setDomainGraph: (graph) => {
    set({ domainGraph: graph });
  },

  setIsKnowledgeGraph: (value) => {
    set({ isKnowledgeGraph: value });
  },

  setViewMode: (mode) => {
    set({
      viewMode: mode,
      selectedNodeId: null,
      focusNodeId: null,
      codeViewerOpen: false,
      codeViewerNodeId: null,
      codeViewerExpanded: false,
    });
  },

  navigateToDomain: (domainId) => {
    const { selectedNodeId, nodeHistory } = get();
    const newHistory = selectedNodeId
      ? [...nodeHistory, selectedNodeId].slice(-MAX_HISTORY)
      : nodeHistory;
    set({
      viewMode: "domain" as const,
      activeDomainId: domainId,
      focusNodeId: null,
      nodeHistory: newHistory,
    });
  },

  clearActiveDomain: () => {
    set({
      activeDomainId: null,
      selectedNodeId: null,
      focusNodeId: null,
    });
  },

  expandedContainers: new Set<string>(),
  pendingFocusContainer: null,
  setPendingFocusContainer: (containerId) =>
    set({ pendingFocusContainer: containerId }),
  tourFitPending: false,
  setTourFitPending: (pending) => set({ tourFitPending: pending }),
  toggleContainer: (containerId) =>
    set((state) => {
      const next = new Set(state.expandedContainers);
      const willExpand = !next.has(containerId);
      if (willExpand) next.add(containerId);
      else next.delete(containerId);
      return {
        expandedContainers: next,
        pendingFocusContainer: willExpand
          ? containerId
          : state.pendingFocusContainer,
      };
    }),
  expandContainer: (containerId) =>
    set((state) => {
      if (state.expandedContainers.has(containerId)) return {};
      const next = new Set(state.expandedContainers);
      next.add(containerId);
      return { expandedContainers: next };
    }),
  collapseContainer: (containerId) =>
    set((state) => {
      if (!state.expandedContainers.has(containerId)) return {};
      const next = new Set(state.expandedContainers);
      next.delete(containerId);
      return { expandedContainers: next };
    }),
  collapseAllContainers: () => set({ expandedContainers: new Set() }),

  containerLayoutCache: new Map(),
  setContainerLayout: (containerId, childPositions, actualSize) =>
    set((state) => {
      const next = new Map(state.containerLayoutCache);
      next.set(containerId, { childPositions, actualSize });
      const sizeNext = new Map(state.containerSizeMemory);
      sizeNext.set(containerId, actualSize);
      return { containerLayoutCache: next, containerSizeMemory: sizeNext };
    }),
  clearContainerLayouts: () =>
    set({ containerLayoutCache: new Map(), expandedContainers: new Set(), pendingFocusContainer: null }),

  containerSizeMemory: new Map(),

  stage1Tick: 0,
  bumpStage1Tick: () => set((s) => ({ stage1Tick: s.stage1Tick + 1 })),

  layoutIssues: [],
  appendLayoutIssues: (issues) =>
    set((state) => {
      if (issues.length === 0) return {};
      // Dedupe by level+message so a re-running effect doesn't repeatedly
      // pile up identical issues.
      const seen = new Set(
        state.layoutIssues.map((i) => `${i.level}|${i.message}`),
      );
      const fresh = issues.filter((i) => !seen.has(`${i.level}|${i.message}`));
      if (fresh.length === 0) return {};
      return { layoutIssues: [...state.layoutIssues, ...fresh] };
    }),
  clearLayoutIssues: () => set({ layoutIssues: [] }),
}));

