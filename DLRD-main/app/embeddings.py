"""Local embedding index for semantic search in the RAG pipeline.

Uses fastembed (onnxruntime-based, no PyTorch) to compute dense embeddings
locally from node text. Stores the index in:
  data/embeddings.npz          float32 matrix (N × D), numpy compressed
  data/embeddings_meta.json    model name, graph hash, node id list

Freshness is checked via a hash of the node (id, name, summary) triples —
if the knowledge graph changes, the index is transparently invalidated and
the caller falls back to lexical search until a new analysis (or /api/reindex)
rebuilds it.

Every public method fails gracefully (returns None/False) when fastembed is
not installed or the model is not cached, so the rest of the app never crashes.

SECURITY: zero network egress. fastembed reads from a local directory
(~/.cache/fastembed by default, or FASTEMBED_CACHE_PATH if set).

──────────────────────────────────────────────────
Pre-seeding the model cache (machine with internet):

    python -c "
    from fastembed import TextEmbedding
    list(TextEmbedding('BAAI/bge-small-en-v1.5').embed(['warm-up']))
    print('cached at ~/.cache/fastembed/')
    "

Then copy ~/.cache/fastembed/ to the work machine and set:
    export FASTEMBED_CACHE_PATH=/path/to/that/copy
──────────────────────────────────────────────────
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from .config import DATA_DIR

if TYPE_CHECKING:  # avoid hard numpy import at module load time
    import numpy as np

log = logging.getLogger("data_lineage_retro_documentation.embeddings")

MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBED_NPZ_PATH  = DATA_DIR / "embeddings.npz"
EMBED_META_PATH = DATA_DIR / "embeddings_meta.json"

# Honour FASTEMBED_CACHE_PATH for pre-seeded offline caches.
_CACHE_DIR: str | None = os.environ.get("FASTEMBED_CACHE_PATH") or None

# Module-level ONNX session cache: one TextEmbedding instance per model name.
# fastembed's ONNX session is thread-safe for inference, so sharing it across
# the pipeline thread and FastAPI request threads is safe under the GIL.
_model_cache: dict[str, object] = {}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_model(model_name: str) -> object:
    """Return a cached TextEmbedding instance (loads ONNX session once)."""
    if model_name not in _model_cache:
        from fastembed import TextEmbedding  # type: ignore[import-untyped]
        kwargs: dict = {"model_name": model_name}
        if _CACHE_DIR:
            kwargs["cache_dir"] = _CACHE_DIR
        _model_cache[model_name] = TextEmbedding(**kwargs)
    return _model_cache[model_name]


def _node_text(node: dict) -> str:
    """Compact text representation of a node for embedding."""
    ntype   = node.get("type", "")
    name    = node.get("name", "")
    summary = node.get("summary", "")
    fp      = node.get("filePath", "") or ""
    tags    = node.get("tags", [])
    parts   = [f"{ntype} {name}".strip()]
    if summary:
        parts.append(summary)
    if fp:
        parts.append(fp)
    if tags:
        parts.append("tags: " + " ".join(str(t) for t in tags[:6]))
    return " | ".join(p for p in parts if p)


def _nodes_hash(nodes: list[dict]) -> str:
    """Stable 20-char hash of the embedding-relevant content of the node list.

    Hashing (id, name, summary) triples sorted by id means the hash changes
    whenever a node is added, removed, or its summary is updated — exactly
    the cases where embeddings would be stale.
    """
    payload = sorted(
        (n.get("id", ""), n.get("name", ""), n.get("summary", ""))
        for n in nodes
    )
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


# ---------------------------------------------------------------------------
# EmbeddingIndex
# ---------------------------------------------------------------------------

class EmbeddingIndex:
    """In-memory embedding index backed by data/embeddings.npz."""

    def __init__(
        self,
        node_ids: list[str],
        vectors: "np.ndarray",   # shape (N, D), float32, L2-normalised
        model_name: str = MODEL_NAME,
    ) -> None:
        self._node_ids   = node_ids
        self._vectors    = vectors
        self._model_name = model_name

    # ------------------------------------------------------------------ build

    @classmethod
    def build(
        cls,
        nodes: list[dict],
        model_name: str = MODEL_NAME,
    ) -> "EmbeddingIndex | None":
        """Compute embeddings for every node and persist the index.

        Returns the built index on success, None on any failure (missing dep,
        model not cached, I/O error).  The caller must treat None as
        "embeddings unavailable" and fall back to lexical search.
        """
        try:
            import numpy as np
            _get_model(model_name)          # probe import + cache availability
        except ImportError:
            log.info("fastembed/numpy not installed — embedding index skipped.")
            return None
        except Exception as exc:
            log.warning("Embedding model unavailable: %s — index skipped.", exc)
            return None

        if not nodes:
            return None

        texts    = [_node_text(n) for n in nodes]
        node_ids = [n["id"] for n in nodes]

        try:
            model = _get_model(model_name)
            raw   = list(model.embed(texts))          # type: ignore[attr-defined]
            vectors: np.ndarray = np.array(raw, dtype=np.float32)
        except Exception as exc:
            log.warning("Embedding computation failed: %s — index skipped.", exc)
            return None

        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(str(EMBED_NPZ_PATH), vectors=vectors)
            meta = {
                "model":      model_name,
                "graph_hash": _nodes_hash(nodes),
                "node_ids":   node_ids,
                "node_count": len(node_ids),
                "dim":        int(vectors.shape[1]) if vectors.ndim == 2 else 0,
                "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            }
            EMBED_META_PATH.write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            log.warning("Failed to persist embedding index: %s", exc)
            return None

        log.info(
            "Embedding index built: %d nodes, dim=%d, model=%s",
            len(node_ids), vectors.shape[1], model_name,
        )
        return cls(node_ids=node_ids, vectors=vectors, model_name=model_name)

    # ------------------------------------------------------------------- load

    @classmethod
    def load_if_valid(cls, nodes: list[dict]) -> "EmbeddingIndex | None":
        """Load the persisted index if it exists and is fresh for these nodes.

        Returns None (triggering lexical fallback) when:
        - numpy / fastembed are not importable
        - index files are missing
        - the graph content changed since the index was built
        - any I/O or format error
        """
        try:
            import numpy as np
        except ImportError:
            return None

        if not EMBED_META_PATH.exists() or not EMBED_NPZ_PATH.exists():
            return None

        try:
            meta = json.loads(EMBED_META_PATH.read_text(encoding="utf-8"))
        except Exception:
            return None

        if meta.get("graph_hash") != _nodes_hash(nodes):
            log.info("Embedding index is stale (graph changed) — lexical fallback.")
            return None

        stored_ids: list[str] = meta.get("node_ids", [])
        model_name: str = meta.get("model", MODEL_NAME)
        if not stored_ids:
            return None

        try:
            data    = np.load(str(EMBED_NPZ_PATH))
            vectors = data["vectors"]                 # (N, D)
        except Exception as exc:
            log.warning("Failed to load embeddings.npz: %s", exc)
            return None

        if vectors.shape[0] != len(stored_ids):
            log.warning("Embedding matrix / meta size mismatch — lexical fallback.")
            return None

        return cls(node_ids=stored_ids, vectors=vectors, model_name=model_name)

    # ------------------------------------------------------------------ search

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Return the top_k (node_id, cosine_score) pairs for the query.

        fastembed returns L2-normalised vectors → dot product == cosine similarity.
        Uses numpy argpartition (O(N) average) rather than full sort.
        """
        try:
            import numpy as np
            model  = _get_model(self._model_name)
            q_vecs = list(model.embed([query]))       # type: ignore[attr-defined]
            q_vec  = np.array(q_vecs[0], dtype=np.float32)
        except Exception as exc:
            log.warning("Query embedding failed: %s", exc)
            return []

        try:
            scores  = self._vectors @ q_vec           # (N,)
            k       = min(top_k, len(self._node_ids))
            if k == 0:
                return []
            # argpartition selects top-k without full sort — O(N) average
            idx     = np.argpartition(scores, -k)[-k:]
            idx     = idx[np.argsort(-scores[idx])]  # sort the k winners desc
            return [(self._node_ids[int(i)], float(scores[i])) for i in idx]
        except Exception as exc:
            log.warning("Similarity search failed: %s", exc)
            return []

    # -------------------------------------------------------- availability probe

    @staticmethod
    def is_available() -> bool:
        """True if fastembed and numpy are importable (model cache not checked)."""
        try:
            import fastembed   # noqa: F401  type: ignore[import-untyped]
            import numpy       # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def index_info() -> dict:
        """Metadata about the current index (for /api/reindex status response)."""
        if not EMBED_META_PATH.exists():
            return {"indexed": False}
        try:
            meta = json.loads(EMBED_META_PATH.read_text(encoding="utf-8"))
            return {
                "indexed":    True,
                "model":      meta.get("model"),
                "node_count": meta.get("node_count", 0),
                "dim":        meta.get("dim", 0),
                "created_at": meta.get("created_at"),
            }
        except Exception:
            return {"indexed": False}
