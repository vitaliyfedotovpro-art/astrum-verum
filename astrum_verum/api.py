"""
FastAPI service — REST wrapper around AstrumEngine.

Endpoints (from design doc §6):
    POST  /memory/add              — add a memory
    POST  /memory/search           — topological search (AstrumSearch)
    GET   /memory/state            — current engine state
    GET   /memory/cells            — list cells with node counts
    GET   /memory/cell/{id}/nodes  — nodes in a specific cell
    GET   /lattice/info            — lattice metadata

Run with::

    uvicorn astrum_verum.api:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .engine import AstrumEngine

# ---------------------------------------------------------------------------
# Request / Response models (must be at module level for FastAPI)
# ---------------------------------------------------------------------------


class AddMemoryRequest(BaseModel):
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class AddMemoryResponse(BaseModel):
    node_id: str


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResultItem(BaseModel):
    node_id: str
    text: str
    score: float
    cell_id: int
    breakdown: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    results: list[SearchResultItem]


class CellInfo(BaseModel):
    cell_id: int
    node_count: int


class NodeInfo(BaseModel):
    node_id: str
    text: str
    cell_id: int
    access_count: int
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Global engine instance (module-level singleton)
# ---------------------------------------------------------------------------
_engine: AstrumEngine | None = None


def get_engine() -> AstrumEngine:
    """Get or lazily create the global engine instance."""
    global _engine
    if _engine is None:
        _engine = AstrumEngine()
    return _engine


def create_app(engine: AstrumEngine | None = None) -> FastAPI:
    """
    Create a FastAPI app with an optional pre-configured engine.

    If ``engine`` is provided, it is used as the global singleton.
    Otherwise, a default ``AstrumEngine()`` is created on first request.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _engine
        if engine is not None:
            _engine = engine
        yield

    app = FastAPI(
        title="Astrum Verum",
        description="Geometric Cognitive Memory Architecture on Perfect Lattices",
        version="0.1.0",
        lifespan=lifespan,
    )

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------
    @app.post("/memory/add", response_model=AddMemoryResponse)
    async def add_memory(req: AddMemoryRequest):
        eng = get_engine()
        node_id = eng.add(req.text, req.metadata)
        return AddMemoryResponse(node_id=node_id)

    @app.post("/memory/search", response_model=SearchResponse)
    async def search_memory(req: SearchRequest):
        eng = get_engine()
        results = eng.search(req.query, req.top_k)
        return SearchResponse(
            results=[
                SearchResultItem(
                    node_id=r.node_id,
                    text=r.text,
                    score=r.score,
                    cell_id=r.cell_id,
                    breakdown=r.breakdown,
                    metadata=r.metadata,
                )
                for r in results
            ]
        )

    @app.get("/memory/state")
    async def memory_state():
        return get_engine().state()

    @app.get("/memory/cells", response_model=list[CellInfo])
    async def list_cells():
        eng = get_engine()
        stats = eng.store.stats()
        return [
            CellInfo(cell_id=int(cid), node_count=count)
            for cid, count in stats["cell_counts"].items()
        ]

    @app.get("/memory/cell/{cell_id}/nodes", response_model=list[NodeInfo])
    async def cell_nodes(cell_id: int):
        eng = get_engine()
        nodes = eng.store.get_nodes_in_cell(cell_id)
        return [
            NodeInfo(
                node_id=n.id,
                text=n.text,
                cell_id=cell_id,
                access_count=n.access_count,
                metadata=n.metadata,
            )
            for n in nodes
        ]

    @app.get("/lattice/info")
    async def lattice_info():
        eng = get_engine()
        info = eng.lattice.info()
        return {
            "name": info.name,
            "dimension": info.dimension,
            "num_vertices": info.num_vertices,
            "num_edges": info.num_edges,
            "neighbors_per_vertex": info.neighbors_per_vertex,
            "symmetry_group_order": info.symmetry_group_order,
        }

    return app


# Default app instance for `uvicorn astrum_verum.api:app`.
app = create_app()
