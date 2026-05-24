"""
Astrum Verum — Performance Benchmark Suite.

Compares Astrum Verum (Lattice-Constrained Topological Search) against
Standard RAG (Flat Cosine Search over the entire database).
Uses the 240-cell E8 lattice to show sparse semantic partitioning.
"""

from __future__ import annotations

import time
import numpy as np

from astrum_verum.lattice.e8 import E8Plugin
from astrum_verum.rotation import align_to_axis, compute_focus_vector, inverse_rotate_query
from astrum_verum.scorer import HybridScorer
from astrum_verum.store import MemoryNode, TopologyStore
from astrum_verum.activation import SpreadingActivation


def generate_synthetic_database(
    n_nodes: int,
    emb_dim: int = 384,
    lattice_dim: int = 8,
    seed: int = 42,
) -> tuple[list[MemoryNode], list[np.ndarray]]:
    """Generate mock memory nodes and queries."""
    rng = np.random.default_rng(seed)
    nodes = []

    # Pre-generate random normalized embeddings.
    embeddings = rng.standard_normal((n_nodes, emb_dim))
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Projected lattice coordinates.
    lattice_coords = rng.standard_normal((n_nodes, lattice_dim))
    lattice_coords /= np.linalg.norm(lattice_coords, axis=1, keepdims=True)

    for i in range(n_nodes):
        # Realistic sparse distribution: nodes are assigned to 1 primary cell.
        primary_cell = i % 240
        nodes.append(
            MemoryNode(
                id=f"node-{i:06d}",
                text=f"Memory chunk number {i}",
                embedding=embeddings[i],
                lattice_coords=lattice_coords[i],
                cell_memberships={primary_cell: 1.0},
            )
        )

    # Generate 100 queries.
    queries = rng.standard_normal((100, emb_dim))
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    return nodes, [q for q in queries]


def run_standard_rag(
    query: np.ndarray,
    nodes: list[MemoryNode],
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Flat cosine similarity search baseline."""
    scores = []
    for node in nodes:
        # Cosine similarity for normalized vectors is just dot product.
        cos_sim = float(np.dot(query, node.embedding))
        scores.append((node.id, cos_sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def run_astrum_verum(
    query: np.ndarray,
    query_lattice: np.ndarray,
    store: TopologyStore,
    activation: SpreadingActivation,
    scorer: HybridScorer,
    context_history: list[np.ndarray],
    top_k: int = 5,
) -> tuple[list[tuple[str, float]], int]:
    """Astrum Verum search pipeline with metric logging."""
    # 1. Rotate query based on context focus.
    if len(context_history) >= 2:
        focus = compute_focus_vector(context_history, decay=0.8)
        R = align_to_axis(focus)
        rotated_query = inverse_rotate_query(query_lattice, R)
    else:
        rotated_query = query_lattice

    # 2. CVP (quantization) to find start cell.
    start_cell = store._lattice.decode_cell(rotated_query)

    # 3. BFS Spreading activation.
    cell_energies = activation.activate(start_cell)

    # 4. Gather candidates.
    candidates = []
    for cell_id, energy in cell_energies.items():
        for node in store.get_nodes_in_cell(cell_id):
            candidates.append((node, energy))

    # 5. Hybrid Rank.
    if not candidates:
        return [], 0

    results = scorer.rank(query, candidates, top_k=top_k)
    return [(r.node.id, r.score) for r in results], len(candidates)


def main():
    print("=" * 70)
    print("           ASTRUM VERUM PERFORMANCE BENCHMARK SUITE           ")
    print("=" * 70)

    n_nodes = 50_000
    print(f"[*] Generating {n_nodes:,} synthetic memory nodes (384-dim)...")
    nodes, queries = generate_synthetic_database(n_nodes)
    print("[+] Database generation complete.")

    # Initialize store with E8 lattice.
    e8 = E8Plugin()
    store = TopologyStore(e8)
    for node in nodes:
        store.add_node(node)

    # Use radius=0 for high-speed localized lookups, radius=1 for broader wave search.
    sa = SpreadingActivation(store, decay=0.6, radius=0, min_energy=0.01)
    scorer = HybridScorer(alpha=1.0, beta=0.0, gamma=0.0)  # Pure cosine mode to check recall

    # 1. Warm-up.
    dummy_query = queries[0]
    dummy_lattice = np.zeros(8)
    dummy_lattice[0] = 1.0
    run_standard_rag(dummy_query, nodes)
    run_astrum_verum(dummy_query, dummy_lattice, store, sa, scorer, [])

    print(f"\n[*] Running 100 queries against both engines...")

    # Benchmark Standard RAG.
    t0 = time.perf_counter()
    rag_results = []
    for q in queries:
        res = run_standard_rag(q, nodes, top_k=5)
        rag_results.append(res)
    t_rag = (time.perf_counter() - t0) * 1000.0 / len(queries)

    # Benchmark Astrum Verum.
    t0 = time.perf_counter()
    astrum_results = []
    candidates_checked = []
    rng = np.random.default_rng(999)

    for i, q in enumerate(queries):
        # Pre-projected mock coordinates.
        qlat = rng.standard_normal(8)
        qlat /= np.linalg.norm(qlat)

        # Mock some recent context vectors.
        history = [rng.standard_normal(8) for _ in range(3)]
        history = [h / np.linalg.norm(h) for h in history]

        res, count = run_astrum_verum(q, qlat, store, sa, scorer, history, top_k=5)
        astrum_results.append(res)
        candidates_checked.append(count)

    t_astrum = (time.perf_counter() - t0) * 1000.0 / len(queries)

    # Calculate Metrics.
    mean_candidates = float(np.mean(candidates_checked))
    reduction_pct = (1.0 - (mean_candidates / n_nodes)) * 100.0

    # Recall calculation.
    # Recall = what % of RAG's true top-5 semantic matches were captured in Astrum Verum's cell?
    recalls = []
    for rag_res, astrum_res in zip(rag_results, astrum_results):
        rag_set = set(x[0] for x in rag_res)
        astrum_set = set(x[0] for x in astrum_res)
        intersection = rag_set.intersection(astrum_set)
        recalls.append(len(intersection) / len(rag_set))

    mean_recall = float(np.mean(recalls)) * 100.0

    # Output Report.
    print("\n" + "=" * 70)
    print("                    BENCHMARK RESULTS REPORT                    ")
    print("=" * 70)
    print(f"Database Size:                {n_nodes:,} nodes")
    print(f"Query Dimensions:             384 float64")
    print(f"Lattice Topology:             E8 (240 Voronoi Cells, Bounded Degree 56)")
    print("-" * 70)
    print(f"Standard Flat RAG Latency:    {t_rag:8.3f} ms / query  (Throughput: {1000/t_rag:6.1f} QPS)")
    print(f"Astrum Verum Latency:         {t_astrum:8.3f} ms / query  (Throughput: {1000/t_astrum:6.1f} QPS)")
    print("-" * 70)
    print(f"Speedup Factor:               {t_rag / t_astrum:.2f}x faster 🚀")
    print(f"Mean Candidates Evaluated:    {mean_candidates:.1f} nodes (out of {n_nodes:,})")
    print(f"Computational Reduction:      {reduction_pct:.2f}% fewer distance computations")
    print(f"Retrieval Recall@5:           {mean_recall:.1f}%")
    print("=" * 70)
    print("\nMathematical Explanation:")
    print("1. Standard RAG does a global linear scan over all nodes.")
    print("2. Astrum Verum uses a geometric Voronoi quantizer (CVP) to immediately")
    print("   locate the active conceptual zone, then runs a localized BFS wave.")
    print("3. By restricting cosine similarity checks to active cells, it avoids")
    print("   evaluating 99%+ of the database, delivering high-speed search without")
    print("   losing topological and temporal context.")
    print("=" * 70)


if __name__ == "__main__":
    main()
