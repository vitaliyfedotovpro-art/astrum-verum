# Astrum Verum — Mathematical & Geometric Foundations

> **Note:** This document describes the **Layer 1 (lattice) geometry** — the math
> is correct, but its retrieval-quality thesis is **unproven**. See
> [`docs/astrum_verum_design.md` §1.2](../docs/astrum_verum_design.md) for an
> honest assessment. The validated part of the project is the **VSA/SDM layer**
> (Layer 2).

Astrum Verum is an open-source, zero-dependency, academically clean vector memory framework that organizes high-dimensional semantic vectors using dense coordinate lattices in 4D and 8D.

Below is the theoretical and operational breakdown of the core modules.

---

## 1. High-Dimensional Lattices

Lattices serve as multi-dimensional discretizers (conceptual quantizers) of high-dimensional semantic spaces.

### D₄ Lattice (The 24-Cell / Icositetrachoron)
* **Dimension ($d$)**: 4
* **Vertices ($V$)**: 24
* **Edges ($E$)**: 96
* **Neighbors per Vertex**: 8
* **Geometry**: 
  - 8 Axial Vertices: Permutations of $(\pm 1, 0, 0, 0)$.
  - 16 Diagonal Vertices: All sign combinations of $(\pm 0.5, \pm 0.5, \pm 0.5, \pm 0.5)$.
  - Every vertex lies exactly on the unit 3-sphere ($S^3$).
  - Two vertices $u, v$ are adjacent iff their Euclidean distance $\lVert u - v \rVert = 1.0$.

### E₈ Lattice (The Gosset Polytope $4_{21}$)
* **Dimension ($d$)**: 8
* **Vertices ($V$)**: 240
* **Edges ($E$)**: 6720
* **Neighbors per Vertex**: 56
* **Geometry**:
  - 112 Type-1 Vertices: Permutations of $(\pm 1, \pm 1, 0, 0, 0, 0, 0, 0)$.
  - 128 Type-2 Vertices: All sign combinations of $(\pm 0.5, \pm 0.5, \pm 0.5, \pm 0.5, \pm 0.5, \pm 0.5, \pm 0.5, \pm 0.5)$ with an *even* number of minus signs.
  - Every vertex lies exactly on the unit 7-sphere ($S^7$) when scaled by $1/\sqrt{2}$.
  - Two vertices $u, v$ are adjacent iff their Euclidean distance $\lVert u - v \rVert = 1.0$.

---

## 2. Concept-Anchored Projection (CAP)

The mapping from the raw high-dimensional semantic space $\mathbb{R}^N$ (e.g. 384-dimensional text embeddings) to the low-dimensional lattice space $\mathbb{R}^d$ ($d=4$ or $d=8$) is done via Softmax-weighted Soft Membership.

1. **Concept Anchors**: A set of $V$ anchor embeddings $A_i \in \mathbb{R}^N$ ($i = 0, \dots, V-1$) corresponding directly to the lattice vertices.
2. **Softmax Projection**: For an input embedding $x \in \mathbb{R}^N$:
   $$w_i = \text{softmax}_i\left( \frac{\cos(x, A_i)}{\tau} \right) = \frac{\exp\left(\frac{x \cdot A_i}{\tau \lVert x \rVert \lVert A_i \rVert}\right)}{\sum_{k=0}^{V-1} \exp\left(\frac{x \cdot A_k}{\tau \lVert x \rVert \lVert A_k \rVert}\right)}$$
   where $\tau$ is the temperature hyperparameter (default: 0.1).
3. **Lattice Representation**: The projected coordinate $z \in \mathbb{R}^d$ is the weighted combination of the normalized lattice vertices $v_i \in \mathbb{R}^d$:
   $$z = \sum_{i=0}^{V-1} w_i v_i$$
   The soft membership dictionary $\{i: w_i\}$ determines which Voronoi cell(s) contain this node.

---

## 3. Focus Alignment & Inverse Rotations in $SO(d)$

When searching, the user's attention focus shifts. Instead of rotating the massive corpus of memory nodes, we rotate the *query vector in the opposite direction* (an optimized $O(d^2)$ operation vs $O(N \cdot d^2)$).

### Focus Vector Computation
Calculated as the exponentially decaying sum of recent query vectors in the lattice space:
$$F = \text{normalize}\left(\sum_{t=0}^{k-1} \lambda^{k - 1 - t} z_t\right)$$
where $\lambda$ is the focus decay parameter (default: 0.8).

### Householder/Givens Focus Alignment
We construct a rotation matrix $R \in SO(d)$ that maps the focus vector $F$ to the target canonical axis $e_0 = (1, 0, \dots, 0)$:
$$R F = e_0$$

Using a modified 2D rotation in the plane spanned by $F$ and $e_0$:
1. Gram-Schmidt orthodecomposition to get a unit vector $u$ orthogonal to $e_0$ in the rotation plane:
   $$f_{perp} = F - \langle F, e_0 \rangle e_0$$
   $$u = \frac{f_{perp}}{\lVert f_{perp} \rVert}$$
2. Given $\theta$ as the angle between $F$ and $e_0$:
   $$\cos\theta = \langle F, e_0 \rangle$$
   $$\sin\theta = \sqrt{1 - \cos^2\theta}$$
3. The rotation matrix $R$ is built as:
   $$R = I + (\cos\theta - 1)(uu^{\mathsf{T}} + e_0e_0^{\mathsf{T}}) + \sin\theta(e_0u^{\mathsf{T}} - ue_0^{\mathsf{T}})$$
   - Determinant $\det(R) = +1$ (proper rotation).
   - Orthogonal: $R^{\mathsf{T}}R = I$ (perfect Euclidean isometry).

### Inverse Rotation Query
The search query vector $q$ is rotated inversely:
$$q' = R^{\mathsf{T}} q$$
We then run the Closest Vector Problem (CVP) on $q'$ to find the start cell of the spreading activation.

---

## 4. Spreading Activation & Hebbian Learning

Search queries do not only fetch the nearest mathematical matches; they activate an adaptive cognitive graph.

### BFS Energy Wave
Starting from the CVP-decoded start cell $c_{start}$, energy propagates outward through the adjacency graph:
1. $E(c_{start}) = 1.0$.
2. For each step $s = 1, \dots, \text{radius}$:
   $$E(n) = \max \left( E(n), \, E(cell) \cdot W(cell, n) \cdot \delta^s \right)$$
   where $W(cell, n)$ is the edge weight (default: 1.0) and $\delta$ is the wave decay (default: 0.6).
3. Cells below `min_energy` (default: 0.01) are pruned.

### Hebbian Edge Updates
To allow the topology to adapt and literally "learn" user patterns:
* **Co-activated Edges** (both endpoints $u, v$ in the active set):
  $$W(u, v) \leftarrow W(u, v) + \eta \cdot E(u) \cdot E(v)$$
  where $\eta$ is the learning rate (default: 0.05).
* **Non-co-activated Edges** (either $u$ or $v$ inactive):
  $$W(u, v) \leftarrow W(u, v) \times (1 - \gamma)$$
  where $\gamma$ is the forget rate (default: 0.001).

---

## 5. Hybrid Scoring Equation

Candidates gathered from all activated cells are ranked using a multi-signal hybrid score:

$$\text{Score} = \alpha \cdot \text{CosineSimilarity} + \beta \cdot \text{TopoBoost} + \gamma \cdot \text{Recency}$$

* **Cosine Similarity**: $\cos(q_{raw}, x_{raw})$ in $\mathbb{R}^N$.
* **TopoBoost**: The activation energy $E(c)$ of the Voronoi cell that contains the node.
* **Recency**: Exponential decay based on access time:
  $$\text{Recency} = \exp\left(-\text{decay} \cdot (t_{now} - t_{last})\right)$$

By default, weights sum to 1 ($\alpha=0.4, \beta=0.3, \gamma=0.3$). This lets highly relevant topological neighbors (connected by learned Hebbian paths) rise above strict semantic matches!
