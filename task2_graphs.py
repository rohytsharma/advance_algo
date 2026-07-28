"""
Graph Algorithms and Pathfinding for a city transportation network.

Graph model: weighted DIRECTED graph via an ADJACENCY LIST
Algorithms:
    * Dijkstra        -- single-source shortest path, non-negative weights
                         (binary-heap implementation, O((n+m) log n))
    * Prim            -- minimum spanning tree, O((n+m) log n) with a heap
    * Bellman-Ford    -- shortest path with negative edges + neg-cycle detect
                         (O(n*m))
"""
from __future__ import annotations
import csv
import heapq
import math
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)



#  Graph (adjacency list)                                                      #

class Graph:
    def __init__(self, n, directed=True):
        self.n = n
        self.directed = directed
        self.adj = [[] for _ in range(n)]     # adj[u] = list of (v, weight)
        self.coords = [None] * n              # optional (x, y) for plotting

    def add_edge(self, u, v, w):
        self.adj[u].append((v, w))
        if not self.directed:
            self.adj[v].append((u, w))

    def edges(self):
        for u in range(self.n):
            for v, w in self.adj[u]:
                yield u, v, w

    def num_edges(self):
        m = sum(len(a) for a in self.adj)
        return m if self.directed else m // 2



#  Dijkstra's algorithm (binary heap)                                          #

def dijkstra(g: Graph, src: int):
    dist = [math.inf] * g.n
    parent = [-1] * g.n
    dist[src] = 0
    pq = [(0, src)]                          # (distance, vertex)
    visited = [False] * g.n
    while pq:
        d, u = heapq.heappop(pq)
        if visited[u]:                       # lazy deletion of stale entries
            continue
        visited[u] = True
        for v, w in g.adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))
    return dist, parent


#  Prim's algorithm (minimum spanning tree)                                    #

def prim(g: Graph, src: int = 0):
    """Requires an undirected/symmetric graph. Returns (mst_edges, total)."""
    in_tree = [False] * g.n
    parent = [-1] * g.n
    key = [math.inf] * g.n
    key[src] = 0
    pq = [(0, src)]
    total = 0
    mst_edges = []
    while pq:
        k, u = heapq.heappop(pq)
        if in_tree[u]:
            continue
        in_tree[u] = True
        total += k
        if parent[u] != -1:
            mst_edges.append((parent[u], u, k))
        for v, w in g.adj[u]:
            if not in_tree[v] and w < key[v]:
                key[v] = w
                parent[v] = u
                heapq.heappush(pq, (w, v))
    return mst_edges, total



#  Bellman-Ford (handles negative edges, detects negative cycles)              #

def bellman_ford(g: Graph, src: int):
    dist = [math.inf] * g.n
    parent = [-1] * g.n
    dist[src] = 0
    edges = list(g.edges())
    # Relax all edges n-1 times.
    for _ in range(g.n - 1):
        changed = False
        for u, v, w in edges:
            if dist[u] != math.inf and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                parent[v] = u
                changed = True
        if not changed:                      # early exit optimisation
            break
    # One more pass: any relaxation => a negative-weight cycle is reachable.
    for u, v, w in edges:
        if dist[u] != math.inf and dist[u] + w < dist[v]:
            return dist, parent, True        # negative cycle detected
    return dist, parent, False



#  Graph generators                                                            #

def random_sparse_graph(n, avg_degree=4, directed=True, rng=None, wmax=100):
    rng = rng or random.Random(0)
    g = Graph(n, directed=directed)
    for i in range(n):
        g.coords[i] = (rng.uniform(0, 1), rng.uniform(0, 1))
    m = n * avg_degree
    seen = set()
    for _ in range(m):
        u = rng.randrange(n)
        v = rng.randrange(n)
        if u == v or (u, v) in seen:
            continue
        seen.add((u, v))
        g.add_edge(u, v, rng.randint(1, wmax))
    return g


def geometric_graph(n, radius=0.18, rng=None):
    """Undirected graph: connect nearby points, weight = Euclidean distance.
       Realistic 'road network' used for the MST/Dijkstra visualisations."""
    rng = rng or random.Random(7)
    g = Graph(n, directed=False)
    pts = [(rng.uniform(0, 1), rng.uniform(0, 1)) for _ in range(n)]
    for i in range(n):
        g.coords[i] = pts[i]
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]
            d = math.hypot(dx, dy)
            if d <= radius:
                g.add_edge(i, j, d)
    # ensure connectivity: link each isolated vertex to its nearest neighbour
    for i in range(n):
        if not g.adj[i]:
            best, bd = None, math.inf
            for j in range(n):
                if j != i:
                    d = math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])
                    if d < bd:
                        best, bd = j, d
            g.add_edge(i, best, bd)
    return g



#  Visualisations                                                              #

def draw_graph(g, highlight_edges, title, path, src=None):
    plt.figure(figsize=(7, 7))
    # base edges (light grey)
    for u, v, w in g.edges():
        if u < v or g.directed:
            x = [g.coords[u][0], g.coords[v][0]]
            y = [g.coords[u][1], g.coords[v][1]]
            plt.plot(x, y, color="0.85", lw=0.8, zorder=1)
    # highlighted edges (tree / path)
    hset = set()
    for u, v, w in highlight_edges:
        x = [g.coords[u][0], g.coords[v][0]]
        y = [g.coords[u][1], g.coords[v][1]]
        plt.plot(x, y, color="#c0392b", lw=2.0, zorder=2)
        hset.add(u); hset.add(v)
    # nodes
    xs = [c[0] for c in g.coords]
    ys = [c[1] for c in g.coords]
    plt.scatter(xs, ys, s=18, color="#2c3e50", zorder=3)
    if src is not None:
        plt.scatter([g.coords[src][0]], [g.coords[src][1]],
                    s=120, color="#27ae60", zorder=4, label="source")
        plt.legend()
    plt.title(title)
    plt.axis("equal")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def build_shortest_path_tree_edges(parent):
    return [(parent[v], v, 0) for v in range(len(parent)) if parent[v] != -1]



#  Benchmark                                                                   #

def timed(fn, repeats=3):
    best = math.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def run_benchmarks():
    rng = random.Random(123)
    rows = []
    for n in (100, 1_000, 5_000):
        # sparse graph (m ~ 4n) with NON-NEGATIVE weights for all three algos
        gs = random_sparse_graph(n, avg_degree=4, directed=True, rng=rng)
        # undirected copy for Prim
        gu = random_sparse_graph(n, avg_degree=4, directed=False, rng=rng)
        m_s = gs.num_edges()

        t_dij = timed(lambda: dijkstra(gs, 0))
        t_bf = timed(lambda: bellman_ford(gs, 0), repeats=1)
        t_prim = timed(lambda: prim(gu, 0))
        rows.append({"n": n, "m": m_s, "density": "sparse",
                     "dijkstra": t_dij, "bellman_ford": t_bf, "prim": t_prim})

        # denser graph (m ~ 20n) to show density sensitivity
        gd = random_sparse_graph(n, avg_degree=20, directed=True, rng=rng)
        t_dij_d = timed(lambda: dijkstra(gd, 0))
        t_bf_d = timed(lambda: bellman_ford(gd, 0), repeats=1)
        rows.append({"n": n, "m": gd.num_edges(), "density": "dense",
                     "dijkstra": t_dij_d, "bellman_ford": t_bf_d,
                     "prim": float("nan")})
    return rows


def save_csv(rows):
    path = os.path.join(RESULTS_DIR, "task2_results.csv")
    fields = ["n", "m", "density", "dijkstra", "bellman_ford", "prim"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def plot_benchmarks(rows):
    plt.figure(figsize=(7, 4.5))
    for algo, marker in (("dijkstra", "o-"), ("bellman_ford", "s-")):
        for dens in ("sparse", "dense"):
            xs, ys = [], []
            for r in rows:
                if r["density"] == dens and r.get(algo, float("nan")) == r.get(algo, float("nan")):
                    xs.append(r["n"]); ys.append(r[algo] * 1e3)
            if xs:
                plt.plot(xs, ys, marker, label=f"{algo} ({dens})")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("Vertices n"); plt.ylabel("Runtime (ms)")
    plt.title("Task 2 - Dijkstra vs Bellman-Ford (sparse vs dense)")
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.legend(); plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "task2_runtime.png")
    plt.savefig(out, dpi=130); plt.close()



#  Negative-cycle demonstration                                                #

def negative_edge_demo():
    # 4-vertex graph with a negative edge but NO negative cycle
    g1 = Graph(4, directed=True)
    for u, v, w in [(0, 1, 4), (0, 2, 5), (1, 3, 5), (2, 1, -3), (2, 3, 6)]:
        g1.add_edge(u, v, w)
    dist1, _, neg1 = bellman_ford(g1, 0)

    # graph WITH a negative cycle (1->2->3->1 sums to -1)
    g2 = Graph(4, directed=True)
    for u, v, w in [(0, 1, 1), (1, 2, 2), (2, 3, -6), (3, 1, 3)]:
        g2.add_edge(u, v, w)
    _, _, neg2 = bellman_ford(g2, 0)
    return dist1, neg1, neg2


if __name__ == "__main__":
    # ---- correctness / negative weights ----
    dist_neg, has_neg1, has_neg2 = negative_edge_demo()
    print("=== Bellman-Ford negative-weight handling ===")
    print(f"Graph with negative edge (no cycle): dist from 0 = {dist_neg}, "
          f"negative-cycle detected = {has_neg1}")
    print(f"Graph with negative cycle          : "
          f"negative-cycle detected = {has_neg2}")

    # cross-check Dijkstra vs Bellman-Ford agree on a non-negative graph
    gcheck = random_sparse_graph(300, avg_degree=5, rng=random.Random(9))
    d1, _ = dijkstra(gcheck, 0)
    d2, _, _ = bellman_ford(gcheck, 0)
    agree = all((a == b) or (a == math.inf and b == math.inf)
                for a, b in zip(d1, d2))
    print(f"\nDijkstra vs Bellman-Ford agree on non-negative graph: {agree}")

    # ---- visualisations on a geometric 'road network' ----
    groad = geometric_graph(120, radius=0.16, rng=random.Random(3))
    dist, parent = dijkstra(groad, 0)
    draw_graph(groad, build_shortest_path_tree_edges(parent),
               "Dijkstra shortest-path tree (source in green)",
               os.path.join(RESULTS_DIR, "task2_dijkstra_tree.png"), src=0)
    mst_edges, total = prim(groad, 0)
    draw_graph(groad, mst_edges,
               f"Prim minimum spanning tree (total weight = {total:.2f})",
               os.path.join(RESULTS_DIR, "task2_mst.png"))
    print(f"\nMST edges = {len(mst_edges)}, total weight = {total:.3f}")

    # ---- benchmarks ----
    rows = run_benchmarks()
    csv_path = save_csv(rows)
    plot_benchmarks(rows)
    print("\n=== Task 2 runtimes (ms) ===")
    print(f"{'n':>6} {'m':>7} {'density':<8} {'Dijkstra':>10} "
          f"{'Bellman-F':>10} {'Prim':>10}")
    for r in rows:
        def ms(x):
            return "     n/a" if x != x else f"{x*1e3:9.3f}"
        print(f"{r['n']:>6} {r['m']:>7} {r['density']:<8} "
              f"{ms(r['dijkstra'])} {ms(r['bellman_ford'])} {ms(r['prim'])}")
    print(f"\nCSV: {csv_path}")
    print("Plots: task2_runtime.png, task2_dijkstra_tree.png, task2_mst.png")
