"""
ST5003CEM Advanced Algorithms - Task 1
Implements and benchmarks:
    * Binary Search Tree (BST)                -- unbalanced ordered map
    * AVL Tree (self-balancing BST)           -- guaranteed O(log n) height
    * Min-Heap (binary heap priority queue)   -- next-nearest-city access
    * Hash Table (separate chaining)          -- O(1) average city lookup
Each city record stores: name (key), coordinates, population, distance.
"""
from __future__ import annotations
import csv
import os
import random
import sys
import time

sys.setrecursionlimit(50_000)  # degenerate BST on sorted input recurses deeply
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)



#  City record (the "associated data" stored at every key)                    #

@dataclass
class City:
    key: int                 # numeric city id used as the ordering key
    name: str
    x: float                 # coordinate
    y: float                 # coordinate
    population: int
    distance: float = 0.0    # distance from an origin (used by the heap)



#  1. Binary Search Tree                                                       #

class BSTNode:
    __slots__ = ("key", "value", "left", "right")

    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.left = None
        self.right = None


class BST:
    """Ordered map. Average O(log n) ops, but O(n) worst case on sorted input."""

    def __init__(self):
        self.root = None
        self._n = 0

    def insert(self, key, value):
        self.root = self._insert(self.root, key, value)

    def _insert(self, node, key, value):
        if node is None:
            self._n += 1
            return BSTNode(key, value)
        if key < node.key:
            node.left = self._insert(node.left, key, value)
        elif key > node.key:
            node.right = self._insert(node.right, key, value)
        else:
            node.value = value          # update on duplicate key
        return node

    def search(self, key):
        node = self.root
        while node is not None:
            if key == node.key:
                return node.value
            node = node.left if key < node.key else node.right
        return None

    def delete(self, key):
        self.root = self._delete(self.root, key)

    def _delete(self, node, key):
        if node is None:
            return None
        if key < node.key:
            node.left = self._delete(node.left, key)
        elif key > node.key:
            node.right = self._delete(node.right, key)
        else:
            if node.left is None:
                return node.right
            if node.right is None:
                return node.left
            succ = node.right           # in-order successor
            while succ.left is not None:
                succ = succ.left
            node.key, node.value = succ.key, succ.value
            node.right = self._delete(node.right, succ.key)
        return node

    def height(self):
        def h(n):
            return 0 if n is None else 1 + max(h(n.left), h(n.right))
        return h(self.root)


#  2. AVL Tree (self-balancing)                                                #

class AVLNode:
    __slots__ = ("key", "value", "left", "right", "height")

    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.left = None
        self.right = None
        self.height = 1


class AVLTree:
    """Height-balanced BST: insert/delete/search all guaranteed O(log n)."""

    def __init__(self):
        self.root = None

    @staticmethod
    def _h(node):
        return node.height if node else 0

    def _balance(self, node):
        return self._h(node.left) - self._h(node.right)

    def _update(self, node):
        node.height = 1 + max(self._h(node.left), self._h(node.right))

    def _rotate_right(self, y):
        x = y.left
        y.left = x.right
        x.right = y
        self._update(y)
        self._update(x)
        return x

    def _rotate_left(self, x):
        y = x.right
        x.right = y.left
        y.left = x
        self._update(x)
        self._update(y)
        return y

    def _rebalance(self, node):
        self._update(node)
        bf = self._balance(node)
        if bf > 1:                                  # left heavy
            if self._balance(node.left) < 0:        # left-right
                node.left = self._rotate_left(node.left)
            return self._rotate_right(node)
        if bf < -1:                                 # right heavy
            if self._balance(node.right) > 0:       # right-left
                node.right = self._rotate_right(node.right)
            return self._rotate_left(node)
        return node

    def insert(self, key, value):
        self.root = self._insert(self.root, key, value)

    def _insert(self, node, key, value):
        if node is None:
            return AVLNode(key, value)
        if key < node.key:
            node.left = self._insert(node.left, key, value)
        elif key > node.key:
            node.right = self._insert(node.right, key, value)
        else:
            node.value = value
            return node
        return self._rebalance(node)

    def search(self, key):
        node = self.root
        while node is not None:
            if key == node.key:
                return node.value
            node = node.left if key < node.key else node.right
        return None

    def delete(self, key):
        self.root = self._delete(self.root, key)

    def _delete(self, node, key):
        if node is None:
            return None
        if key < node.key:
            node.left = self._delete(node.left, key)
        elif key > node.key:
            node.right = self._delete(node.right, key)
        else:
            if node.left is None:
                return node.right
            if node.right is None:
                return node.left
            succ = node.right
            while succ.left is not None:
                succ = succ.left
            node.key, node.value = succ.key, succ.value
            node.right = self._delete(node.right, succ.key)
        return self._rebalance(node)

    def height(self):
        return self._h(self.root)



#  3. Min-Heap priority queue (next nearest city)                             #

class MinHeap:
    """Binary min-heap keyed on a priority value (e.g. distance)."""

    def __init__(self):
        self._a = []                    # list of (priority, item)

    def __len__(self):
        return len(self._a)

    def push(self, priority, item):
        a = self._a
        a.append((priority, item))
        i = len(a) - 1
        while i > 0:                    # sift up
            parent = (i - 1) >> 1
            if a[parent][0] <= a[i][0]:
                break
            a[parent], a[i] = a[i], a[parent]
            i = parent

    def pop(self):
        a = self._a
        if not a:
            raise IndexError("pop from empty heap")
        top = a[0]
        last = a.pop()
        if a:
            a[0] = last
            i, n = 0, len(a)
            while True:                 # sift down
                l, r, smallest = 2 * i + 1, 2 * i + 2, i
                if l < n and a[l][0] < a[smallest][0]:
                    smallest = l
                if r < n and a[r][0] < a[smallest][0]:
                    smallest = r
                if smallest == i:
                    break
                a[i], a[smallest] = a[smallest], a[i]
                i = smallest
        return top

    def peek(self):
        return self._a[0]


#  4. Hash Table (separate chaining)                                           #

class HashTable:
    """Chaining hash table with dynamic resize to keep load factor bounded."""

    def __init__(self, capacity=16, max_load=0.75):
        self._cap = capacity
        self._buckets = [[] for _ in range(capacity)]
        self._n = 0
        self._max_load = max_load

    def _index(self, key):
        return hash(key) & (self._cap - 1)      # cap is a power of two

    def _resize(self):
        old = self._buckets
        self._cap *= 2
        self._buckets = [[] for _ in range(self._cap)]
        self._n = 0
        for bucket in old:
            for k, v in bucket:
                self.insert(k, v)

    def insert(self, key, value):
        bucket = self._buckets[self._index(key)]
        for i, (k, _) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return
        bucket.append((key, value))
        self._n += 1
        if self._n / self._cap > self._max_load:
            self._resize()

    def search(self, key):
        for k, v in self._buckets[self._index(key)]:
            if k == key:
                return v
        return None

    def delete(self, key):
        bucket = self._buckets[self._index(key)]
        for i, (k, _) in enumerate(bucket):
            if k == key:
                bucket.pop(i)
                self._n -= 1
                return True
        return False



#  Benchmark harness                                                           #

def make_cities(n, rng):
    cities = []
    for i in range(n):
        cities.append(City(
            key=i,
            name=f"City{i}",
            x=rng.uniform(0, 1000),
            y=rng.uniform(0, 1000),
            population=rng.randint(1_000, 5_000_000),
            distance=rng.uniform(0, 5000),
        ))
    return cities


def timed(fn, repeats=1):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def bench_structure(name, factory, cities, do_insert, do_search, do_delete):
    struct = factory()
    keys = [c.key for c in cities]

    t_ins = timed(lambda: do_insert(struct, cities))
    # search a random shuffled order of every key
    lookup = keys[:]
    random.Random(1).shuffle(lookup)
    t_search = timed(lambda: do_search(struct, lookup))
    t_del = timed(lambda: do_delete(struct, lookup))
    return {"structure": name, "insert": t_ins,
            "search": t_search, "delete": t_del}


def run(sizes=(100, 1_000, 10_000)):
    rng = random.Random(42)
    rows = []

    for n in sizes:
        cities = make_cities(n, rng)
        # Insertion order is randomised so the BST is near-balanced
        # (keys are 0..n-1, so inserting in list order would be worst case).
        rng.shuffle(cities)

        # ---- BST (random insertion order -> near-balanced) ----
        rows.append({"n": n, **bench_structure(
            "BST", BST, cities,
            lambda s, cs: [s.insert(c.key, c) for c in cs],
            lambda s, ks: [s.search(k) for k in ks],
            lambda s, ks: [s.delete(k) for k in ks])})

        # ---- BST worst case (sorted insertion -> degenerate list) ----
        # Capped at n<=1000: recursion depth == n, and deeper risks a C-stack
        # overflow. 1000 already demonstrates the O(n^2) degeneration clearly.
        if n <= 1000:
            sorted_cities = sorted(cities, key=lambda c: c.key)
            bst_deg = BST()
            t_ins = timed(lambda: [bst_deg.insert(c.key, c)
                                   for c in sorted_cities])
            rows.append({"n": n, "structure": "BST(sorted-input)",
                         "insert": t_ins, "search": float("nan"),
                         "delete": float("nan"), "height": bst_deg.height()})

        # ---- AVL ----
        rows.append({"n": n, **bench_structure(
            "AVL", AVLTree, cities,
            lambda s, cs: [s.insert(c.key, c) for c in cs],
            lambda s, ks: [s.search(k) for k in ks],
            lambda s, ks: [s.delete(k) for k in ks])})

        # ---- Hash Table ----
        rows.append({"n": n, **bench_structure(
            "HashTable", HashTable, cities,
            lambda s, cs: [s.insert(c.key, c) for c in cs],
            lambda s, ks: [s.search(k) for k in ks],
            lambda s, ks: [s.delete(k) for k in ks])})

        # ---- Min-Heap (insert all, then pop all == priority ordering) ----
        heap = MinHeap()
        t_ins = timed(lambda: [heap.push(c.distance, c.key) for c in cities])
        heap2 = MinHeap()
        for c in cities:
            heap2.push(c.distance, c.key)
        t_pop = timed(lambda: [heap2.pop() for _ in range(len(heap2))])
        rows.append({"n": n, "structure": "MinHeap",
                     "insert": t_ins, "search": t_pop, "delete": float("nan")})

    # record tree heights for the report
    heights = {}
    for n in sizes:
        cities = make_cities(n, rng)
        rng.shuffle(cities)
        b, a = BST(), AVLTree()
        for c in cities:
            b.insert(c.key, c)
            a.insert(c.key, c)
        heights[n] = (b.height(), a.height())
    return rows, heights


def save_csv(rows):
    path = os.path.join(RESULTS_DIR, "task1_results.csv")
    fields = ["n", "structure", "insert", "search", "delete"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def plot(rows, sizes):
    for op in ("insert", "search"):
        plt.figure(figsize=(7, 4.5))
        for name in ("BST", "AVL", "HashTable", "MinHeap"):
            xs, ys = [], []
            for n in sizes:
                for r in rows:
                    if r.get("structure") == name and r["n"] == n:
                        v = r.get(op, float("nan"))
                        if v == v:                  # not NaN
                            xs.append(n)
                            ys.append(v * 1e3)      # ms
            if xs:
                label = name + (" (pop-all)" if name == "MinHeap"
                                and op == "search" else "")
                plt.plot(xs, ys, "o-", label=label)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Number of nodes (n)")
        plt.ylabel(f"Total {op} time for n ops (ms)")
        plt.title(f"Task 1 - {op.capitalize()} performance vs n")
        plt.grid(True, which="both", ls=":", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        out = os.path.join(RESULTS_DIR, f"task1_{op}.png")
        plt.savefig(out, dpi=130)
        plt.close()


if __name__ == "__main__":
    sizes = (100, 1_000, 10_000)
    rows, heights = run(sizes)
    csv_path = save_csv(rows)
    plot(rows, sizes)

    print("=== Task 1 wall-clock timings (total time for n operations) ===")
    print(f"{'n':>7} {'structure':<18} {'insert(ms)':>11} "
          f"{'search(ms)':>11} {'delete(ms)':>11}")
    for r in rows:
        def ms(x):
            return "     n/a" if x != x else f"{x*1e3:10.3f}"
        print(f"{r['n']:>7} {r['structure']:<18} "
              f"{ms(r.get('insert', float('nan')))} "
              f"{ms(r.get('search', float('nan')))} "
              f"{ms(r.get('delete', float('nan')))}")

    print("\n=== Tree heights (random insertion order) ===")
    for n, (bh, ah) in heights.items():
        import math
        print(f"n={n:>6}: BST height={bh:>3}, AVL height={ah:>3}, "
              f"log2(n)~={math.log2(n):.1f}")
    print(f"\nCSV : {csv_path}")
    print("Plots: results/task1_insert.png, results/task1_search.png")
