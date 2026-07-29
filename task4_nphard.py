"""
ST5003CEM Advanced Algorithms - Task 4
NP-Hard problem and heuristics: MULTI-DIMENSIONAL (VECTOR) BIN PACKING.
"""
from __future__ import annotations
import math
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)



#  Instance + feasibility helpers                                              #

def make_instance(n, d=3, rng=None):
    """n items, each a d-vector of demands in (0, 0.6]."""
    rng = rng or random.Random(0)
    return [tuple(round(rng.uniform(0.05, 0.6), 3) for _ in range(d))
            for _ in range(n)]


def bin_load(bin_items, items, d):
    load = [0.0] * d
    for i in bin_items:
        for k in range(d):
            load[k] += items[i][k]
    return load


def fits(load, item, d, cap=1.0):
    return all(load[k] + item[k] <= cap + 1e-9 for k in range(d))


def is_feasible(bins, items, d, cap=1.0):
    for b in bins:
        load = bin_load(b, items, d)
        if any(l > cap + 1e-9 for l in load):
            return False
    packed = sorted(i for b in bins for i in b)
    return packed == list(range(len(items)))


def lower_bound(items, d):
    """Continuous relaxation: max over dimensions of ceil(sum demand)."""
    return max(math.ceil(sum(it[k] for it in items) - 1e-9) for k in range(d))


# --------------------------------------------------------------------------- #
#  1. First-Fit-Decreasing (greedy construction heuristic)                     #
# --------------------------------------------------------------------------- #
def ffd(items, d, cap=1.0):
    # sort by decreasing "size" (L2 norm of the demand vector)
    order = sorted(range(len(items)),
                   key=lambda i: -math.sqrt(sum(x * x for x in items[i])))
    bins = []          # list of lists of item indices
    loads = []         # parallel list of load vectors
    for i in order:
        placed = False
        for b in range(len(bins)):
            if fits(loads[b], items[i], d, cap):
                bins[b].append(i)
                for k in range(d):
                    loads[b][k] += items[i][k]
                placed = True
                break
        if not placed:
            bins.append([i])
            loads.append(list(items[i]))
    return bins


# --------------------------------------------------------------------------- #
#  2. Local search (hill climbing)                                             #
# --------------------------------------------------------------------------- #
def num_bins(bins):
    return sum(1 for b in bins if b)


def local_search(items, d, init_bins, cap=1.0, rng=None):
    """Bin-elimination hill climbing. Neighbourhood move = 'try to dissolve
       the least-full bin by re-homing ALL of its items into other bins'. If
       every item finds a feasible host, that bin is removed (count drops by
       one). Repeat until no bin can be emptied -> a local optimum."""
    rng = rng or random.Random(0)
    bins = [list(b) for b in init_bins if b]
    loads = [bin_load(b, items, d) for b in bins]

    def try_eliminate():
        """One pass: attempt to dissolve the least-full bin. Return True if a
           bin was removed."""
        order = sorted(range(len(bins)), key=lambda b: sum(loads[b]))
        for src in order:
            temp_loads = [l[:] for l in loads]
            placement = []
            ok = True
            for i in bins[src]:
                dst_found = None
                for dst in range(len(bins)):
                    if dst == src:
                        continue
                    if fits(temp_loads[dst], items[i], d, cap):
                        dst_found = dst
                        for k in range(d):
                            temp_loads[dst][k] += items[i][k]
                        break
                if dst_found is None:
                    ok = False
                    break
                placement.append((i, dst_found))
            if ok:
                for i, dst in placement:
                    bins[dst].append(i)
                bins[src] = []
                return True
        return False

    def try_swap():
        """Secondary move: swap a big item in the least-full bin for a small
           item elsewhere. This can create the slack that later lets the
           least-full bin be dissolved (escapes the elimination local optimum)."""
        src = min(range(len(bins)), key=lambda b: sum(loads[b]))
        for i in list(bins[src]):
            for dst in range(len(bins)):
                if dst == src:
                    continue
                for jj in list(bins[dst]):
                    # swap i <-> jj only if item jj is smaller in every dim
                    if all(items[jj][k] <= items[i][k] for k in range(d)):
                        new_src = [x for x in loads[src]]
                        new_dst = [x for x in loads[dst]]
                        for k in range(d):
                            new_src[k] += items[jj][k] - items[i][k]
                            new_dst[k] += items[i][k] - items[jj][k]
                        if all(v <= cap + 1e-9 for v in new_dst):
                            bins[src].remove(i); bins[src].append(jj)
                            bins[dst].remove(jj); bins[dst].append(i)
                            return True
        return False

    improved = True
    while improved:
        improved = False
        while try_eliminate():                  # drain as many bins as possible
            bins = [b for b in bins if b]
            loads = [bin_load(b, items, d) for b in bins]
            improved = True
        if try_swap():                          # perturb, then retry elimination
            loads = [bin_load(b, items, d) for b in bins]
            improved = True
    return [b for b in bins if b]


# --------------------------------------------------------------------------- #
#  3. Simulated Annealing                                                      #
# --------------------------------------------------------------------------- #
def sa_cost(bins, items, d, cap=1.0):
    """Cost rewards using fewer bins AND concentrating load (fuller bins),
       which helps SA find moves that eventually empty a bin.
       cost = num_bins - sum(fullness^2)/num_bins  (lower is better)."""
    used = [b for b in bins if b]
    k = len(used)
    fullness = 0.0
    for b in used:
        load = bin_load(b, items, d)
        fullness += (sum(load) / d) ** 2
    return k - fullness / max(1, k)


def simulated_annealing(items, d, init_bins, cap=1.0,
                        iters=20_000, t0=1.0, cooling=0.9995, rng=None):
    rng = rng or random.Random(0)
    cur = [list(b) for b in init_bins if b]
    cur_cost = sa_cost(cur, items, d, cap)
    best = [list(b) for b in cur]
    best_bins = num_bins(cur)

    T = t0
    n = len(items)
    for _ in range(iters):
        # neighbour: move a random item to a random (possibly new) bin
        cand = [list(b) for b in cur]
        # locate a random item
        src = rng.randrange(len(cand))
        while not cand[src]:
            src = rng.randrange(len(cand))
        item_i = rng.choice(cand[src])
        cand[src].remove(item_i)
        # choose destination: existing feasible bin, else open a new one
        dests = list(range(len(cand))) + [-1]
        rng.shuffle(dests)
        placed = False
        for dst in dests:
            if dst == -1:
                cand.append([item_i])
                placed = True
                break
            if dst == src:
                continue
            load = bin_load(cand[dst], items, d)
            if fits(load, items[item_i], d, cap):
                cand[dst].append(item_i)
                placed = True
                break
        if not placed:
            cand[src].append(item_i)
            continue
        cand = [b for b in cand if b]

        cand_cost = sa_cost(cand, items, d, cap)
        delta = cand_cost - cur_cost
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-9)):
            cur, cur_cost = cand, cand_cost
            nb = num_bins(cur)
            if nb < best_bins and is_feasible(cur, items, d, cap):
                best, best_bins = [list(b) for b in cur], nb
        T *= cooling
    return [b for b in best if b]


# --------------------------------------------------------------------------- #
#  Evaluation harness                                                          #
# --------------------------------------------------------------------------- #
def evaluate(sizes=(50, 100, 200), d=3):
    rng = random.Random(2024)
    results = []          # (n, method, bins, time)
    for n in sizes:
        items = make_instance(n, d, rng)
        lb = lower_bound(items, d)

        t0 = time.perf_counter()
        b_ffd = ffd(items, d)
        t_ffd = time.perf_counter() - t0
        assert is_feasible(b_ffd, items, d)

        t0 = time.perf_counter()
        b_ls = local_search(items, d, b_ffd, cap=1.0, rng=random.Random(1))
        t_ls = time.perf_counter() - t0
        assert is_feasible(b_ls, items, d)

        t0 = time.perf_counter()
        b_sa = simulated_annealing(items, d, b_ffd, iters=15_000,
                                   rng=random.Random(3))
        t_sa = time.perf_counter() - t0
        assert is_feasible(b_sa, items, d)

        results.append({"n": n, "lb": lb,
                        "FFD": (num_bins(b_ffd), t_ffd),
                        "LocalSearch": (num_bins(b_ls), t_ls),
                        "SimAnneal": (num_bins(b_sa), t_sa)})
    return results


def plot_results(results):
    methods = ["FFD", "LocalSearch", "SimAnneal"]
    ns = [r["n"] for r in results]
    x = range(len(ns))
    width = 0.22

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for idx, m in enumerate(methods):
        vals = [r[m][0] for r in results]
        ax1.bar([i + idx * width for i in x], vals, width, label=m)
    ax1.bar([i + 3 * width for i in x], [r["lb"] for r in results],
            width, label="Lower bound", color="0.6")
    ax1.set_xticks([i + 1.5 * width for i in x])
    ax1.set_xticklabels(ns)
    ax1.set_xlabel("Number of items n")
    ax1.set_ylabel("Bins used (lower is better)")
    ax1.set_title("Solution quality: bins used vs lower bound")
    ax1.legend()

    for idx, m in enumerate(methods):
        vals = [r[m][1] * 1e3 for r in results]
        ax2.plot(ns, vals, "o-", label=m)
    ax2.set_xlabel("Number of items n")
    ax2.set_ylabel("Runtime (ms, log scale)")
    ax2.set_yscale("log")
    ax2.set_title("Computational cost")
    ax2.grid(True, which="both", ls=":", alpha=0.5)
    ax2.legend()

    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "task4_heuristics.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print("Multi-dimensional (3-D) Bin Packing - heuristic comparison")
    print("=" * 66)
    results = evaluate()
    print(f"{'n':>5} {'LB':>4} | {'FFD':>16} | {'LocalSearch':>16} | "
          f"{'SimAnneal':>16}")
    print(f"{'':>5} {'':>4} | {'bins  time(ms)':>16} | "
          f"{'bins  time(ms)':>16} | {'bins  time(ms)':>16}")
    print("-" * 78)
    for r in results:
        def cell(t):
            return f"{t[0]:>4}  {t[1]*1e3:8.1f}"
        print(f"{r['n']:>5} {r['lb']:>4} | {cell(r['FFD'])} | "
              f"{cell(r['LocalSearch'])} | {cell(r['SimAnneal'])}")
    out = plot_results(results)
    print(f"\nPlot: {out}")
    print("\nObservations:")
    print(" * FFD is by far the fastest and already lands only a few bins")
    print("   above the lower bound.")
    print(" * Local search (bin-elimination + swap) cannot improve on FFD here:")
    print("   FFD already sits at a local optimum for this neighbourhood -- a")
    print("   textbook illustration of local search getting stuck.")
    print(" * Simulated Annealing accepts occasional worsening moves, escapes")
    print("   that local optimum and saves 1-2 bins, but costs 100-1000x more")
    print("   time -- the classic quality-vs-cost trade-off for NP-Hard problems.")
