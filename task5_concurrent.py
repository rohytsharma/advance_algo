"""
ST5003CEM Advanced Algorithms - Task 5
Concurrent Programming: PARALLEL MULTI-SOURCE DIJKSTRA.

Synchronisation primitives (brief asks for mutexes / semaphores / condition
variables):
    * multiprocessing.Queue  -- the shared work queue. It is internally
                                implemented with a MUTEX + a CONDITION VARIABLE
                                (and a SEMAPHORE tracking capacity); workers
                                block on it until an item is available.
    * multiprocessing.Lock   -- an explicit MUTEX guarding the shared result
                                accumulator `total` and `done` counter. This
                                is the CRITICAL SECTION; without it the
                                read-modify-write of the shared Value races.

"""
from __future__ import annotations
import csv
import multiprocessing as mp
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from task2_graphs import random_sparse_graph, dijkstra

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Module-global graph: with fork() every worker inherits it copy-on-write,
# so we never pickle/ship the graph to workers.
G = None


# --------------------------------------------------------------------------- #
#  Sequential baseline                                                         #
# --------------------------------------------------------------------------- #
def sequential(sources):
    total = 0.0
    for s in sources:
        dist, _ = dijkstra(G, s)
        total += sum(d for d in dist if d != float("inf"))
    return total


# --------------------------------------------------------------------------- #
#  Parallel worker                                                             #
# --------------------------------------------------------------------------- #
def worker(task_q, lock, shared_total, shared_done):
    """Pop sources off the shared queue until the sentinel; accumulate the
       result into shared memory inside a mutex-guarded critical section."""
    local = 0.0
    local_count = 0
    while True:
        s = task_q.get()                # blocks on the queue's condition var
        if s is None:                   # sentinel -> shut down
            break
        dist, _ = dijkstra(G, s)
        local += sum(d for d in dist if d != float("inf"))
        local_count += 1
    # ---- CRITICAL SECTION: update shared accumulators under the mutex ----
    with lock:
        shared_total.value += local
        shared_done.value += local_count


def parallel(sources, num_procs):
    task_q = mp.Queue()
    for s in sources:
        task_q.put(s)
    for _ in range(num_procs):           # one sentinel per worker
        task_q.put(None)

    lock = mp.Lock()                     # the MUTEX
    shared_total = mp.Value("d", 0.0)    # shared float accumulator
    shared_done = mp.Value("i", 0)       # shared processed-count

    procs = [mp.Process(target=worker,
                        args=(task_q, lock, shared_total, shared_done))
             for _ in range(num_procs)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()                         # barrier: wait for all workers
    return shared_total.value, shared_done.value


# --------------------------------------------------------------------------- #
#  Optional: show that THREADS do not help (GIL) for the same workload         #
# --------------------------------------------------------------------------- #
def threaded(sources, num_threads):
    import threading
    from queue import Queue
    q = Queue()
    for s in sources:
        q.put(s)
    for _ in range(num_threads):
        q.put(None)
    lock = threading.Lock()
    total = [0.0]

    def tworker():
        local = 0.0
        while True:
            s = q.get()
            if s is None:
                break
            dist, _ = dijkstra(G, s)
            local += sum(d for d in dist if d != float("inf"))
        with lock:                       # mutex-guarded critical section
            total[0] += local

    ts = [threading.Thread(target=tworker) for _ in range(num_threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return total[0]



#  Experiment                                                                  #

def run(n_nodes=4000, n_sources=240, proc_counts=(1, 2, 4, 8), repeats=2):
    global G
    G = random_sparse_graph(n_nodes, avg_degree=6, directed=True)
    sources = list(range(0, n_nodes, max(1, n_nodes // n_sources)))[:n_sources]

    # sequential baseline
    seq_best = float("inf")
    ref = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        ref = sequential(sources)
        seq_best = min(seq_best, time.perf_counter() - t0)

    rows = []
    for P in proc_counts:
        best = float("inf")
        for _ in range(repeats):
            t0 = time.perf_counter()
            total, done = parallel(sources, P)
            dt = time.perf_counter() - t0
            best = min(best, dt)
        assert done == len(sources), "not all sources processed"
        assert abs(total - ref) < 1e-6 * max(1.0, abs(ref)), "wrong result"
        speedup = seq_best / best
        rows.append({"procs": P, "time": best, "speedup": speedup,
                     "efficiency": speedup / P})

    # single thread-vs-process comparison at 4 workers to expose the GIL
    t0 = time.perf_counter()
    threaded(sources, 4)
    thr4 = time.perf_counter() - t0

    return seq_best, rows, thr4, len(sources)


def save_csv(rows):
    path = os.path.join(RESULTS_DIR, "task5_results.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["procs", "time", "speedup",
                                          "efficiency"])
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.5f}" if isinstance(v, float) else v)
                        for k, v in r.items()})
    return path


def plot(seq_best, rows):
    P = [r["procs"] for r in rows]
    sp = [r["speedup"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    ax1.plot(P, sp, "o-", label="Measured speed-up (processes)")
    ax1.plot(P, P, "--", color="0.6", label="Ideal linear speed-up")
    ax1.set_xlabel("Number of workers")
    ax1.set_ylabel("Speed-up  (T_seq / T_parallel)")
    ax1.set_title("Task 5 - Parallel multi-source Dijkstra speed-up")
    ax1.set_xticks(P)
    ax1.grid(True, ls=":", alpha=0.5)
    ax1.legend()

    eff = [r["efficiency"] * 100 for r in rows]
    ax2.plot(P, eff, "s-", color="#c0392b")
    ax2.set_xlabel("Number of workers")
    ax2.set_ylabel("Parallel efficiency (%)")
    ax2.set_title("Efficiency falls as overhead grows")
    ax2.set_xticks(P)
    ax2.set_ylim(0, 110)
    ax2.grid(True, ls=":", alpha=0.5)

    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "task5_speedup.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


if __name__ == "__main__":
    try:
        mp.set_start_method("fork")
    except RuntimeError:
        pass
    print("Parallel multi-source Dijkstra - process scalability")
    print("=" * 60)
    seq_best, rows, thr4, n_src = run()
    print(f"Graph: 4000 nodes, ~6 avg out-degree; {n_src} source vertices")
    print(f"Sequential baseline: {seq_best*1e3:.1f} ms\n")
    print(f"{'workers':>7} {'time(s)':>9} {'speedup':>8} {'efficiency':>11}")
    for r in rows:
        print(f"{r['procs']:>7} {r['time']:>9.3f} {r['speedup']:>8.2f} "
              f"{r['efficiency']*100:>10.0f}%")
    print(f"\nGIL check - 4 THREADS took {thr4*1e3:.1f} ms vs sequential "
          f"{seq_best*1e3:.1f} ms")
    print("  (threads give ~no speed-up: the GIL serialises pure-Python work,")
    print("   which is exactly why we use processes for real parallelism.)")

    csv_path = save_csv(rows)
    out = plot(seq_best, rows)
    print(f"\nCSV : {csv_path}")
    print(f"Plot: {out}")
    print("\nOverheads limiting scalability: process creation/fork cost, "
          "queue\nserialisation of each task, lock contention on the shared "
          "accumulator,\nand memory/cache pressure from N interpreters.")
