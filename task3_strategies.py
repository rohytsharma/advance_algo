"""
ST5003CEM Advanced Algorithms - Task 3
Algorithmic strategies for complex problems.

  (a) Dynamic Programming : Weighted Job Scheduling with time windows.
  (b) Greedy              : Minimum Number of Platforms (train station).
  (c) Backtracking        : Knight's Tour with Warnsdorff pruning.

"""
from __future__ import annotations
import bisect
import random
import time


# =========================================================================== #
#  (a) DYNAMIC PROGRAMMING - Weighted Job Scheduling with time windows         #
# --------------------------------------------------------------------------- #
#  Each job j has (start_j, end_j, profit_j). Two jobs conflict if their       #
#  [start, end) intervals overlap. Select a non-overlapping subset of jobs     #
#  that MAXIMISES total profit.                                                #
#                                                                              #
#  Subproblem : sort jobs by end time.  OPT(i) = max profit using only the     #
#               first i jobs (jobs 1..i in end-time order).                    #
#  Recurrence : OPT(i) = max( OPT(i-1),                    # skip job i        #
#                             profit_i + OPT(p(i)) )       # take job i        #
#               where p(i) = largest index j < i whose end <= start_i          #
#               (the last job that does not conflict with i).                  #
#  Base case  : OPT(0) = 0.                                                     #
#  Complexity : sorting O(n log n); each p(i) via binary search O(log n);      #
#               table fill O(n)  =>  O(n log n) time, O(n) space.              #
# =========================================================================== #
def weighted_job_scheduling(jobs):
    """jobs: list of (start, end, profit). Returns (max_profit, chosen list)."""
    if not jobs:
        return 0, []
    jobs = sorted(jobs, key=lambda j: j[1])          # sort by end time
    n = len(jobs)
    ends = [j[1] for j in jobs]

    # p[i] = index (1-based count) of last non-conflicting job before i
    p = [0] * n
    for i in range(n):
        # last job whose end <= start_i ; bisect on end times
        p[i] = bisect.bisect_right(ends, jobs[i][0], 0, i)

    dp = [0] * (n + 1)                                # dp[i] over first i jobs
    take = [False] * (n + 1)
    for i in range(1, n + 1):
        job = jobs[i - 1]
        incl = job[2] + dp[p[i - 1]]
        if incl > dp[i - 1]:
            dp[i] = incl
            take[i] = True
        else:
            dp[i] = dp[i - 1]

    # reconstruct chosen jobs
    chosen = []
    i = n
    while i > 0:
        if take[i]:
            chosen.append(jobs[i - 1])
            i = p[i - 1]
        else:
            i -= 1
    chosen.reverse()
    return dp[n], chosen


def brute_force_jobs(jobs):
    """Exponential O(2^n) reference to validate the DP on small inputs."""
    n = len(jobs)
    best = 0
    for mask in range(1 << n):
        subset = [jobs[i] for i in range(n) if mask & (1 << i)]
        subset.sort(key=lambda j: j[0])
        ok = all(subset[k][1] <= subset[k + 1][0]
                 for k in range(len(subset) - 1))
        if ok:
            best = max(best, sum(j[2] for j in subset))
    return best


# =========================================================================== #
#  (b) GREEDY - Minimum Number of Platforms                                     #
# --------------------------------------------------------------------------- #
#  Given arrival[] and departure[] times of trains, find the minimum number    #
#  of platforms so that no train waits.                                        #
#                                                                              #
#  Greedy choice : sort arrivals and departures separately; sweep in time      #
#  order. On an arrival we need a platform (+1); on a departure one frees       #
#  (-1). The running maximum of concurrent trains is the answer.               #
#                                                                              #
#  Optimality (proof sketch): the minimum number of platforms equals the       #
#  maximum number of trains present at the station simultaneously (a clique    #
#  in the interval-overlap graph). No schedule can use fewer platforms than    #
#  the peak overlap, and the sweep realises exactly that peak, so it is        #
#  optimal. Complexity: O(n log n) for the two sorts, O(n) sweep.              #
# =========================================================================== #
def min_platforms(arrivals, departures):
    arr = sorted(arrivals)
    dep = sorted(departures)
    n = len(arr)
    i = j = 0
    platforms = max_platforms = 0
    while i < n and j < n:
        if arr[i] <= dep[j]:        # a train arrives before/at next departure
            platforms += 1
            max_platforms = max(max_platforms, platforms)
            i += 1
        else:                       # a train departs, freeing a platform
            platforms -= 1
            j += 1
    return max_platforms


def brute_force_platforms(arrivals, departures):
    """O(n^2) reference: max trains present at any arrival instant."""
    best = 0
    for a in arrivals:
        count = sum(1 for s, e in zip(arrivals, departures) if s <= a <= e)
        best = max(best, count)
    return best


# =========================================================================== #
#  (c) BACKTRACKING - Knight's Tour with Warnsdorff pruning                     #
# --------------------------------------------------------------------------- #
#  Find a sequence of knight moves visiting every square of an n x n board      #
#  exactly once.                                                                #
#                                                                              #
#  Worst case is exponential: naive backtracking explores up to 8^(n^2) move   #
#  sequences. Pruning = WARNSDORFF'S RULE: always move to the reachable        #
#  unvisited square that itself has the FEWEST onward moves. This heuristic     #
#  orders the branches so the search almost always descends straight to a       #
#  full tour without backtracking, turning an intractable search into a         #
#  near-linear one in practice.                                                #
# =========================================================================== #
KNIGHT_MOVES = [(2, 1), (1, 2), (-1, 2), (-2, 1),
                (-2, -1), (-1, -2), (1, -2), (2, -1)]


def knights_tour(n, start=(0, 0), use_warnsdorff=True):
    board = [[-1] * n for _ in range(n)]
    stats = {"nodes": 0, "backtracks": 0}

    def on_board(r, c):
        return 0 <= r < n and 0 <= c < n

    def degree(r, c):
        """Number of unvisited squares reachable from (r, c)."""
        d = 0
        for dr, dc in KNIGHT_MOVES:
            nr, nc = r + dr, c + dc
            if on_board(nr, nc) and board[nr][nc] == -1:
                d += 1
        return d

    def solve(r, c, move_idx):
        stats["nodes"] += 1
        board[r][c] = move_idx
        if move_idx == n * n - 1:
            return True
        nbrs = []
        for dr, dc in KNIGHT_MOVES:
            nr, nc = r + dr, c + dc
            if on_board(nr, nc) and board[nr][nc] == -1:
                nbrs.append((nr, nc))
        if use_warnsdorff:                       # PRUNING: fewest-onward first
            nbrs.sort(key=lambda p: degree(*p))
        for nr, nc in nbrs:
            if solve(nr, nc, move_idx + 1):
                return True
        board[r][c] = -1                         # undo (backtrack)
        stats["backtracks"] += 1
        return False

    ok = solve(start[0], start[1], 0)
    return ok, board, stats


# =========================================================================== #
#  Demonstration / self-tests                                                  #
# =========================================================================== #
def demo_dp():
    print("=" * 70)
    print("(a) DYNAMIC PROGRAMMING - Weighted Job Scheduling")
    print("=" * 70)
    jobs = [(1, 4, 50), (3, 5, 20), (0, 6, 70), (5, 7, 30),
            (3, 8, 60), (6, 9, 40), (8, 10, 55)]
    profit, chosen = weighted_job_scheduling(jobs)
    print(f"jobs (start,end,profit): {jobs}")
    print(f"max profit = {profit}")
    print(f"chosen jobs = {chosen}")

    # validate DP == brute force on many random small instances
    rng = random.Random(1)
    mismatches = 0
    for _ in range(500):
        m = rng.randint(1, 12)
        js = []
        for _ in range(m):
            s = rng.randint(0, 20)
            e = s + rng.randint(1, 8)
            js.append((s, e, rng.randint(1, 100)))
        if weighted_job_scheduling(js)[0] != brute_force_jobs(js):
            mismatches += 1
    print(f"DP vs brute force over 500 random instances: "
          f"{'ALL MATCH' if mismatches == 0 else f'{mismatches} MISMATCHES'}")

    # timing on a large instance
    rng2 = random.Random(5)
    big = []
    for _ in range(200_000):
        s = rng2.randint(0, 10**7)
        big.append((s, s + rng2.randint(1, 1000), rng2.randint(1, 1000)))
    t0 = time.perf_counter()
    p, _ = weighted_job_scheduling(big)
    dt = time.perf_counter() - t0
    print(f"200,000 jobs solved in {dt*1e3:.1f} ms (O(n log n)); "
          f"max profit = {p}")


def demo_greedy():
    print("\n" + "=" * 70)
    print("(b) GREEDY - Minimum Number of Platforms")
    print("=" * 70)
    arr = [900, 940, 950, 1100, 1500, 1800]
    dep = [910, 1200, 1120, 1130, 1900, 2000]
    ans = min_platforms(arr, dep)
    print(f"arrivals   = {arr}")
    print(f"departures = {dep}")
    print(f"minimum platforms = {ans} (expected 3)")

    # validate greedy == brute force on random instances
    rng = random.Random(2)
    mismatches = 0
    for _ in range(2000):
        k = rng.randint(1, 12)
        a, d = [], []
        for _ in range(k):
            s = rng.randint(0, 100)
            a.append(s)
            d.append(s + rng.randint(0, 50))
        if min_platforms(a, d) != brute_force_platforms(a, d):
            mismatches += 1
    print(f"Greedy vs brute force over 2000 random instances: "
          f"{'ALL MATCH (greedy is optimal)' if mismatches == 0 else f'{mismatches} MISMATCHES'}")


def demo_backtracking():
    print("\n" + "=" * 70)
    print("(c) BACKTRACKING - Knight's Tour (Warnsdorff pruning)")
    print("=" * 70)
    for n in (5, 6, 8):
        t0 = time.perf_counter()
        ok, board, stats = knights_tour(n, (0, 0), use_warnsdorff=True)
        dt = time.perf_counter() - t0
        print(f"n={n}: tour found={ok}, nodes explored={stats['nodes']:>6}, "
              f"backtracks={stats['backtracks']:>5}, time={dt*1e3:7.2f} ms "
              f"(perfect search would visit {n*n} nodes)")

    # contrast: WITHOUT pruning on a small board to show the blow-up
    print("\nWithout Warnsdorff pruning (naive order):")
    for n in (5, 6):
        t0 = time.perf_counter()
        ok, _, stats = knights_tour(n, (0, 0), use_warnsdorff=False)
        dt = time.perf_counter() - t0
        print(f"n={n}: tour found={ok}, nodes explored={stats['nodes']:>8}, "
              f"backtracks={stats['backtracks']:>8}, time={dt*1e3:8.2f} ms")


if __name__ == "__main__":
    demo_dp()
    demo_greedy()
    demo_backtracking()
