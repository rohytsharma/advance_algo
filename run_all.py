"""Run all five tasks in sequence."""
import runpy
import sys

TASKS = [
    "task1_data_structures.py",
    "task2_graphs.py",
    "task3_strategies.py",
    "task4_nphard.py",
    "task5_concurrent.py",
]

for t in TASKS:
    print("\n" + "#" * 72)
    print("# RUNNING", t)
    print("#" * 72)
    runpy.run_path(t, run_name="__main__")
    sys.stdout.flush()

print("\nAll tasks complete. See results/ for CSVs and PNGs.")
