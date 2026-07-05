#!/usr/bin/env python3
"""
compare_groups.py -- Cross-group comparison wrapper around benchmark_plot.py

benchmark_plot.py's own multi-run comparison (policy_comparison_bar.png +
combined summary table) only fires when multiple runs exist WITHIN a single
process invocation (main(): "if len(runs) > 1: ..."). Since each Scenario /
FOV / Calib / Policy group is normally averaged via its own separate
"python benchmark_plot.py -f N ..." call (one process per group), there is
no single invocation that sees every group's averaged result at once -- so
no cross-group chart is produced by benchmark_plot.py alone.

This script closes that gap WITHOUT modifying benchmark_plot.py: it imports
benchmark_plot as a module, reuses its existing process_file() /
process_averaged_files() / write_summary_table() / plot_policy_comparison() /
_multi_run_dir() functions, computes ONE averaged run object per group
(exactly like "-f N" would internally), then feeds the full list of
per-group averaged runs into the SAME multi-run comparison path main() uses
for single-file lists -- producing one policy_comparison_bar.png and one
combined summary table across ALL groups (e.g. calib vs. non-calib, or
S1 vs. S2 vs. S3 vs. S4, or different FOVs) in a single ./exp/results/*-multi/
folder.

Usage
-----
python compare_groups.py <file1.json> <file2.json> ...

Files are grouped internally by the same (Scenario, FOV, HeightCM, Calib,
Policy) combination as aggregate_benchmarks.ps1, using the filename
convention:

    exp-log-{Scenario}-r{Rep}-{FOV}fov-{HeightCM}cm-{calib|non-calib}-{Policy}[-{N}].json

Each group with >1 file is averaged via benchmark_plot.process_averaged_files();
a group with exactly 1 file is passed through benchmark_plot.process_file().
Per-group individual summary tables are also written (mirroring main()'s
per-run write_summary_table loop), then the combined cross-group comparison
is written last.

This script must live in the SAME directory as benchmark_plot.py (or
benchmark_plot.py must be importable on PYTHONPATH).
"""

import argparse
import os
import re
import sys
from collections import defaultdict
from typing import List

try:
    import benchmark_plot as bp
except ImportError:
    print("ERROR: could not import benchmark_plot.py. "
          "Make sure compare_groups.py is in the same folder as benchmark_plot.py.")
    sys.exit(1)

_FILE_RE = re.compile(
    r"exp-log-(?P<scenario>S\d+)-r(?P<rep>\d+)-(?P<fov>\d+)fov-(?P<cm>\d+)cm-"
    r"(?P<calib>non-calib|calib)-(?P<policy>cooperative|non_cooperative)(?:-(?P<dup>\d+))?\.json$"
)


def _group_key(path: str):
    """Return (scenario, fov, cm, calib, policy) parsed from filename, or
    None if the filename doesn't match the naming convention."""
    name = os.path.basename(path)
    m = _FILE_RE.search(name)
    if not m:
        return None
    g = m.groupdict()
    return (g["scenario"], g["fov"], g["cm"], g["calib"], g["policy"])


def group_files(paths: List[str]) -> "dict[tuple, list]":
    groups = defaultdict(list)
    skipped = []
    for p in paths:
        key = _group_key(p)
        if key is None:
            skipped.append(p)
            continue
        groups[key].append(p)
    if skipped:
        print("Skipping files that don't match the exp-log naming convention:")
        for s in skipped:
            print(f"  {s}")
    return groups


def main():
    parser = argparse.ArgumentParser(
        description="Average each Scenario/FOV/Calib/Policy group of "
                     "experiment JSON files, then produce ONE cross-group "
                     "comparison (policy_comparison_bar.png + summary table) "
                     "across all groups.")
    parser.add_argument("files", nargs="*",
                         help="Paths to experiment .json files (any mix of "
                              "scenarios / FOVs / calib / policies)")
    parser.add_argument("--min-files", type=int, default=1, metavar="N",
                         help="Skip groups with fewer than N matching files "
                              "(default: 1)")
    args = parser.parse_args()

    if not args.files:
        parser.print_help()
        sys.exit(0)

    os.makedirs(os.path.join(".", "exp", "results"), exist_ok=True)

    groups = group_files(args.files)
    if not groups:
        print("No files matched the exp-log naming convention. Nothing to do.")
        sys.exit(1)

    runs = []
    print(f"Found {len(groups)} group(s):\n")
    for key, paths in sorted(groups.items()):
        scenario, fov, cm, calib, policy = key
        tag = f"{scenario}-{fov}fov-{cm}cm-{calib}-{policy}"
        if len(paths) < args.min_files:
            print(f"  [skip] {tag}: {len(paths)} file(s) < --min-files {args.min_files}")
            continue

        print(f"  [group] {tag}: {len(paths)} file(s)")
        for p in sorted(paths):
            print(f"      {os.path.basename(p)}")

        if len(paths) > 1:
            run = bp.process_averaged_files(paths)
        else:
            run = bp.process_file(paths[0])

        runs.append(run)

        # Per-group individual outputs (same as main()'s per-run loop)
        bp.write_summary_table([run], bp._results_dir(run.get("meta", {})))

    if not runs:
        print("\nNo groups met --min-files threshold. Nothing to compare.")
        sys.exit(1)

    if len(runs) > 1:
        multi_dir = bp._multi_run_dir(runs)
        bp.write_summary_table(runs, multi_dir)
        bp.plot_policy_comparison(runs, multi_dir)
        print(f"\nCross-group comparison outputs -> {multi_dir}")
    else:
        print("\nOnly one group matched -- nothing to compare across groups. "
              "Per-group outputs were still written above.")

    print("\nAll outputs written under ./exp/results/")


if __name__ == "__main__":
    main()
