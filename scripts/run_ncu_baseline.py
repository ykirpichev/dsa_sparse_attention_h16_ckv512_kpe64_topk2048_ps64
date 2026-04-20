"""
Run flashinfer_bench_run_ncu on the packed Triton solution (first workload by default).

Requires:
  - FIB_DATASET_PATH pointing at a full FlashInfer trace tree with workloads (e.g. mlsys26-contest clone)
  - `ncu` on PATH (Nsight Compute)
  - Often **root** for GPU hardware counters (see NCU warning if output is empty)

Example:
  python scripts/pack_solution.py
  export FIB_DATASET_PATH=/path/to/mlsys26-contest
  sudo -E env PATH="$PATH" $(which python) scripts/run_ncu_baseline.py --set basic
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flashinfer_bench import Solution, TraceSet
from flashinfer_bench.agents import flashinfer_bench_run_ncu

from scripts.pack_solution import pack_solution


def main() -> None:
    p = argparse.ArgumentParser(description="Run Nsight Compute via flashinfer_bench_run_ncu")
    p.add_argument(
        "--dataset",
        default=os.environ.get("FIB_DATASET_PATH"),
        help="Trace set root (default: FIB_DATASET_PATH)",
    )
    p.add_argument("--workload-index", type=int, default=0, help="Index into workload list for this definition")
    p.add_argument("--set", dest="ncu_set", default="basic", help="NCU section set (e.g. basic, full)")
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--no-pack", action="store_true", help="Skip pack_solution.py")
    args = p.parse_args()

    if not args.dataset:
        print("Set --dataset or FIB_DATASET_PATH.", file=sys.stderr)
        sys.exit(1)

    if not args.no_pack:
        pack_solution()

    solution = Solution.model_validate_json((PROJECT_ROOT / "solution.json").read_text())
    trace_set = TraceSet.from_path(args.dataset)
    traces = trace_set.workloads.get(solution.definition, [])
    if not traces:
        print(f"No workload traces for {solution.definition} under {args.dataset}", file=sys.stderr)
        sys.exit(1)
    if args.workload_index < 0 or args.workload_index >= len(traces):
        print("workload-index out of range", file=sys.stderr)
        sys.exit(1)

    workload = traces[args.workload_index].workload
    print(f"Definition: {solution.definition}  workload[{args.workload_index}] axes={workload.axes}", flush=True)

    result = flashinfer_bench_run_ncu(
        solution,
        workload,
        trace_set_path=args.dataset,
        set=args.ncu_set,
        page="details",
        timeout=args.timeout,
        max_lines=None,
    )
    print(result)
    if result.strip().startswith("ERROR:"):
        sys.exit(1)
    if "Insufficient privileges" in result or len(result) < 200:
        print(
            "\n[Note] If output is only an NCU privileges warning, run with sudo and preserve PATH, e.g.\n"
            "  sudo -E env PATH=\"$PATH\" PYTHONPATH=\"$PYTHONPATH\" "
            f"{sys.executable} {Path(__file__)} --no-pack --dataset {args.dataset}\n",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
