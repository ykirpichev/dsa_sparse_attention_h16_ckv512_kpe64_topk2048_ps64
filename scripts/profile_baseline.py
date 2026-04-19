"""
Profile the packed Triton baseline with PyTorch profiler (CUDA kernels + Python).

Requires: CUDA, FIB_DATASET_PATH, and `python scripts/pack_solution.py` run first.

For Nsight Compute on a full NVIDIA stack, use flashinfer_bench.agents.flashinfer_bench_run_ncu
(see README.md) with solution.json and a workload JSON from the trace set.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from flashinfer_bench import Solution, TraceSet
from flashinfer_bench.bench.utils import gen_inputs
from flashinfer_bench.compile import BuilderRegistry

from scripts.pack_solution import pack_solution


def main() -> None:
    dataset = os.environ.get("FIB_DATASET_PATH")
    if not dataset:
        print("Set FIB_DATASET_PATH to the flashinfer-trace / mlsys26-contest dataset.", file=sys.stderr)
        sys.exit(1)
    if not torch.cuda.is_available():
        print("CUDA is required for this profiler.", file=sys.stderr)
        sys.exit(1)

    print("Packing solution...")
    pack_solution()

    solution_path = PROJECT_ROOT / "solution.json"
    solution = Solution.model_validate_json(solution_path.read_text())
    trace_set = TraceSet.from_path(dataset)
    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])
    if not workloads:
        print(f"No workloads for {solution.definition}", file=sys.stderr)
        sys.exit(1)

    workload = workloads[0]
    device = "cuda:0"
    inputs = gen_inputs(definition, workload, device=device)

    print("Building runnable...")
    runnable = BuilderRegistry.get_instance().build(definition, solution)

    # Warmup (Triton JIT, caches)
    for _ in range(3):
        runnable(*inputs)
    torch.cuda.synchronize()

    print("Running torch.profiler (CUDA + CPU stacks)...")
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
        for _ in range(5):
            runnable(*inputs)
        torch.cuda.synchronize()

    print("\n=== Top CUDA ops by self CUDA time ===")
    print(
        prof.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=25,
        )
    )


if __name__ == "__main__":
    main()
