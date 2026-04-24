"""Direct kernel probe for debugging the CUDA-graph capture path.

Loads the evaluator trace's first workload, calls `solution.run` N times
with the same (cloned, per-iter) inputs, and surfaces any Python exception
with a full traceback. Unlike ``run_modal.py`` this bypasses
flashinfer-bench's persistent-worker stdio redirection so all fprintf
output from the extension lands in the parent Modal log.

Usage:
    modal run scripts/run_modal_cg_probe.py
    modal run scripts/run_modal_cg_probe.py --graph-cache 0
    modal run scripts/run_modal_cg_probe.py --graph-debug 1 --n-iters 5
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal
from flashinfer_bench import Solution, TraceSet

app = modal.App("flashinfer-bench-cg-probe")
trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

image = (
    modal.Image.from_registry("flashinfer/flashinfer-ci-cu132:20260401-2c675fb")
    .apt_install("git")
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .pip_install("wheel", "setuptools")
    .run_commands(
        "git clone --recursive --depth 1 https://github.com/deepseek-ai/DeepGEMM.git /tmp/DeepGEMM",
        "pip install --no-build-isolation /tmp/DeepGEMM",
        "git clone --recursive --depth 1 https://github.com/flashinfer-ai/flashinfer.git /tmp/flashinfer",
        "pip install --no-build-isolation /tmp/flashinfer",
        "git clone --depth 1 https://github.com/flashinfer-ai/flashinfer-bench.git /tmp/flashinfer-bench",
        "pip install /tmp/flashinfer-bench",
    )
)


@app.function(image=image, gpu="B200:1", timeout=1200,
              volumes={TRACE_SET_PATH: trace_volume})
def probe_kernel(solution_json: str,
                 n_iters: int = 3,
                 graph_cache: str = "1",
                 graph_debug: str = "1",
                 workload_index: int = 0) -> dict:
    """Build the solution and run it n_iters times on a single workload."""
    import sys
    import traceback

    os.environ["WOMBAT_CUDA_GRAPH"] = graph_cache
    os.environ["WOMBAT_CUDA_GRAPH_DEBUG"] = graph_debug

    import torch
    from flashinfer_bench import Solution, TraceSet
    from flashinfer_bench.compile import BuilderRegistry
    from flashinfer_bench.bench.utils import gen_inputs, load_safetensors
    from flashinfer_bench.bench.evaluators.utils import allocate_outputs

    solution = Solution.model_validate_json(solution_json)
    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])
    wl_entry = workloads[workload_index]
    wl = wl_entry.workload if hasattr(wl_entry, "workload") else wl_entry

    safe = load_safetensors(definition, wl, trace_set.root)
    raw_inputs = gen_inputs(definition, wl, "cuda", safe)

    registry = BuilderRegistry.get_instance()
    runnable = registry.build(definition, solution)

    # Allocate output tensors. For DPS style the runnable appends outputs
    # automatically — we emulate what flashinfer-bench does in
    # time_runnable: clone args per iteration.
    def cloned(seq):
        return [a.clone() if isinstance(a, torch.Tensor) else a for a in seq]

    print(f"[probe] Running {n_iters} iterations on workload "
          f"{wl.uuid[:8]} (graph_cache={graph_cache}, debug={graph_debug})",
          flush=True)

    is_dps = runnable.metadata.destination_passing_style
    print(f"[probe] destination_passing_style={is_dps}", flush=True)

    results = {"per_iter_ok": [], "errors": []}
    for i in range(n_iters):
        inp = cloned(raw_inputs)
        try:
            if is_dps:
                out_tensors = allocate_outputs(definition, inp, "cuda")
                runnable(*inp, *out_tensors)
            else:
                runnable(*inp)
            torch.cuda.synchronize()
            results["per_iter_ok"].append(True)
            print(f"[probe] iter {i}: OK", flush=True)
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc()
            results["per_iter_ok"].append(False)
            results["errors"].append({
                "iter": i, "type": type(e).__name__, "msg": str(e), "tb": tb,
            })
            print(f"[probe] iter {i}: ERROR\n{tb}", flush=True)
            # Stop after first failure so the traceback is fresh.
            break

    # Flush stderr from the extension (our fprintf diagnostics).
    sys.stderr.flush()
    sys.stdout.flush()
    return results


@app.local_entrypoint()
def main(graph_cache: str = "1", graph_debug: str = "1", n_iters: int = 3,
         workload_index: int = 0):
    from scripts.pack_solution import pack_solution
    solution_path = pack_solution()
    solution_json = solution_path.read_text()
    results = probe_kernel.remote(
        solution_json,
        n_iters=n_iters,
        graph_cache=graph_cache,
        graph_debug=graph_debug,
        workload_index=workload_index,
    )
    print("\n=== PROBE RESULTS ===")
    print(f"per_iter_ok = {results['per_iter_ok']}")
    for err in results["errors"]:
        print(f"\n[iter {err['iter']}] {err['type']}: {err['msg']}")
        print(err["tb"])
