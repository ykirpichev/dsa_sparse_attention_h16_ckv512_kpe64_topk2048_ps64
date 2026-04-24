"""
FlashInfer-Bench Modal Cloud Benchmark Runner.

Automatically packs the solution from source files and runs benchmarks
on NVIDIA B200 GPUs via Modal.

Setup (one-time):
    modal setup
    modal volume create flashinfer-trace
    modal volume put flashinfer-trace /path/to/flashinfer-trace/

Usage:
    # Full suite (contest-style), all workloads:
    modal run scripts/run_modal.py

    # Smoke (fast correctness check): 10 random workloads, warmup=1 / iters=1 / trials=1:
    modal run scripts/run_modal.py --smoke

    # Smoke with custom sample size:
    modal run scripts/run_modal.py --smoke --smoke-workloads 5
"""

import random
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal
from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet

SMOKE_WORKLOAD_COUNT_DEFAULT = 10
SMOKE_RANDOM_SEED = 42

app = modal.App("flashinfer-bench")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

# Pin to the same image digest documented in EVALUATION.md (evaluator environment).
# CUDA 13.2 + nvcc; required to compile the TVM-FFI CUDA entry at runtime.
image = (
    modal.Image.from_registry("flashinfer/flashinfer-ci-cu132:20260401-2c675fb")
    .apt_install("git")
    # sm_100a enables Blackwell-only features (tcgen05, TMA tensor ops, ...).
    # Without the trailing "a", nvcc targets the portable sm_100 subset and
    # the tcgen05.mma / tcgen05.ld / tcgen05.alloc intrinsics fail to
    # assemble. The BuildSpec's extra_cuda_cflags are not honored by the
    # tvm_ffi.cpp.build() caller path, so we inject via TVM_FFI_CUDA_ARCH_LIST.
    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "TVM_FFI_CUDA_ARCH_LIST": "10.0a",
        }
    )
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


@app.function(image=image, gpu="B200:1", timeout=3600, volumes={TRACE_SET_PATH: trace_volume})
def run_benchmark(
    solution: Solution,
    config: BenchmarkConfig = None,
    smoke: bool = False,
    smoke_workloads: int = SMOKE_WORKLOAD_COUNT_DEFAULT,
    smoke_seed: int = SMOKE_RANDOM_SEED,
    tma: bool = False,
    ws: bool = False,
    ws3: bool = False,
    split_g16: str = "auto",  # "on" / "off" / "auto" (kernel default)
    split_diag: bool = False,
    fused: bool | None = None,
    cta_per_t: bool = False,
    cta_per_t_regs: int = 1,  # 1 (safe, max regs) or 2 (2 CTAs/SM)
    cta_per_t_th512: bool = False,  # 512-thread, 1-warp-per-head variant (7a.1)
    cta_per_t_mma: bool = False,    # WMMA logit GEMM variant (Phase 7c)
    cta_per_t_mma_v: bool = False,  # WMMA logit + WMMA V-accum (Phase 7c-next)
    cta_per_t_min_t: int = 0,       # T-threshold dispatch (Phase 7c-next)
    cta_per_t_min_kv: int = 0,      # num_kv_rows-threshold dispatch (Phase 7g)
    cta_per_t_density: str = "auto",  # "on" / "off" / "auto" (Phase 7h)
    cta_per_t_density_min_avg: int = -1,  # avg-valid-K threshold (Phase 7h)
) -> dict:
    """Run benchmark on Modal B200 and return results.

    If ``smoke`` is True, randomly sample ``smoke_workloads`` workloads (seeded)
    and use warmup=1 / iterations=1 / num_trials=1 for a fast correctness pass.

    If ``tma`` is True, sets MLA_SPARSE_ATTN_TMA=1 before the solution's
    kernel.cu is imported, routing the G==1 path through the TMA-gather4
    variant (Phase 5 step 2).

    If ``ws`` is True, sets MLA_SPARSE_ATTN_WS=1 which routes the G==1 path
    through the triple-buffer WS variant (Phase 5c). Implies TMA-style
    descriptor usage; the two env vars are independent but ``ws`` takes
    precedence when both are set.

    If ``ws3`` is True, sets MLA_SPARSE_ATTN_WS3=1 which routes the G==1 path
    through the full warp-specialized variant (Phase 5c Stage 3 — dedicated
    DMA warp, setmaxnreg, named barriers, V-accum 128 thr × 4 bf16). Takes
    precedence over both ``ws`` and ``tma`` in the launcher.
    """
    import os as _os
    if tma:
        _os.environ["MLA_SPARSE_ATTN_TMA"] = "1"
        print("  TMA enabled (MLA_SPARSE_ATTN_TMA=1)")
    if ws:
        _os.environ["MLA_SPARSE_ATTN_WS"] = "1"
        print("  WS enabled (MLA_SPARSE_ATTN_WS=1)")
    if ws3:
        _os.environ["MLA_SPARSE_ATTN_WS3"] = "1"
        print("  WS3 enabled (MLA_SPARSE_ATTN_WS3=1)")
    if split_g16 in ("on", "off"):
        _os.environ["MLA_SPARSE_ATTN_SPLIT_G16"] = "1" if split_g16 == "on" else "0"
        print(f"  Split-K G=16 {'ENABLED' if split_g16 == 'on' else 'DISABLED'} "
              f"(MLA_SPARSE_ATTN_SPLIT_G16={'1' if split_g16 == 'on' else '0'})")
    elif split_g16 != "auto":
        raise ValueError(f"split_g16 must be 'on', 'off', or 'auto', got {split_g16!r}")
    if split_diag:
        _os.environ["MLA_SPARSE_ATTN_SPLIT_DIAG"] = "1"
        print("  Split-K diag enabled (MLA_SPARSE_ATTN_SPLIT_DIAG=1)")
    if fused is not None:
        _os.environ["MLA_SPARSE_ATTN_FUSED"] = "1" if fused else "0"
        print(f"  Fused partial+reduce {'ENABLED' if fused else 'DISABLED'} "
              f"(MLA_SPARSE_ATTN_FUSED={'1' if fused else '0'})")
    if cta_per_t:
        _os.environ["MLA_SPARSE_ATTN_CTA_PER_T"] = "1"
        _os.environ["MLA_SPARSE_ATTN_CTA_PER_T_REGS"] = str(cta_per_t_regs)
        if cta_per_t_min_t > 0:
            _os.environ["MLA_SPARSE_ATTN_CTA_PER_T_MIN_T"] = str(cta_per_t_min_t)
        if cta_per_t_min_kv > 0:
            _os.environ["MLA_SPARSE_ATTN_CTA_PER_T_MIN_KV"] = str(cta_per_t_min_kv)
        if cta_per_t_density in ("on", "off"):
            _os.environ["MLA_SPARSE_ATTN_CTA_PER_T_DENSITY"] = (
                "1" if cta_per_t_density == "on" else "0"
            )
        elif cta_per_t_density != "auto":
            raise ValueError(
                f"cta_per_t_density must be 'on', 'off', or 'auto', "
                f"got {cta_per_t_density!r}"
            )
        if cta_per_t_density_min_avg >= 0:
            _os.environ["MLA_SPARSE_ATTN_CTA_PER_T_DENSITY_MIN_AVG"] = str(
                cta_per_t_density_min_avg
            )
        min_t_tag = (
            (f", min_t={cta_per_t_min_t}" if cta_per_t_min_t > 0 else "")
          + (f", min_kv={cta_per_t_min_kv}" if cta_per_t_min_kv > 0 else "")
          + (f", density={cta_per_t_density}" if cta_per_t_density != "auto" else "")
          + (f", density_min_avg={cta_per_t_density_min_avg}"
             if cta_per_t_density_min_avg >= 0 else "")
        )
        if cta_per_t_mma_v:
            _os.environ["MLA_SPARSE_ATTN_CTA_PER_T_MMA_V"] = "1"
            # mma_v implies mma (cascading precedence on the kernel side).
            _os.environ["MLA_SPARSE_ATTN_CTA_PER_T_MMA"] = "1"
            print(f"  CTA-per-t ENABLED [mma_v variant] (regs variant={cta_per_t_regs}{min_t_tag})")
        elif cta_per_t_mma:
            _os.environ["MLA_SPARSE_ATTN_CTA_PER_T_MMA"] = "1"
            print(f"  CTA-per-t ENABLED [mma variant] (regs variant={cta_per_t_regs}{min_t_tag})")
        elif cta_per_t_th512:
            _os.environ["MLA_SPARSE_ATTN_CTA_PER_T_TH512"] = "1"
            print(f"  CTA-per-t ENABLED [th512 variant] (regs variant={cta_per_t_regs}{min_t_tag})")
        else:
            print(f"  CTA-per-t ENABLED (regs variant={cta_per_t_regs}{min_t_tag})")

    if config is None:
        if smoke:
            config = BenchmarkConfig(warmup_runs=1, iterations=1, num_trials=1)
        else:
            config = BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5)

    trace_set = TraceSet.from_path(TRACE_SET_PATH)

    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])

    if smoke and workloads and len(workloads) > smoke_workloads:
        rng = random.Random(smoke_seed)
        workloads = rng.sample(workloads, smoke_workloads)
        print(
            f"  Smoke mode: {len(workloads)} workloads (seed={smoke_seed}), "
            f"warmup={config.warmup_runs}, iterations={config.iterations}, "
            f"num_trials={config.num_trials}"
        )

    if not workloads:
        raise ValueError(f"No workloads found for definition '{solution.definition}'")

    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )

    benchmark = Benchmark(bench_trace_set, config)
    result_trace_set = benchmark.run_all(dump_traces=True)

    traces = result_trace_set.traces.get(definition.name, [])
    results = {definition.name: {}}

    for trace in traces:
        if trace.evaluation:
            entry = {
                "status": trace.evaluation.status.value,
                "solution": trace.solution,
            }
            if trace.evaluation.performance:
                entry["latency_ms"] = trace.evaluation.performance.latency_ms
                entry["reference_latency_ms"] = trace.evaluation.performance.reference_latency_ms
                entry["speedup_factor"] = trace.evaluation.performance.speedup_factor
            if trace.evaluation.correctness:
                entry["max_abs_error"] = trace.evaluation.correctness.max_absolute_error
                entry["max_rel_error"] = trace.evaluation.correctness.max_relative_error
            # Capture any potentially useful debug fields
            for attr in ("error_message", "error", "message", "detail", "log", "stderr"):
                val = getattr(trace.evaluation, attr, None)
                if val:
                    entry["error_message"] = str(val)
                    break
            try:
                entry["evaluation_repr"] = repr(trace.evaluation)[:1500]
            except Exception:
                pass
            results[definition.name][trace.workload.uuid] = entry

    return results


def print_results(results: dict):
    """Print benchmark results in a formatted way."""
    for def_name, traces in results.items():
        print(f"\n{def_name}:")
        for workload_uuid, result in traces.items():
            status = result.get("status")
            print(f"  Workload {workload_uuid[:8]}...: {status}", end="")

            if result.get("latency_ms") is not None:
                print(f" | {result['latency_ms']:.3f} ms", end="")

            if result.get("speedup_factor") is not None:
                print(f" | {result['speedup_factor']:.2f}x speedup", end="")

            if result.get("max_abs_error") is not None:
                abs_err = result["max_abs_error"]
                rel_err = result.get("max_rel_error", 0)
                print(f" | abs_err={abs_err:.2e}, rel_err={rel_err:.2e}", end="")

            print()
            if result.get("error_message"):
                msg = result["error_message"]
                print("    error:", msg[:3000])
            elif result.get("status") not in (None, "PASSED", "PASS", "passed"):
                er = result.get("evaluation_repr")
                if er:
                    print("    eval:", er)


@app.local_entrypoint()
def main(
    smoke: bool = False,
    smoke_workloads: int = SMOKE_WORKLOAD_COUNT_DEFAULT,
    smoke_seed: int = SMOKE_RANDOM_SEED,
    tma: bool = False,
    ws: bool = False,
    ws3: bool = False,
    split_g16: str = "auto",
    split_diag: bool = False,
    fused: bool | None = None,
    cta_per_t: bool = False,
    cta_per_t_regs: int = 1,
    cta_per_t_th512: bool = False,
    cta_per_t_mma: bool = False,
    cta_per_t_mma_v: bool = False,
    cta_per_t_min_t: int = 0,
    cta_per_t_min_kv: int = 0,
    cta_per_t_density: str = "auto",
    cta_per_t_density_min_avg: int = -1,
):
    """Pack solution and run benchmark on Modal.

    Args:
        smoke: If True, run on a random subset with minimal timing (fast correctness check).
        smoke_workloads: How many random workloads to run in smoke mode (default 10).
        smoke_seed: Random seed for the smoke workload sample (default 42).
        tma: If True, sets MLA_SPARSE_ATTN_TMA=1 on the Modal worker before the
            solution's kernel.cu is imported (Phase 5 step 2 gather4 path).
        ws: If True, sets MLA_SPARSE_ATTN_WS=1 on the Modal worker (Phase 5c
            triple-buffered TMA variant).
        ws3: If True, sets MLA_SPARSE_ATTN_WS3=1 on the Modal worker (Phase 5c
            Stage 3 full warp-specialized variant).
    """
    from scripts.pack_solution import pack_solution

    print("Packing solution from source files...")
    solution_path = pack_solution()

    print("\nLoading solution...")
    solution = Solution.model_validate_json(solution_path.read_text())
    print(f"Loaded: {solution.name} ({solution.definition})")

    tag = ""
    if cta_per_t:
        if cta_per_t_mma_v:
            var = "mma_v"
        elif cta_per_t_mma:
            var = "mma"
        elif cta_per_t_th512:
            var = "th512"
        else:
            var = "scalar"
        min_t_tag = (
            (f" min_t={cta_per_t_min_t}" if cta_per_t_min_t > 0 else "")
          + (f" min_kv={cta_per_t_min_kv}" if cta_per_t_min_kv > 0 else "")
          + (f" density={cta_per_t_density}" if cta_per_t_density != "auto" else "")
          + (f" min_avg={cta_per_t_density_min_avg}"
             if cta_per_t_density_min_avg >= 0 else "")
        )
        tag = f" [CPT=1 {var} regs={cta_per_t_regs}{min_t_tag}]"
    elif ws3:
        tag = " [WS3=1]"
    elif ws:
        tag = " [WS=1]"
    elif tma:
        tag = " [TMA=1]"
    if smoke:
        print(f"\nRunning SMOKE benchmark on Modal B200 "
              f"({smoke_workloads} random workloads, seed={smoke_seed})"
              + tag + "...")
    else:
        print("\nRunning FULL benchmark on Modal B200" + tag + "...")
    results = run_benchmark.remote(
        solution,
        smoke=smoke,
        smoke_workloads=smoke_workloads,
        smoke_seed=smoke_seed,
        tma=tma,
        ws=ws,
        ws3=ws3,
        split_g16=split_g16,
        split_diag=split_diag,
        fused=fused,
        cta_per_t=cta_per_t,
        cta_per_t_regs=cta_per_t_regs,
        cta_per_t_th512=cta_per_t_th512,
        cta_per_t_mma=cta_per_t_mma,
        cta_per_t_mma_v=cta_per_t_mma_v,
        cta_per_t_min_t=cta_per_t_min_t,
        cta_per_t_min_kv=cta_per_t_min_kv,
        cta_per_t_density=cta_per_t_density,
        cta_per_t_density_min_avg=cta_per_t_density_min_avg,
    )

    if not results:
        print("No results returned!")
        return

    print_results(results)
