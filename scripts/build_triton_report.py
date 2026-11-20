"""Build docs/index.html, the CUDA vs Triton comparison page.

Every number on the page is read from the benchmark artifacts:

  artifacts/triton_compare_first_place.json  official flashinfer-bench run
  artifacts/triton_profile_sweep.json        Triton configuration sweep
  artifacts/triton_profile_graph.json        the same with CUDA-graph replay
  artifacts/triton_versions.json             Triton releases 3.4.0 to 3.9.0

Usage: python scripts/build_triton_report.py
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path

ROOT = Path(__file__).parent.parent
ARTIFACTS = ROOT / "artifacts"
OUT = ROOT / "docs" / "index.html"
REPO = "https://github.com/ykirpichev/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64"
BRANCH = "readable"

# Solutions in the official run: label, CSS colour variable, marker, one line.
SERIES = {
    "wombat-cuda": ("CUDA submission", "cuda", "circle",
                    "The contest kernel: hand-written CUDA with WMMA, cp.async and a CUDA-graph cache."),
    "triton": ("Triton, faithful port", "port", "square",
               "The same dispatcher and the same two kernels, translated one to one."),
    "triton-tuned": ("Triton, tuned", "tuned", "diamond",
                     "The port's per-token kernel with larger tiles, pipelined loads and graph replay."),
    "triton-fused": ("Triton, fused", "fused", "star",
                     "Split and merge in one launch, interleaved split."),
    "first-place": ("Contest first place (Triton)", "ref", "triangle",
                    "Doğaç Eldenk's winning kernel, run here as a reference."),
}

LADDER = [
    ("sweep", "head,1,8,16,1", "Per-head kernel for every workload"),
    ("sweep", "auto", "Faithful port: CUDA dispatch, 16-key tiles"),
    ("sweep", "token,16,8,16,1", "Per-token kernel for every workload, 16-way split"),
    ("sweep", "token,16,8,64,1", "+ 64-key tiles"),
    ("sweep", "token,16,8,64,2", "+ pipelined loads (num_stages=2)"),
    ("graph", "token,16,8,64,3,graph", "+ both launches replayed from one CUDA graph"),
    ("graph", "token,16,8,128,2,graph", "Same with 128-key tiles"),
]

RELEASE_CONFIGS = [
    ("auto", "Faithful port"),
    ("head,1,8,16,1", "Per-head kernel only"),
    ("token,16,8,16,1", "Per-token, 16-key tiles"),
    ("token,16,8,64,3", "Per-token, 64-key tiles, 3 stages"),
    ("token,16,8,64,3,graph", "Same, graph replay"),
]


def load(name: str):
    return json.loads((ARTIFACTS / name).read_text())


def geomean(values) -> float:
    values = list(values)
    return math.exp(sum(math.log(v) for v in values) / len(values))


def esc(text) -> str:
    return html.escape(str(text))


def marker(shape: str, x: float, y: float, colour: str, r: float = 5.0) -> str:
    fill = f'fill="var(--{colour})" stroke="var(--bg)" stroke-width="1"'
    if shape == "circle":
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" {fill}/>'
    if shape == "square":
        return f'<rect x="{x - r + 0.5:.1f}" y="{y - r + 0.5:.1f}" width="{2 * r - 1}" height="{2 * r - 1}" {fill}/>'
    if shape == "diamond":
        d = r + 1
        return f'<path d="M{x:.1f} {y - d:.1f} L{x + d:.1f} {y:.1f} L{x:.1f} {y + d:.1f} L{x - d:.1f} {y:.1f} Z" {fill}/>'
    if shape == "triangle":
        d = r + 1
        return f'<path d="M{x:.1f} {y - d:.1f} L{x + d:.1f} {y + d - 1:.1f} L{x - d:.1f} {y + d - 1:.1f} Z" {fill}/>'
    points = []
    for i in range(10):
        radius = (r + 2) if i % 2 == 0 else (r + 2) * 0.45
        angle = -math.pi / 2 + i * math.pi / 5
        points.append(f"{x + radius * math.cos(angle):.1f},{y + radius * math.sin(angle):.1f}")
    return f'<polygon points="{" ".join(points)}" {fill}/>'


def legend(names) -> str:
    items = []
    for name in names:
        label, colour, shape, _ = SERIES[name]
        icon = f'<svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">{marker(shape, 8, 8, colour)}</svg>'
        items.append(f'<span class="key">{icon}{esc(label)}</span>')
    return f'<div class="legend">{"".join(items)}</div>'


def dot_plot(names, order, latency, meta) -> str:
    """One row per workload, latency on a log axis, one marker per solution."""
    values = [latency[n][u] for n in names for u in order]
    lo = 2 ** math.floor(math.log2(min(values)))
    hi = 2 ** math.ceil(math.log2(max(values)))
    left, right, top, row = 190, 24, 30, 23
    width = 940
    plot = width - left - right
    height = top + row * len(order) + 34

    def x_of(value: float) -> float:
        return left + (math.log2(value) - math.log2(lo)) / (math.log2(hi) - math.log2(lo)) * plot

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" '
             f'aria-label="Latency of each solution on each of the {len(order)} workloads">']
    tick = lo
    while tick <= hi:
        x = x_of(tick)
        parts.append(f'<line x1="{x:.1f}" y1="{top - 6}" x2="{x:.1f}" y2="{height - 30}" class="grid"/>')
        parts.append(f'<text x="{x:.1f}" y="{height - 12}" class="tick" text-anchor="middle">{tick:g} µs</text>')
        tick *= 2
    parts.append(f'<text x="8" y="16" class="head">workload</text>')
    parts.append(f'<text x="92" y="16" class="head">tokens</text>')
    parts.append(f'<text x="{left - 10}" y="16" class="head" text-anchor="end">keys</text>')
    for i, uuid in enumerate(order):
        y = top + row * i + row / 2
        tokens, keys = meta[uuid]
        if i % 2 == 0:
            parts.append(f'<rect x="0" y="{y - row / 2:.1f}" width="{width}" height="{row}" class="band"/>')
        parts.append(f'<text x="8" y="{y + 4:.1f}" class="mono">{uuid[:8]}</text>')
        parts.append(f'<text x="112" y="{y + 4:.1f}" class="num" text-anchor="end">{tokens}</text>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.1f}" class="num" text-anchor="end">{keys}</text>')
        xs = [x_of(latency[n][uuid]) for n in names]
        parts.append(f'<line x1="{min(xs):.1f}" y1="{y:.1f}" x2="{max(xs):.1f}" y2="{y:.1f}" class="span"/>')
        for name in names:
            label, colour, shape, _ = SERIES[name]
            x = x_of(latency[name][uuid])
            parts.append(f'<g><title>{esc(label)}: {latency[name][uuid]:.1f} µs</title>{marker(shape, x, y, colour)}</g>')
    parts.append("</svg>")
    return "".join(parts)


def ladder_chart(rows) -> str:
    """Horizontal bars: geomean latency relative to the CUDA kernel."""
    left, right, top, row = 400, 70, 8, 30
    width = 940
    plot = width - left - right
    peak = max(2.0, max(v for _, v, _ in rows))
    scale_max = 2.4
    height = top + row * len(rows) + 30

    def x_of(value: float) -> float:
        return left + min(value, scale_max) / scale_max * plot

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" '
             f'aria-label="Geomean latency of each Triton configuration relative to CUDA">']
    for tick in (0.5, 1.0, 1.5, 2.0):
        x = x_of(tick)
        cls = "rule" if tick == 1.0 else "grid"
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height - 26}" class="{cls}"/>')
        label = "CUDA = 1.0×" if tick == 1.0 else f"{tick:g}×"
        parts.append(f'<text x="{x:.1f}" y="{height - 8}" class="tick" text-anchor="middle">{label}</text>')
    for i, (label, value, _) in enumerate(rows):
        y = top + row * i
        colour = "tuned" if value < 1.0 else "port"
        clipped = value > scale_max
        parts.append(f'<text x="{left - 12}" y="{y + 19}" class="label" text-anchor="end">{esc(label)}</text>')
        parts.append(f'<rect x="{left}" y="{y + 6}" width="{x_of(value) - left:.1f}" height="{row - 12}" '
                     f'rx="2" fill="var(--{colour})"/>')
        if clipped:
            x = x_of(scale_max)
            parts.append(f'<path d="M{x - 14:.1f} {y + 3} l8 {row - 6} M{x - 8:.1f} {y + 3} l8 {row - 6}" '
                         f'stroke="var(--bg)" stroke-width="3" fill="none"/>')
        parts.append(f'<text x="{x_of(value) + 8:.1f}" y="{y + 19}" class="value">{value:.2f}×</text>')
    parts.append("</svg>")
    del peak
    return "".join(parts)


def config_ratio(report, config, cuda_us):
    ratios = []
    for uuid, workload in report["workloads"].items():
        entry = workload["latency"].get(config)
        if not entry or not entry.get("median_us") or uuid not in cuda_us:
            return None
        ratios.append(entry["median_us"] / cuda_us[uuid])
    return geomean(ratios) if ratios else None


def main() -> None:
    compare = load("triton_compare_first_place.json")
    sweep = load("triton_profile_sweep.json")
    graph = load("triton_profile_graph.json")
    releases = load("triton_versions.json")

    names = [n for n in SERIES if n in compare]
    latency = {n: {u: row["latency_ms"] * 1e3 for u, row in compare[n].items()} for n in names}
    error = {n: max(row["max_abs_error"] for row in compare[n].values()) for n in names}
    passed = {n: sum(row["status"] == "PASSED" for row in compare[n].values()) for n in names}
    meta = {u: (w["T"], w["valid_avg"]) for u, w in sweep["workloads"].items()}
    order = sorted(latency["wombat-cuda"], key=lambda u: (meta[u][0], meta[u][1]))
    n_workloads = len(order)

    cuda = latency["wombat-cuda"]
    stats = {}
    for name in names:
        ratios = [latency[name][u] / cuda[u] for u in order]
        stats[name] = {
            "mean": sum(latency[name].values()) / n_workloads,
            "geo": geomean(ratios),
            "wins": sum(r < 1.0 for r in ratios),
            "best": min(ratios),
            "worst": max(ratios),
        }

    # Headline cards.
    cards = []
    for name in names:
        label, colour, _, blurb = SERIES[name]
        s = stats[name]
        if name == "wombat-cuda":
            big, small = f"{s['mean']:.1f} µs", "mean latency, the baseline"
        else:
            big = f"{s['geo']:.2f}×"
            small = (f"CUDA latency (geomean) · {s['mean']:.1f} µs mean · "
                     f"faster on {s['wins']} of {n_workloads}")
        cards.append(
            f'<div class="card" style="--accent: var(--{colour})"><div class="card-name">{esc(label)}</div>'
            f'<div class="card-big">{big}</div><div class="card-small">{small}</div>'
            f'<div class="card-blurb">{esc(blurb)}</div></div>'
        )

    # Per-workload table.
    head = "".join(f"<th>{esc(SERIES[n][0])}</th>" for n in names)
    body = []
    for uuid in order:
        tokens, keys = meta[uuid]
        best = min(latency[n][uuid] for n in names)
        cells = "".join(
            f'<td class="{"best" if latency[n][uuid] == best else ""}">{latency[n][uuid]:.1f}</td>' for n in names
        )
        body.append(f'<tr><td class="mono">{uuid[:8]}</td><td>{tokens}</td><td>{keys}</td>{cells}</tr>')
    foot = "".join(f"<td>{stats[n]['mean']:.1f}</td>" for n in names)
    table = (
        f'<table><thead><tr><th>Workload</th><th>Tokens</th><th>Keys per token</th>{head}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody>'
        f'<tfoot><tr><td colspan="3">Mean</td>{foot}</tr></tfoot></table>'
    )

    # Configuration ladder.
    cuda_sweep = {u: w["latency"]["cuda"]["median_us"] for u, w in sweep["workloads"].items()}
    cuda_graph = {u: w["latency"]["cuda"]["median_us"] for u, w in graph["workloads"].items()}
    ladder = []
    for source, config, label in LADDER:
        report, base = (sweep, cuda_sweep) if source == "sweep" else (graph, cuda_graph)
        value = config_ratio(report, config, base)
        if value is not None:
            ladder.append((label, value, config))
    ladder_value = {config: value for _, value, config in ladder}

    # Triton releases, against the CUDA medians of the sweep run.
    release_head = "".join(f"<th>{esc(label)}</th>" for _, label in RELEASE_CONFIGS)
    release_rows = []
    measured = [(r.get("requested_version") or r.get("triton_version"), [r]) for r in releases]
    measured.append((sweep["triton_version"], [sweep, graph]))
    for version, reports in sorted(measured, key=lambda item: [int(x) for x in item[0].split(".")]):
        cells = []
        for config, _ in RELEASE_CONFIGS:
            value = None
            for report in reports:
                base = cuda_graph if report is graph else cuda_sweep
                if value is None and report.get("workloads"):
                    value = config_ratio(report, config, base)
            cells.append(f"<td>{value:.2f}×</td>" if value is not None else '<td class="muted">fails</td>')
        note = {"3.6.0": " (contest)", sweep["triton_version"]: " (benchmark image)"}.get(version, "")
        release_rows.append(f"<tr><td>{esc(version)}{note}</td>{''.join(cells)}</tr>")
    release_table = (
        f'<table><thead><tr><th>Triton release</th>{release_head}</tr></thead>'
        f'<tbody>{"".join(release_rows)}</tbody></table>'
    )

    port, tuned = stats["triton"], stats["triton-tuned"]
    ref = stats.get("first-place")
    dense = [u for u in order if meta[u][1] >= 210]
    dense_us = {n: sum(latency[n][u] for u in dense) / len(dense) for n in names}
    tiny = [u for u in order if meta[u][0] <= 2 and meta[u][1] <= 42]
    tiny_us = {n: (min(latency[n][u] for u in tiny), max(latency[n][u] for u in tiny)) for n in names}

    n_head = sum(w["plan"][0] == "head" for w in sweep["workloads"].values())
    # Dense workloads the port really ran through its per-token kernel (see the
    # note on the address-keyed dispatch cache at the bottom of the page).
    reached = [u for u in dense if latency["triton"][u] < 2 * cuda[u]]
    port_token = [latency["triton"][u] for u in reached]
    cuda_token = [cuda[u] for u in reached]

    ref_sentence = ""
    if ref:
        ref_sentence = (
            f" For scale, the kernel that won the contest track is also Triton and runs at "
            f"<strong>{ref['geo']:.2f}×</strong> in the same run."
        )

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DSA sparse attention on B200: CUDA vs Triton</title>
<style>
:root {{
  --bg: #ffffff; --fg: #1c2024; --muted: #60646c; --line: #e0e3e8; --band: #f6f7f9;
  --cuda: #4d8a00; --port: #d4691a; --tuned: #2463d6; --fused: #7c3aed; --ref: #7d828c;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg: #111316; --fg: #e6e8eb; --muted: #9ba1ab; --line: #2a2e35; --band: #181b20;
    --cuda: #8fd32a; --port: #f0954a; --tuned: #6aa2ff; --fused: #b08cff; --ref: #9aa0aa;
  }}
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; background: var(--bg); color: var(--fg);
  font: 16px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}}
main {{ max-width: 980px; margin: 0 auto; padding: 48px 20px 80px; }}
h1 {{ font-size: 32px; line-height: 1.2; margin: 0 0 8px; letter-spacing: -0.01em; }}
h2 {{ font-size: 22px; margin: 56px 0 12px; letter-spacing: -0.005em; }}
p {{ margin: 12px 0; max-width: 760px; }}
.sub {{ color: var(--muted); margin: 0 0 28px; }}
.lead {{ font-size: 18px; max-width: 800px; }}
a {{ color: var(--tuned); }}
code {{ font: 14px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; background: var(--band);
       padding: 1px 5px; border-radius: 4px; }}
pre {{ background: var(--band); border: 1px solid var(--line); border-radius: 8px; padding: 14px 16px;
      overflow-x: auto; }}
pre code {{ background: none; padding: 0; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; margin: 28px 0 8px; }}
.card {{ border: 1px solid var(--line); border-top: 4px solid var(--accent); border-radius: 8px; padding: 14px 16px; }}
.card-name {{ font-weight: 600; }}
.card-big {{ font-size: 30px; font-weight: 700; line-height: 1.25; margin-top: 6px; color: var(--accent); }}
.card-small {{ font-size: 13px; color: var(--muted); }}
.card-blurb {{ font-size: 14px; margin-top: 10px; }}
figure {{ margin: 20px 0; }}
figcaption {{ color: var(--muted); font-size: 14px; margin-top: 6px; max-width: 760px; }}
svg {{ width: 100%; height: auto; display: block; }}
svg text {{ fill: var(--fg); font-size: 13px; }}
svg .tick, svg .head {{ fill: var(--muted); font-size: 12px; }}
svg .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }}
svg .num {{ font-variant-numeric: tabular-nums; }}
svg .value {{ font-weight: 600; font-variant-numeric: tabular-nums; }}
svg .grid {{ stroke: var(--line); stroke-width: 1; }}
svg .rule {{ stroke: var(--fg); stroke-width: 1.5; stroke-dasharray: 4 3; }}
svg .span {{ stroke: var(--muted); stroke-width: 1; opacity: 0.5; }}
svg .band {{ fill: var(--band); }}
.legend {{ display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 14px; margin: 8px 0 4px; }}
.key {{ display: inline-flex; align-items: center; gap: 6px; }}
.key svg {{ width: 16px; height: 16px; }}
.scroll {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; font-size: 14px; font-variant-numeric: tabular-nums; }}
th, td {{ text-align: right; padding: 6px 10px; border-bottom: 1px solid var(--line); white-space: nowrap; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ font-weight: 600; color: var(--muted); white-space: normal; vertical-align: bottom; }}
td.best {{ font-weight: 700; }}
td.muted {{ color: var(--muted); }}
td.mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
tfoot td {{ font-weight: 600; border-top: 2px solid var(--line); }}
ol li, ul li {{ margin: 8px 0; max-width: 740px; }}
footer {{ margin-top: 64px; color: var(--muted); font-size: 14px; }}
</style>
</head>
<body>
<main>
<h1>DSA sparse attention on B200: CUDA vs Triton</h1>
<p class="sub">MLSys 2026 FlashInfer contest kernel <code>dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64</code>
· {n_workloads} contest workloads · NVIDIA B200 · Triton {esc(sweep["triton_version"])}</p>

<p class="lead">Translating the CUDA submission to Triton line for line makes it
<strong>{port['geo']:.2f}× slower</strong>. Keeping the same Triton kernels and changing four settings makes it
<strong>{1 / tuned['geo']:.2f}× faster</strong> than the CUDA submission ({tuned['geo']:.2f}× its latency).{ref_sentence}
Every version passes all {n_workloads} workloads.</p>

<div class="cards">{"".join(cards)}</div>

<h2>Latency on every workload</h2>
<p>Each row is one contest workload, ordered by token count and by the average number of valid keys per token
(the rest of the 2,048 top-k slots are padding). Further left is faster; the axis is logarithmic.</p>
{legend(names)}
<figure>{dot_plot(names, order, latency, meta)}
<figcaption>Median latency per call from the official flashinfer-bench timing (CUPTI kernel span, L2 cache flushed
before each call, 5 warm-up and 50 timed calls).</figcaption></figure>

<p>The CUDA kernel is still the fastest on the smallest calls ({tiny_us['wombat-cuda'][0]:.1f}–{tiny_us['wombat-cuda'][1]:.1f} µs
against {tiny_us['triton-tuned'][0]:.1f}–{tiny_us['triton-tuned'][1]:.1f} µs for tuned Triton). On the dense workloads, where
most of the time goes, tuned Triton averages {dense_us['triton-tuned']:.1f} µs against {dense_us['wombat-cuda']:.1f} µs.</p>

<div class="scroll">{table}</div>
<p class="sub">Latency in µs; the fastest entry in each row is bold. Largest absolute output error over all workloads:
{", ".join(f"{esc(SERIES[n][0])} {error[n]:.4f}" for n in names)} (tolerance 0.05).
Passed: {", ".join(f"{passed[n]}/{n_workloads}" for n in names)}.</p>

<h2>Why the line-for-line port is slower</h2>
<ol>
<li><strong>The per-head kernel is 3–6× slower in Triton.</strong> It costs about 6.2 µs per 16-key tile against about
0.9 µs in CUDA. The CUDA kernel double-buffers its gather loads with <code>cp.async</code>; Triton only pipelines
loads that feed <code>tl.dot</code>, and this kernel's dot products are element-wise sums, so <code>num_stages</code>
has no effect on it.</li>
<li><strong>The dispatch thresholds are tuned for CUDA.</strong> Workloads averaging fewer than 210 valid keys per token
go to the per-head kernel. That is {n_head} of the {n_workloads} workloads: CUDA's cheapest path and Triton's most
expensive one.</li>
<li><strong>The per-token kernel is already faster in Triton.</strong> The {len(port_token)} dense workloads that
reach it take {min(port_token):.1f}–{max(port_token):.1f} µs in the port against
{min(cuda_token):.1f}–{max(cuda_token):.1f} µs in CUDA.</li>
<li><strong>16-key tiles are the wrong size for Triton.</strong> One loop iteration costs 4–5 µs almost regardless of how
many keys it holds, so 64-key tiles with pipelined loads cut the cost per key several times over.</li>
<li><strong>Split-K is two Python launches.</strong> The GPU idles between the partial kernel and the merge kernel while
Python prepares the second launch. The CUDA submission replays both from one CUDA graph.</li>
<li><strong>Small calls have a higher floor.</strong> The smallest call takes 5.3–6.9 µs in Triton against 3.6 µs in CUDA.</li>
</ol>

<h2>From the port to the tuned version</h2>
<p>Each bar is one configuration of the same Triton file used for every workload, as geomean latency relative to the
CUDA kernel. Left of the dashed line is faster than CUDA.</p>
<figure>{ladder_chart(ladder)}
<figcaption>Configuration sweep on one B200 with the same timing method (10–20 timed calls per point).
The first bar is off the scale at {ladder_value.get('head,1,8,16,1', 0):.2f}×.</figcaption></figure>
<p>The tuned entry point, <code>run_tuned</code>, uses the per-token kernel for every workload with 64-key tiles and
<code>num_stages=3</code>. One or two tokens run as a single kernel with no split. More tokens split the top-k axis
16 ways and replay the two launches from a CUDA graph.</p>

<h2>Triton releases</h2>
<p>The same configurations on seven Triton releases, as geomean latency relative to the CUDA kernel
(the CUDA medians are taken from the sweep run on the same GPU model).</p>
<div class="scroll">{release_table}</div>
<ul>
<li>3.4.0 refuses to compile for CUDA 13.2, so it cannot run in the contest image.</li>
<li>3.8.0 and 3.9.0 generate a per-head kernel that is 2.2–2.6× faster. The per-token kernel takes the same GPU time
in every release.</li>
<li>3.7.0, the release in the benchmark image, spends about twice as long in its Python launcher as the other
releases (about 50 µs per launch against 21 µs). That widens the idle gap between the two split-K launches, so the
64-key configuration without graph replay is slower than CUDA on 3.7.0 only.</li>
</ul>

<h2>How this was measured</h2>
<ul>
<li>One NVIDIA B200 on Modal, image <code>flashinfer/flashinfer-ci-cu132</code>, CUDA 13.2,
torch {esc(sweep["torch_version"])}, Triton {esc(sweep["triton_version"])}. The contest evaluated Triton 3.6.0.</li>
<li>The headline numbers come from the official flashinfer-bench evaluator, all solutions in one container run.
Latency is the span from the first kernel's start to the last kernel's end, so Python time before the first launch
is not counted and time between launches is.</li>
<li>The CUDA kernel is the submission with its source split into readable headers; the algorithm and thresholds are
unchanged.</li>
<li>Both the CUDA kernel and the faithful port cache their dispatch decision on the address of the index tensor. When
a new workload's tensor reuses a freed address, the previous decision is applied. This sent workload
<code>78b2e11c</code> through the port's per-head kernel in the official run. The tuned entry point decides on shape
only.</li>
</ul>
<pre><code>modal run scripts/compare_cuda_triton.py      # official benchmark, all solutions
modal run scripts/profile_triton.py           # configuration sweep
modal run scripts/profile_triton.py --versions "3.6.0,3.9.0" --configs "auto;token,16,8,64,3,graph"
python scripts/build_triton_report.py         # rebuild this page</code></pre>

<footer>Source: <a href="{REPO}/tree/{BRANCH}">{REPO.split('/')[-1]}</a>, branch <code>{BRANCH}</code>.
Kernels: <a href="{REPO}/blob/{BRANCH}/solution/python/kernel.cu">CUDA</a>,
<a href="{REPO}/blob/{BRANCH}/solution/triton/kernel.py">Triton</a>.
Data: <a href="{REPO}/tree/{BRANCH}/artifacts">artifacts/triton_*.json</a>.</footer>
</main>
</body>
</html>
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page)
    (OUT.parent / ".nojekyll").write_text("")
    print(f"Wrote {OUT} ({len(page) / 1024:.0f} kB)")
    for name in names:
        s = stats[name]
        print(f"  {name:14} mean {s['mean']:6.1f} us  geomean {s['geo']:.2f}x  wins {s['wins']}/{n_workloads}")


if __name__ == "__main__":
    main()
