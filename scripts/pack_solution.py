"""
Pack solution source files into solution.json.

Reads configuration from config.toml and packs the appropriate source files
(Triton or CUDA) into a Solution JSON file for submission.
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from flashinfer_bench import BuildSpec
from flashinfer_bench.agents import pack_solution_from_files


def load_config(config_path: Path | None = None) -> dict:
    """Load configuration from config.toml (or an alternate path)."""
    path = config_path or (PROJECT_ROOT / "config.toml")
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "rb") as f:
        return tomllib.load(f)


def pack_solution(output_path: Path = None, config_path: Path | None = None) -> Path:
    """Pack solution files into a Solution JSON."""
    config = load_config(config_path)

    solution_config = config["solution"]
    build_config = config["build"]

    language = build_config["language"]
    entry_point = build_config["entry_point"]

    sub = build_config.get("source_dir")
    if sub:
        source_dir = PROJECT_ROOT / "solution" / sub
    elif language == "triton":
        source_dir = PROJECT_ROOT / "solution" / "triton"
    elif language == "cuda":
        source_dir = PROJECT_ROOT / "solution" / "cuda"
    elif language == "python":
        source_dir = PROJECT_ROOT / "solution" / "python"
    else:
        raise ValueError(f"Unsupported language: {language}")

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    # Create build spec
    dps = build_config.get("destination_passing_style", True)
    spec_kwargs: dict = dict(
        language=language,
        target_hardware=["cuda"],
        entry_point=entry_point,
        destination_passing_style=dps,
    )
    if build_config.get("binding") is not None:
        spec_kwargs["binding"] = build_config["binding"]
    spec = BuildSpec(**spec_kwargs)

    description = solution_config.get("description", "")

    # Pack the solution
    solution = pack_solution_from_files(
        path=str(source_dir),
        spec=spec,
        name=solution_config["name"],
        definition=solution_config["definition"],
        author=solution_config["author"],
        description=description,
    )

    # Write to output file
    if output_path is None:
        output_path = PROJECT_ROOT / "solution.json"

    output_path.write_text(solution.model_dump_json(indent=2))
    print(f"Solution packed: {output_path}")
    print(f"  Name: {solution.name}")
    print(f"  Definition: {solution.definition}")
    print(f"  Author: {solution.author}")
    print(f"  Language: {language}")

    return output_path


def main():
    """Entry point for pack_solution script."""
    import argparse

    parser = argparse.ArgumentParser(description="Pack solution files into solution.json")
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output path for solution.json (default: ./solution.json)"
    )
    parser.add_argument(
        "-c", "--config",
        type=Path,
        default=None,
        help="Path to config TOML (default: ./config.toml). E.g. config.batched_pytorch.toml for the PyTorch baseline.",
    )
    args = parser.parse_args()

    try:
        pack_solution(args.output, config_path=args.config)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
