from pathlib import Path
import argparse
import yaml
import itertools
import re

def load_yaml(path: Path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def ensure_list(x):
    return x if isinstance(x, list) else [x]

def main():
    p = argparse.ArgumentParser(description="Generate indexed YAML configs from a base + sweep yaml")
    p.add_argument("--base", required=True, help="Path to base YAML (will be copied and overridden)")
    p.add_argument("--sweep", required=True, help="Path to sweep YAML (values should be lists or scalars)")
    p.add_argument("--outdir", default="sweeps", help="Output directory")
    p.add_argument("--prefix", default="config", help="Filename prefix (default: config)")
    p.add_argument("--pad", type=int, default=0, help="Zero-pad width for index. Default uses minimum width to fit all combos (at least 4).")
    args = p.parse_args()

    base_path = Path(args.base)
    sweep_path = Path(args.sweep)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    base = load_yaml(base_path)
    sweep = load_yaml(sweep_path)

    # Normalize sweep values to lists
    sweep_lists = {k: ensure_list(v) for k, v in (sweep.items() if isinstance(sweep, dict) else [])}

    keys = sorted(sweep_lists.keys())
    value_lists = [sweep_lists[k] for k in keys]

    # Cartesian product (if no keys, produce one empty tuple so we still write base)
    combos = list(itertools.product(*value_lists)) if keys else [()]

    total = len(combos)
    pad_width = args.pad if args.pad > 0 else max(4, len(str(total)))

    for idx, combo in enumerate(combos, start=1):
        param_updates = {k: v for k, v in zip(keys, combo)}
        merged = dict(base)  # shallow copy; sweep overrides base
        merged.update(param_updates)

        filename = f"{args.prefix}_{str(idx).zfill(pad_width)}.yaml"
        out_path = outdir / filename

        with out_path.open("w", encoding="utf-8") as of:
            # Use safe_dump; keep order as-is if PyYAML supports it
            yaml.safe_dump(merged, of, sort_keys=False)

    print(f"Wrote {total} YAML file(s) to {outdir}/ (filenames: {args.prefix}_<index>.yaml)")

if __name__ == "__main__":
    main()
