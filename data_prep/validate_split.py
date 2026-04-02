"""
data_prep/validate_split.py

Split integrity validator for few-shot learning experiments.

Checks that:
  1. base_classes and novel_classes are DISJOINT (no class appears in both)
  2. No duplicates within either list
  3. All class names exist as sub-folders in the configured data directories
  4. (CUB only) val_fsl_classes from fsl_split_standard.json are also disjoint
     from base and novel lists defined in the config

Exit codes:
  0 = all checks passed
  1 = one or more checks failed (details printed to stdout)

Usage:
    # Validate a single Birds20 config
    python data_prep/validate_split.py --config configs/birds20/proto_configs.yaml

    # Validate a single CUB config
    python data_prep/validate_split.py --config configs/cub/protonet_resnet50.yaml

    # Validate ALL configs at once
    python data_prep/validate_split.py --all
"""

import os
import sys
import json
import argparse
import glob

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Core check functions
# ---------------------------------------------------------------------------

def _err(msg: str):
    print(f"  [FAIL] {msg}")


def _ok(msg: str):
    print(f"  [OK]   {msg}")


def check_no_duplicates(name: str, classes: list) -> int:
    """Returns number of failures."""
    seen, dupes = set(), []
    for c in classes:
        if c in seen:
            dupes.append(c)
        seen.add(c)
    if dupes:
        _err(f"{name} contains duplicates: {dupes}")
        return 1
    _ok(f"{name} has no duplicates ({len(classes)} classes)")
    return 0


def check_disjoint(name_a: str, set_a: set, name_b: str, set_b: set) -> int:
    intersection = set_a & set_b
    if intersection:
        _err(f"LEAKAGE -- {name_a} & {name_b} = {sorted(intersection)}")
        return 1
    _ok(f"{name_a} & {name_b} = empty  (no leakage)")
    return 0


def check_classes_in_folder(classes: list, folder: str, split_name: str) -> int:
    if not folder or not os.path.isdir(folder):
        print(f"  [SKIP] data folder not found, skipping existence check: {folder}")
        return 0
    existing = set(os.listdir(folder))
    missing = [c for c in classes if c not in existing]
    if missing:
        _err(f"{split_name} — {len(missing)} class(es) missing from {folder}: {missing[:5]}"
             + (" ..." if len(missing) > 5 else ""))
        return len(missing)
    _ok(f"{split_name} — all {len(classes)} classes exist in {os.path.basename(folder)}/")
    return 0


# ---------------------------------------------------------------------------
# Per-config validator
# ---------------------------------------------------------------------------

def validate_config(config_path: str) -> bool:
    """
    Validate one config file. Returns True if all checks pass.
    """
    abs_path = config_path if os.path.isabs(config_path) else \
               os.path.normpath(os.path.join(PROJECT_ROOT, config_path))

    if not os.path.isfile(abs_path):
        print(f"\n[ERROR] Config not found: {abs_path}")
        return False

    with open(abs_path) as f:
        cfg = yaml.safe_load(f)

    print(f"\n{'='*60}")
    print(f"Config : {os.path.relpath(abs_path, PROJECT_ROOT)}")
    print(f"{'='*60}")

    failures = 0

    base_classes  = cfg.get('base_classes', [])
    novel_classes = cfg.get('novel_classes', [])

    # Full-supervised baselines (e.g. Full-20) intentionally have no base/novel split.
    # Skip integrity checks for configs that declare neither list.
    if not base_classes and not novel_classes:
        print(f"  [SKIP] No base_classes/novel_classes defined — "
              f"full-supervised config, no split to validate.")
        print(f"Result: SKIP (not applicable)")
        return True  # not a failure

    if not base_classes:
        _err("base_classes list is empty or missing")
        failures += 1

    if not novel_classes:
        _err("novel_classes list is empty or missing")
        failures += 1

    if not base_classes or not novel_classes:
        print(f"Result: {'PASS' if failures == 0 else 'FAIL'} ({failures} issue(s))")
        return failures == 0

    # 1. Duplicates
    failures += check_no_duplicates("base_classes", base_classes)
    failures += check_no_duplicates("novel_classes", novel_classes)

    # 2. Disjointness: base vs novel
    failures += check_disjoint(
        "base_classes", set(base_classes),
        "novel_classes", set(novel_classes),
    )

    # 3. CUB only: check against val_fsl_classes from metadata
    fsl_meta_path = os.path.join(
        PROJECT_ROOT, 'data_cub_prepared', 'metadata', 'fsl_split_standard.json')
    if os.path.isfile(fsl_meta_path):
        with open(fsl_meta_path) as f:
            fsl_meta = json.load(f)
        val_fsl = fsl_meta.get('val_fsl_classes', [])
        if val_fsl:
            print(f"  CUB metadata found — also checking val_fsl_classes ({len(val_fsl)} classes)")
            # Only check CUB configs (their classes will be in the CUB class names format)
            if set(base_classes) & set(fsl_meta.get('base_classes', [])):
                # This config references CUB classes; check val_fsl leakage
                failures += check_disjoint(
                    "base_classes", set(base_classes),
                    "val_fsl_classes", set(val_fsl),
                )
                failures += check_disjoint(
                    "novel_classes", set(novel_classes),
                    "val_fsl_classes", set(val_fsl),
                )

    # 4. Class existence in data folders
    # Resolve data root from config (handles both baseline and FSL config layouts)
    script_dir = os.path.dirname(abs_path)

    # Baseline configs use data_train/data_test keys
    # FSL configs use data.train/data.test keys
    train_rel = cfg.get('data_train') or (cfg.get('data', {}) or {}).get('train')
    if train_rel:
        # Script directory is the configs/ folder; scripts are resolved relative to the
        # train script, not the config. Try to resolve relative to project root as fallback.
        # For existence checks, try a few candidate base paths.
        candidates = [
            os.path.normpath(os.path.join(PROJECT_ROOT, 'train_evaluate', 'baseline', 'resNet50_bX', train_rel)),
            os.path.normpath(os.path.join(PROJECT_ROOT, 'train_evaluate', 'proto_net', train_rel)),
            os.path.normpath(os.path.join(PROJECT_ROOT, 'train_evaluate', 'matchingnet', train_rel)),
            os.path.normpath(os.path.join(PROJECT_ROOT, 'train_evaluate', 'vit_backbone', 'baseline', train_rel)),
        ]
        train_dir = next((c for c in candidates if os.path.isdir(c)), None)
        if train_dir:
            failures += check_classes_in_folder(
                base_classes + novel_classes, train_dir, "train split")
        else:
            print(f"  [SKIP] Could not resolve train data folder from '{train_rel}'")

    failures_str = f"{failures} issue(s)"
    result = "PASS" if failures == 0 else "FAIL"
    print(f"Result: {result} ({failures_str})")
    return failures == 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Validate base/val/novel class split integrity in FSL configs'
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--config',
        help='Path to a single config yaml (relative to project root or absolute)',
    )
    group.add_argument(
        '--all',
        action='store_true',
        help='Validate all config yaml files in configs/ and configs/cub/',
    )
    args = parser.parse_args()

    if args.all:
        patterns = [
            os.path.join(PROJECT_ROOT, 'configs', '*.yaml'),
            os.path.join(PROJECT_ROOT, 'configs', 'cub', '*.yaml'),
        ]
        configs = []
        for pat in patterns:
            configs.extend(sorted(glob.glob(pat)))
        configs = [os.path.relpath(c, PROJECT_ROOT) for c in configs]
    else:
        configs = [args.config]

    print(f"Validating {len(configs)} config(s)...")

    all_passed = True
    results = []
    for cfg_path in configs:
        passed = validate_config(cfg_path)
        results.append((cfg_path, passed))
        if not passed:
            all_passed = False

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for cfg_path, passed in results:
        status = "PASS" if passed else "FAIL"
        label = os.path.relpath(cfg_path, PROJECT_ROOT) if os.path.isabs(cfg_path) else cfg_path
        print(f"  [{status}] {label}")

    print()
    if all_passed:
        print("All configs passed split integrity checks.")
        sys.exit(0)
    else:
        n_fail = sum(1 for _, p in results if not p)
        print(f"FAILED: {n_fail}/{len(results)} config(s) have split integrity issues.")
        sys.exit(1)


if __name__ == '__main__':
    main()
