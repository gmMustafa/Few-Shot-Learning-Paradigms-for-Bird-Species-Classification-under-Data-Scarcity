"""
utils/split_guard.py

Lightweight split-integrity guard for train/evaluate scripts.

Call split_guard(cfg) at the start of any script that receives a parsed config
dict. It raises ValueError immediately if base_classes and novel_classes overlap,
preventing silent contamination of the episodic training pool.

Usage:
    from utils.split_guard import split_guard

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    split_guard(cfg)   # raises ValueError if split is contaminated
"""


def split_guard(cfg: dict, config_path: str = "<unknown>") -> None:
    """
    Assert that base_classes and novel_classes are non-empty and disjoint.

    Args:
        cfg:         Parsed config dict (output of yaml.safe_load).
        config_path: Path string used only for error messages.

    Raises:
        ValueError: If either list is missing/empty, contains duplicates,
                    or the two lists share any class name.
    """
    base   = cfg.get("base_classes")   or cfg.get("train", {}).get("base_classes")
    novel  = cfg.get("novel_classes")  or cfg.get("train", {}).get("novel_classes")

    if not base:
        raise ValueError(
            f"[split_guard] Config '{config_path}' is missing 'base_classes' or it is empty."
        )
    if not novel:
        raise ValueError(
            f"[split_guard] Config '{config_path}' is missing 'novel_classes' or it is empty."
        )

    # Duplicates within base
    seen = set()
    dupes = []
    for c in base:
        if c in seen:
            dupes.append(c)
        seen.add(c)
    if dupes:
        raise ValueError(
            f"[split_guard] base_classes contains duplicates in '{config_path}': {dupes}"
        )

    # Duplicates within novel
    seen = set()
    dupes = []
    for c in novel:
        if c in seen:
            dupes.append(c)
        seen.add(c)
    if dupes:
        raise ValueError(
            f"[split_guard] novel_classes contains duplicates in '{config_path}': {dupes}"
        )

    # Leakage: intersection must be empty
    leak = set(base) & set(novel)
    if leak:
        raise ValueError(
            f"[split_guard] SPLIT INTEGRITY VIOLATED in '{config_path}': "
            f"{len(leak)} class(es) appear in both base_classes and novel_classes: "
            f"{sorted(leak)}"
        )
