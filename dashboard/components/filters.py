"""
Sidebar filter widgets with session-state persistence.
"""

import streamlit as st
from data.loader import get_fsl_results, get_methods, get_shots, METHOD_ORDER

_FSL_SPLITS_BIRDS = ["B15/N5", "B10/N10"]
_FSL_SPLITS_CUB   = ["B100/N50", "B150/N50"]
_DATASETS         = ["Birds-20", "CUB-200"]


def shot_filter(key: str = "shot", default: int = 5) -> int:
    opts = get_shots()
    if not opts:
        opts = [1, 3, 5]
    idx = opts.index(default) if default in opts else len(opts) - 1
    return st.sidebar.radio(
        "K-Shot", opts, index=idx,
        key=key,
        format_func=lambda x: f"{x}-shot",
        horizontal=True,
    )


def dataset_filter(key: str = "dataset", default: str = "Birds-20") -> str:
    idx = _DATASETS.index(default) if default in _DATASETS else 0
    return st.sidebar.radio("Dataset", _DATASETS, index=idx, key=key)


def split_filter(key: str = "split", dataset: str = "Birds-20",
                 default: str = "B15/N5") -> str:
    opts = _FSL_SPLITS_BIRDS if dataset == "Birds-20" else _FSL_SPLITS_CUB
    idx = opts.index(default) if default in opts else 0
    return st.sidebar.radio("Split", opts, index=idx, key=key)


def backbone_filter(key: str = "backbone", default: str = "All") -> str:
    opts = ["All", "ResNet-50", "DeiT-S"]
    idx = opts.index(default) if default in opts else 0
    return st.sidebar.selectbox("Backbone", opts, index=idx, key=key)


def distance_filter(key: str = "distance", default: str = "Euclidean") -> str:
    opts = ["Euclidean", "Manhattan", "Cosine", "All"]
    idx = opts.index(default) if default in opts else 0
    return st.sidebar.selectbox("Distance", opts, index=idx, key=key)


def method_multiselect(
    key: str = "methods",
    default_methods: list = None,
    exclude: list = None,
) -> list:
    df = get_fsl_results()
    available = get_methods(df)
    if exclude:
        available = [m for m in available if m not in exclude]
    if default_methods is None:
        default_methods = available
    else:
        default_methods = [m for m in default_methods if m in available]
    return st.sidebar.multiselect(
        "Methods", available,
        default=default_methods,
        key=key,
    )


def ci_toggle(key: str = "show_ci", default: bool = True) -> bool:
    return st.sidebar.toggle("Show 95% CI error bars", value=default, key=key)


def get_global_dataset() -> str:
    """Return the globally-selected dataset from the top-right settings popover."""
    return st.session_state.get("ds_filter", "CUB-200")


def get_active_datasets() -> list:
    """Return list of datasets to render based on the global filter setting."""
    v = get_global_dataset()
    return ["Birds-20", "CUB-200"] if v == "All" else [v]
