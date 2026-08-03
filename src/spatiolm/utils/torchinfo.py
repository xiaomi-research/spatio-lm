import pandas as pd
import torch.nn as nn


def human_format_size(num: int) -> str:
    """Convert numbers to human-readable format: K/M/B"""
    if num < 1_000:
        return str(num)
    elif num < 1_000_000:
        return f"{num / 1_000:.2f}K"
    elif num < 1_000_000_000:
        return f"{num / 1_000_000:.2f}M"
    else:
        return f"{num / 1_000_000_000:.2f}B"


def module_param_stats(module: nn.Module, prefix: str = "", depth: int = 0, seen=None):
    if seen is None:
        seen = set()

    # Current module's own parameters
    own_params = 0
    own_trainable = 0
    for param in module.parameters(recurse=False):
        param_id = id(param)
        if param_id in seen:
            continue
        seen.add(param_id)
        own_params += param.numel()
        if param.requires_grad:
            own_trainable += param.numel()

    # Recursive processing of child modules
    rows = []
    child_params, child_trainable = 0, 0
    for name, child in module.named_children():
        child_prefix = f"{prefix}.{name}" if prefix else name
        child_rows, params, trainable = module_param_stats(
            child, child_prefix, depth + 1, seen
        )
        rows.extend(child_rows)
        child_params += params
        child_trainable += trainable

    total_params = own_params + child_params
    total_trainable = own_trainable + child_trainable

    rows.insert(
        0,
        {
            "Module": prefix if prefix else module.__class__.__name__,
            "Depth": depth,
            "Params": total_params,
            "Trainable": total_trainable,
        },
    )
    return rows, total_params, total_trainable


def summarize_model(module: nn.Module) -> pd.DataFrame:
    rows, total_params, total_trainable = module_param_stats(module)
    df = pd.DataFrame(rows)

    # Calculate percentages
    df["% Params"] = round(
        df["Params"] / total_params * 100 if total_params > 0 else 0, 2
    )
    df["% Trainable"] = round(
        df["Trainable"] / total_params * 100 if total_params > 0 else 0, 2
    )

    # Convert to human readable format
    df["Params"] = df["Params"].apply(human_format_size)
    df["Trainable"] = df["Trainable"].apply(human_format_size)

    # Sort and add total
    # df = df.sort_values(["Depth", "Module"]).reset_index(drop=True)
    # df.loc[len(df)] = ["TOTAL", -1, human_format_size(total_params), human_format_size(total_trainable), 100.0]

    return df[["Module", "Depth", "Params", "Trainable", "% Params", "% Trainable"]]
