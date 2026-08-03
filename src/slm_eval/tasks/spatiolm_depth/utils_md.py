import re

import numpy as np
import pandas as pd
from PIL import ImageDraw

MARKER_COLOR = (255, 0, 0)  # red
MARKER_SIZE = 5  # radius of circle
EVAL_POINT_INDEX = 0  # index of the points to evaluate


def _draw_circle(draw, center, color, radius=5):
    center_y, center_x = center
    left = center_x - radius
    top = center_y - radius
    right = center_x + radius
    bottom = center_y + radius
    draw.ellipse([(left, top), (right, bottom)], fill=color, width=2)


def _extract_content(response: str):
    pattern = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"
    response = response.strip()

    if re.search(pattern, response):
        response = re.findall(pattern, response)[0]
        return float(response)
    else:
        return float("nan")


def spatiolm_doc_to_visual(doc: dict):
    if "points" in doc:
        point = doc["points"][EVAL_POINT_INDEX]
        draw = ImageDraw.Draw(doc["image"])
        _draw_circle(draw, point[::-1], MARKER_COLOR, MARKER_SIZE)

        # doc["image"].thumbnail((2560, 1440))
        return [doc["image"].convert("RGB")]
    else:
        point = doc["points_1"][EVAL_POINT_INDEX]
        draw = ImageDraw.Draw(doc["image_1"])
        _draw_circle(draw, point[::-1], MARKER_COLOR, MARKER_SIZE)
        # return [doc["image_1"].convert("RGB")]
        return [doc["image_0"].convert("RGB"), doc["image_1"].convert("RGB")]


def spatiolm_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    return f"{pre_prompt}{post_prompt}"


def spatiolm_doc_to_target(doc, model_specific_target_kwargs=None):
    dists = doc.get("z_distance", doc.get("z_distance_1"))
    return dists[EVAL_POINT_INDEX]


def spatiolm_process_results(doc, results):
    pred = _extract_content(results[0])
    pred = max(pred, 1e-6)

    gt = doc.get("z_distance", doc.get("z_distance_1"))[EVAL_POINT_INDEX]
    gt = max(gt, 1e-6)

    result = {
        "Type": doc["type"],
        "pred": round(pred, 2),
        "gt": round(gt, 2),
        "Delta1": float(max(pred / gt, gt / pred) < 1.25),
        "AbsRel": abs(pred - gt) / gt,
    }
    return {"metric_depth": result}


def detal1_absrel_aggregate_results(results):
    res_df = pd.DataFrame(results)

    group_mean = res_df[["Type", "Delta1", "AbsRel"]].groupby("Type").mean()
    group_mean["Delta1"] *= 100
    results = group_mean.to_dict()

    return results
