import re

import numpy as np
import pandas as pd
from loguru import logger
from PIL import ImageDraw

MARKER_COLOR = (255, 0, 0)  # red
MARKER_SIZE = 5  # radius of circle
EVAL_POINT_INDEX = 0  # index of the points to evaluate


TASK_TEMPLATE = {
    "speed": "Calculate the required speed(m/s) to reach this point in {second:.2f} seconds and provide the direct answer.",
    "time": "Calculate the travel time(s) to reach this point at {speed:.2f}m/s speed and provide the direct answer.",
    "two_point_distance": "Calculate the distance(m) between these two points and provide the direct answer.",
    "camera_pose": "Calculate the movement(m) between given two images and provide the direct answer.",
    "cross_view": "Estimate the 3D spatial distance between the two marked points on the given images and provide the direct answer.",
}


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


def _calc_two_points_distance(p1, p2, d1, d2, intrinsic1, intrinsic2=None):
    if intrinsic2 is None:
        intrinsic2 = intrinsic1

    fx1, fy1, cx1, cy1 = intrinsic1
    fx2, fy2, cx2, cy2 = intrinsic2
    # 3D points in camera coordinate system
    X1 = (p1[0] - cx1) * d1 / fx1
    Y1 = (p1[1] - cy1) * d1 / fy1
    Z1 = d1

    X2 = (p2[0] - cx2) * d2 / fx2
    Y2 = (p2[1] - cy2) * d2 / fy2
    Z2 = d2

    distance = ((X2 - X1) ** 2 + (Y2 - Y1) ** 2 + (Z2 - Z1) ** 2) ** 0.5
    return distance


def spatiolm_doc_to_visual(doc: dict):
    item_type = doc["type"].split(":")[0]
    if item_type in ("speed", "time"):
        point = doc["points_1"][EVAL_POINT_INDEX]
        draw = ImageDraw.Draw(doc["image_1"])
        _draw_circle(draw, point[::-1], MARKER_COLOR, MARKER_SIZE)
        return [doc["image_1"].convert("RGB")]

    elif item_type == "two_point_distance":
        point0 = doc["points_1"][EVAL_POINT_INDEX]
        point1 = doc["points_1"][EVAL_POINT_INDEX + 1]
        draw = ImageDraw.Draw(doc["image_1"])
        _draw_circle(draw, point0[::-1], MARKER_COLOR, MARKER_SIZE)
        _draw_circle(draw, point1[::-1], MARKER_COLOR, MARKER_SIZE)

        return [doc["image_1"].convert("RGB")]

    elif item_type == "camera_pose":
        return [doc["image_0"].convert("RGB"), doc["image_1"].convert("RGB")]

    elif item_type == "cross_view":
        point0 = doc["points_0"][EVAL_POINT_INDEX]
        point1 = doc["points_1"][EVAL_POINT_INDEX]
        draw0 = ImageDraw.Draw(doc["image_0"])
        draw1 = ImageDraw.Draw(doc["image_1"])
        _draw_circle(draw0, point0[::-1], MARKER_COLOR, MARKER_SIZE)
        _draw_circle(draw1, point1[::-1], MARKER_COLOR, MARKER_SIZE)
        return [doc["image_0"].convert("RGB"), doc["image_1"].convert("RGB")]

    else:
        raise TypeError(f"Unsupported task type: {doc['type']}")


def spatiolm_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]

    item_type = doc["type"].split(":")[0].strip()
    if item_type in ("speed", "time"):
        val = float(doc["type"].split(":")[-1].strip())
        prompt = TASK_TEMPLATE[item_type].format(second=val, speed=val)

    elif item_type in ("two_point_distance", "camera_pose", "cross_view"):
        prompt = TASK_TEMPLATE[item_type]
    else:
        raise TypeError(f"Unsupported task type: {doc['type']}")

    return f"{pre_prompt}{prompt}{post_prompt}"


def spatiolm_doc_to_target(doc, model_specific_target_kwargs=None):
    return doc["type"]


def spatiolm_process_results(doc, results):
    pred = _extract_content(results[0])
    pred = max(pred, 1e-6)

    item_type = doc["type"].split(":")[0]
    if item_type in ("speed", "time"):
        distance = doc["z_distance_1"][EVAL_POINT_INDEX]
        val = float(doc["type"].split(":")[-1].strip())

        gt = distance / val

    elif item_type == "two_point_distance":
        gt = _calc_two_points_distance(
            doc["points_1"][EVAL_POINT_INDEX],
            doc["points_1"][EVAL_POINT_INDEX + 1],
            doc["z_distance_0"][EVAL_POINT_INDEX],
            doc["z_distance_0"][EVAL_POINT_INDEX + 1],
            doc["intrinsic_1"],
        )

    elif item_type == "camera_pose":
        cam_xyz0 = np.array(doc["pose_0"])[:3, -1]
        cam_xyz1 = np.array(doc["pose_1"])[:3, -1]
        gt = np.linalg.norm(cam_xyz0 - cam_xyz1)

    elif item_type == "cross_view":
        gt = _calc_two_points_distance(
            doc["points_0"][EVAL_POINT_INDEX],
            doc["points_1"][EVAL_POINT_INDEX],
            doc["z_distance_0"][EVAL_POINT_INDEX],
            doc["z_distance_1"][EVAL_POINT_INDEX],
            doc["intrinsic_0"],
            doc["intrinsic_1"],
        )
    else:
        raise TypeError(f"Unsupported task type: {doc['type']}")

    gt = max(gt, 1e-6)

    result = {
        "Type": item_type,
        "pred": round(pred, 2),
        "gt": round(gt, 2),
        "Delta1": float(max(pred / gt, gt / pred) < 1.25),
        "AbsRel": abs(pred - gt) / gt,
    }
    return {"delta1": result}


def detal1_absrel_aggregate_results(results):
    res_df = pd.DataFrame(results)
    score = res_df["Delta1"].mean() * 100

    group_mean = res_df[["Type", "Delta1", "AbsRel"]].groupby("Type").mean()
    group_mean["Delta1"] *= 100
    results = group_mean.to_dict()

    logger.info("spatiolm_depth_mt AbsRel: {}".format(results.pop('AbsRel')))
    results["avg"] = score

    return results
