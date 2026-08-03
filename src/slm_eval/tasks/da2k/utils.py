import re

import pandas as pd
from PIL import ImageDraw

QUESTION = "In the image, Which marked points A (red) and B (green) is closer to me? Answer only: 'A' or 'B'."


def _match_think_content(text, strict=False):
    """
    Extracts the content after the last </think> tag in the given text.

    Args:
        text (str): The text containing </think> tags

    Returns:
        str: The content after the last </think> tag, or the original text if no </think> tag is found
    """
    # Find the last occurrence of </think>
    last_think_end = text.rfind("</think>")
    if last_think_end != -1:
        return text[last_think_end + len("</think>") :].strip()
    else:
        return "" if strict else text


def _match_answer(pred, target):
    match = re.match("([AB]).*", pred)
    if match:
        pred = match.group(1)
    return 1.0 if pred.lower() == target.lower() else 0.0


def _draw_circle(draw, center, color, radius=5):
    center_y, center_x = center
    left = center_x - radius
    top = center_y - radius
    right = center_x + radius
    bottom = center_y + radius
    draw.ellipse([(left, top), (right, bottom)], fill=color, width=2)


def da2k_doc_to_visual(doc):
    point1 = doc["point1"]
    point2 = doc["point2"]
    draw = ImageDraw.Draw(doc["image"])
    _draw_circle(draw, point1, (255, 0, 0), 5)
    _draw_circle(draw, point2, (0, 255, 0), 5)

    # doc["image"].thumbnail((2560, 1440))
    return [doc["image"].convert("RGB")]


def da2k_process_results(doc, results):
    pred = _match_think_content(results[0])

    target = "A" if doc["closer_point"] == "point1" else "B"
    score = _match_answer(pred, target)
    scene = doc.get("scene", doc.get("tag"))
    accuracy = {"score": score, "pred": pred, "gt": target, "scene": scene}
    return {"accuracy": accuracy}


def da2k_aggregate_results(results):
    res_df = pd.DataFrame(results)
    score = res_df["score"].mean() * 100

    group_mean = res_df[["scene", "score"]].groupby("scene").mean()
    group_mean["score"] *= 100
    # print(group_mean.to_dict())
    results = group_mean["score"].to_dict()
    results["avg"] = score

    return results


def da2k_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    return f"{pre_prompt}{QUESTION}\n{post_prompt}"


def da2k_doc_to_target(doc, model_specific_target_kwargs=None):
    # return doc['closer_point']
    # return "A"
    return "A" if doc["closer_point"] == "point1" else "B"
