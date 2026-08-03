import io
import os
import random
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger as eval_logger
from PIL import Image

# Read config
with open(Path(__file__).parent / "viewspatialbench.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)
config = yaml.safe_load("".join(safe_data))

cache_dir = Path(config["dataset_path"]).parent
if "dataset_kwargs" in config:
    cache_dir = config["dataset_kwargs"].get("cache_dir", cache_dir)


def viewspatialbench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"].strip()
    choices = "Options:\n" + doc["choices"].strip()

    preprompt = lmms_eval_specific_kwargs.get(
        "pre_prompt", "These are frame(s) of a video."
    )
    postprompt = lmms_eval_specific_kwargs.get(
        "post_prompt",
        "Answer with the option's letter from the given choices directly.",
    )

    return "\n".join([preprompt, question, choices, postprompt])


def viewspatialbench_doc_to_visual(doc):
    MAX_FRAMES = os.environ.get("MAX_FRAMES", 8)

    image_list = []

    resize_flag = False
    image_files = doc["image_path"]
    if len(image_files) > MAX_FRAMES:
        image_files = random.sample(image_files, MAX_FRAMES)
        resize_flag = True

    for img_data in image_files:
        # Handle both PIL Image objects and raw bytes
        if isinstance(img_data, Image.Image):
            image = img_data
        elif isinstance(img_data, str):
            image = Image.open(f"{cache_dir}/{img_data}")
        else:
            # Assume it's raw bytes data
            image = Image.open(io.BytesIO(img_data))
        if resize_flag:
            image.thumbnail((448, 448))
        image = image.convert("RGB")
        image_list.append(image)
    return image_list


def viewspatialbench_doc_to_target(doc):
    return doc["answer"].strip().split(".")[0]


def viewspatialbench_process_results(doc, results):
    pred = results[0].strip().split(".")[0]
    gt = viewspatialbench_doc_to_target(doc)

    score = float(pred == gt)

    result = {
        "score": score,
        "pred": pred,
        "gt": gt,
        "question_type": doc["question_type"],
    }

    return {"viewspatialbench_acc": result}


def viewspatialbench_aggregate_results(results):
    res_df = pd.DataFrame(results)

    overall_acc = 100 * res_df["score"].mean()

    group_acc = (
        res_df[["question_type", "score"]]
        .groupby("question_type")
        .mean()
        .to_dict()["score"]
    )
    eval_logger.info(f"ViewSpatialBench Accuracy By Question Type: {group_acc}")
    return overall_acc
