import pandas as pd
from loguru import logger as eval_logger
from PIL import Image


def mindcube_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"].strip()

    preprompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    postprompt = lmms_eval_specific_kwargs.get(
        "post_prompt",
        "Answer with the option's letter from the given choices directly.",
    )

    return f"{preprompt}{question}\n{postprompt}"


def mindcube_doc_to_visual(doc):
    image_list = []

    for img in doc["images"]:
        if isinstance(img, Image.Image):
            image = img
        elif isinstance(img, str):
            image = Image.open(img)
        else:
            raise ValueError("Unsupported image data")

        image = image.convert("RGB")
        image_list.append(image)
    return image_list


def mindcube_process_results(doc, results):
    pred = results[0].strip().split(".")[0]
    gt = doc["gt_answer"].strip()

    score = float(pred == gt)

    result = {
        "score": score,
        "pred": pred,
        "gt": gt,
        "type": doc["type"],
    }

    return {"accuracy": result}


def mindcube_aggregate_results(results):
    res_df = pd.DataFrame(results)

    overall_acc = 100 * res_df["score"].mean()

    group_acc = (
        res_df[["type", "score"]]
        .groupby("type")
        .mean()
        .to_dict()["score"]
    )
    eval_logger.info(f"MindCube Accuracy By Type: {group_acc}")
    return overall_acc
