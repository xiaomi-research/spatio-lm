# Copyright 2025 Xiaomi Corporation.
import re


def embspatial_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def embspatial_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return doc["question"] + lmms_eval_specific_kwargs.get("post_prompt")


def embspatial_process_results(doc, results):
    prediction = results[0]

    match = re.search(r"([A-Da-d])\.?", prediction)
    if match:
        prediction = match.group(1)

    final_answer = prediction.lower()
    gt_answer = doc["answer_letter"].lower()

    acc = float(final_answer == gt_answer)
    return {"accuracy": acc}


def embspatial_aggregate_results(results):
    correct = sum(results)
    total = len(results)
    return correct / total
