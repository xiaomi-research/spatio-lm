import os
import re
from pathlib import Path

import yaml


hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)
with open(Path(__file__).parent / "sqa3d.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)
config = yaml.safe_load("".join(safe_data))

if "dataset_kwargs" in config:
    cache_name = config["dataset_kwargs"]["cache_dir"]
else:
    cache_name = config["dataset_path"]


def clean_answer(data):
    data = data.lower()
    data = re.sub("[ ]+$", "", data)
    data = re.sub("^[ ]+", "", data)
    data = re.sub(" {2,}", " ", data)

    data = re.sub("\.[ ]{2,}", ". ", data)
    data = re.sub("[^a-zA-Z0-9,'\s\-:]+", "", data)
    data = re.sub("ç", "c", data)
    data = re.sub("’", "'", data)
    data = re.sub(r"\bletf\b", "left", data)
    data = re.sub(r"\blet\b", "left", data)
    data = re.sub(r"\btehre\b", "there", data)
    data = re.sub(r"\brigth\b", "right", data)
    data = re.sub(r"\brght\b", "right", data)
    data = re.sub(r"\bbehine\b", "behind", data)
    data = re.sub(r"\btv\b", "TV", data)
    data = re.sub(r"\bchai\b", "chair", data)
    data = re.sub(r"\bwasing\b", "washing", data)
    data = re.sub(r"\bwaslked\b", "walked", data)
    data = re.sub(r"\boclock\b", "o'clock", data)
    data = re.sub(r"\bo\'[ ]+clock\b", "o'clock", data)

    # digit to word, only for answer
    data = re.sub(r"\b0\b", "zero", data)
    data = re.sub(r"\bnone\b", "zero", data)
    data = re.sub(r"\b1\b", "one", data)
    data = re.sub(r"\b2\b", "two", data)
    data = re.sub(r"\b3\b", "three", data)
    data = re.sub(r"\b4\b", "four", data)
    data = re.sub(r"\b5\b", "five", data)
    data = re.sub(r"\b6\b", "six", data)
    data = re.sub(r"\b7\b", "seven", data)
    data = re.sub(r"\b8\b", "eight", data)
    data = re.sub(r"\b9\b", "nine", data)
    data = re.sub(r"\b10\b", "ten", data)
    data = re.sub(r"\b11\b", "eleven", data)
    data = re.sub(r"\b12\b", "twelve", data)
    data = re.sub(r"\b13\b", "thirteen", data)
    data = re.sub(r"\b14\b", "fourteen", data)
    data = re.sub(r"\b15\b", "fifteen", data)
    data = re.sub(r"\b16\b", "sixteen", data)
    data = re.sub(r"\b17\b", "seventeen", data)
    data = re.sub(r"\b18\b", "eighteen", data)
    data = re.sub(r"\b19\b", "nineteen", data)
    data = re.sub(r"\b20\b", "twenty", data)
    data = re.sub(r"\b23\b", "twenty-three", data)

    # misc
    # no1, mat2, etc
    data = re.sub(r"\b([a-zA-Z]+)([0-9])\b", r"\g<1>", data)
    data = re.sub(r"\ba\b ([a-zA-Z]+)", r"\g<1>", data)
    data = re.sub(r"\ban\b ([a-zA-Z]+)", r"\g<1>", data)
    data = re.sub(r"\bthe\b ([a-zA-Z]+)", r"\g<1>", data)

    data = re.sub(r"\bbackwards\b", "backward", data)

    return data


def answer_match(pred, gts):
    # return EM and refined EM
    if pred in gts:
        return 1, 1
    for gt in gts:
        if "".join(pred.split()) in "".join(gt.split()) or "".join(
            gt.split()
        ) in "".join(pred.split()):
            return 0, 1
    return 0, 0


def sqa3d_doc_to_visual(doc):
    if os.path.exists(cache_name):
        cache_dir = cache_name
    else:
        cache_dir = os.path.join(base_cache_dir, cache_name)

    video_path = doc["scene_id"] + ".mp4"
    video_path = os.path.join(cache_dir, video_path)
    if os.path.exists(video_path):
        video_path = video_path
    else:
        raise FileExistsError(f"video path:{video_path} does not exist.")
    return [video_path]


def sqa3d_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    situation = doc["situation"]

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    mid_prompt = lmms_eval_specific_kwargs.get("mid_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    return f"{pre_prompt}{question}{mid_prompt}{situation}{post_prompt}"


def sqa3d_process_results(doc, results):
    target = clean_answer(doc["answers"][0])
    pred = clean_answer(results[0])

    em_flag, em_refined_flag = answer_match(pred, [target])
    em_1 = int(pred == target)
    em_r1 = em_refined_flag

    return {"EM-1": em_1, "EM-R1": em_r1, "target": target, "pred": pred}


def sqa3d_aggregate_results(results, metric):
    score = sum(results) / len(results)
    print(f"{metric}: {score}")
    return score


def sqa3d_em_1(results, metric="EM-1"):
    return sqa3d_aggregate_results(results, metric)


def sqa3d_em_r1(results, metric="EM-R1"):
    return sqa3d_aggregate_results(results, metric)
