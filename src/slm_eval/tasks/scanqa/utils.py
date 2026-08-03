import os
from collections import defaultdict
from pathlib import Path

import yaml
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge as RougeCap

hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)
with open(Path(__file__).parent / "scanqa.yaml", "r") as f:
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


def scanqa_doc_to_visual(doc):
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


def scanqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    situation = doc["situation"]

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    mid_prompt = lmms_eval_specific_kwargs.get("mid_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    return f"{pre_prompt}{question}{mid_prompt}{situation}{post_prompt}"


def scanqa_process_results(doc, results):
    qa_info = {
        "id": doc["question_id"],
        "answer": results[0],
        "gt": doc["answers"],
    }

    return {"language": qa_info}


def scanqa_aggregate_results(results):
    gts = defaultdict(list)
    res = {}

    for sample_id, item in enumerate(results):
        if not isinstance(item, dict):
            continue
        pred = item.get("answer")
        gt = item.get("gt")
        res[sample_id] = [pred]
        gts[sample_id] = gt

    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (RougeCap(), "ROUGE_L"),
        (Meteor(), "Meteor"),
        (Cider(), "CIDEr"),
    ]

    result = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(gts, res)
        if isinstance(method, list):
            for sc, m in zip(score, method):
                result[m] = sc
        else:
            result[method] = score

    language_metric = {
        k: result[k]
        for k in ["Bleu_1", "Bleu_4", "Meteor", "ROUGE_L", "CIDEr"]
        if k in result
    }

    return language_metric
