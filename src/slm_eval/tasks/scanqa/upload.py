import json
import argparse
import datasets


features = datasets.Features(
    {
        "videos": datasets.Value("string"),
        "question": datasets.Value("string"),
        "answers": datasets.Sequence(datasets.Value("string")),
        "situation": datasets.Value("string"),
        "question_id": datasets.Value("string"),
        "scene_id": datasets.Value("string"),
        "object_ids": datasets.Sequence(datasets.Value("int64")),
        "object_names": datasets.Sequence(datasets.Value("string")),
    }
)


def generator(infos):
    for qa in infos:
        yield qa


def main():
    parser = argparse.ArgumentParser(description="Upload ScanQA dataset to disk")
    parser.add_argument(
        "--input",
        type=str,
        default="data/train-vlm/vlm-3d/ScanQA/val.jsonl",
        help="Path to input JSONL file",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="data/eval/scanqa",
        help="Repository ID for saving dataset",
    )

    args = parser.parse_args()

    with open(args.input, "r") as f:
        infos = [json.loads(l) for l in f]
    all_qas = []
    for qa in infos:
        del qa["messages"]
        all_qas.append(qa)

    data_test = datasets.Dataset.from_generator(
        generator,
        gen_kwargs={"infos": all_qas},
        features=features,
        split=datasets.Split.TEST,
        num_proc=1,
    )

    data = datasets.DatasetDict({"test": data_test})
    repo_id = args.repo_id
    data.save_to_disk(repo_id)

    data = datasets.load_from_disk(repo_id)
    print(data)


if __name__ == "__main__":
    main()
