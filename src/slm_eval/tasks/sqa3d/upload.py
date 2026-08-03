import json
import datasets
import argparse

features = datasets.Features(
    {
        "videos": datasets.Value("string"),
        "question": datasets.Value("string"),
        "answers": datasets.Sequence(datasets.Value("string")),
        "situation": datasets.Value("string"),
        "question_id": datasets.Value("int64"),
        "scene_id": datasets.Value("string"),
        "object_ids": datasets.Sequence(datasets.Value("string")),
        "object_names": datasets.Sequence(datasets.Value("string")),
        "position": datasets.Sequence(datasets.Value("float32")),
    }
)


def generator(infos):
    for qa in infos:
        yield qa


def main():
    parser = argparse.ArgumentParser(description="Upload SQA3D dataset to disk")
    parser.add_argument(
        "--input",
        type=str,
        default="data/train-vlm/vlm-3d/SQA3D/test.jsonl",
        help="Path to input JSONL file",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="data/eval/sqa3d",
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
    data.save_to_disk(args.repo_id)

    # Check if the dataset is saved correctly
    data = datasets.load_from_disk(args.repo_id)
    print(data)


if __name__ == "__main__":
    main()
