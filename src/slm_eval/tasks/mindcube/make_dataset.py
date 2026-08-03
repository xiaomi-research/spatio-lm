import argparse
import pandas as pd
from datasets import Dataset, DatasetDict, Image, Sequence


def main():
    parser = argparse.ArgumentParser(description="Process MindCube dataset")
    parser.add_argument(
        "--data_root",
        default="./data/eval/MindCube/data",
        help="Path to raw data directory (default: %(default)s)",
    )
    parser.add_argument(
        "--output_path",
        default="./data/eval/MindCube",
        help="Path to save processed dataset (default: %(default)s)",
    )
    args = parser.parse_args()

    df = pd.read_json(
        f"{args.data_root}/raw/MindCube_tinybench.jsonl",
        lines=True,
        dtype={"type": str, "meta_info": str, "img_time": str},
    )

    df["images"] = df["images"].map(lambda xx: [f"{args.data_root}/{x}" for x in xx])

    dataset = Dataset.from_pandas(df).cast_column("images", Sequence(Image()))

    DatasetDict({"test": dataset}).save_to_disk(args.output_path)


if __name__ == "__main__":
    main()
