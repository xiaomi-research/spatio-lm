import random
from pathlib import Path
from typing import Sequence, Union

import torchvision.transforms as tvf
from torch.utils.data import Subset

from spatiolm.datasets import DepthLmDataset
from swift.llm import DATASET_MAPPING, DatasetMeta, register_dataset

__all__ = ["load_torch_dataset"]


def load_torch_dataset(datasets: Union[str, Sequence[str]], **kwargs):
    if isinstance(datasets, str):
        datasets = [datasets]

    for dataset in datasets:
        # Parse datasest_id, such as, "torch::dataset_id[#data_nums][::kwargs]"
        if not dataset.startswith("torch::"):
            continue

        # Extract dataset_id, data_nums, kwargs
        parts = dataset.split("::")
        dataset_id = parts[1]
        data_nums = None
        if "#" in dataset_id:
            dataset_id, data_nums = dataset_id.split("#")

        dataset_kwargs = {}
        if len(parts) > 2:
            try:
                dataset_kwargs = eval(f"dict({parts[2]})")
            except Exception as e:
                raise ValueError(f"Invalid kwargs: {parts[2]}") from e

        dataset_meta = DATASET_MAPPING.get((dataset_id, dataset_id, None))
        if dataset_meta is None:
            raise ValueError(f"Dataset {dataset_id} not found")

        dataset = dataset_meta.load_function(**dataset_kwargs, **kwargs)

        if data_nums is not None:
            data_nums = int(data_nums)
            if data_nums < len(dataset):
                indices = random.sample(range(len(dataset)), data_nums)

                dataset = Subset(dataset, indices)
            elif data_nums > len(dataset):
                repeat_num = data_nums // len(dataset)
                indices = list(range(len(dataset))) * repeat_num
                indices += random.sample(range(len(dataset)), data_nums - len(indices))

                dataset = Subset(dataset, indices)

        # dataset = load_dataset(dataset, **kwargs)
    return dataset


def load_depthlm_dataset(**kwargs):
    data_root = kwargs.pop("data_root", "data/train-v3r")
    if not Path(data_root).exists():
        raise ValueError(f"load_depthlm_dataset, unexisted data_root={data_root}")

    video_folders = kwargs.pop(
        "video_folders",
        [
            "arkitscenes",  # 8900
            "hypersim#2000",
            "mvs-synth#1000",
            "scannet#2000",
            "scannetpp#1000",
            "mapfree#1000",
            "vkitti#1000",
            "waymo#5000",
        ],
    )
    kwargs.setdefault("mark_radius", 2)
    kwargs.setdefault("pipelines", [tvf.ColorJitter(0.2, 0.2, 0.2, 0.05)])

    return DepthLmDataset(video_folders, data_root, **kwargs)


register_dataset(
    DatasetMeta(
        ms_dataset_id="cvlm3d/depthlm",
        hf_dataset_id="cvlm3d/depthlm",
        load_function=load_depthlm_dataset,
        help="A dataset for depth estimation",
        tags=["depth", "videos", "images"],
    ),
    exist_ok=True,
)
