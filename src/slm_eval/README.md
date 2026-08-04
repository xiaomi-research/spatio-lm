# SpatioLM Evaluation

`slm_eval` extends LMMs-Eval with the SpatioLM checkpoint adapter and additional
spatial perception, understanding, and reasoning tasks. The package is exposed
through the same command-line interface as LMMs-Eval:

```bash
slm_eval --tasks list
```

## SpatioLM depth benchmark

The processed benchmark is published at
[`edatai/spatiolm-depth`](https://huggingface.co/datasets/edatai/spatiolm-depth)
in the Hugging Face `DatasetDict.save_to_disk` format.

Download the complete repository without changing its directory structure:

```bash
hf download edatai/spatiolm-depth \
  --repo-type dataset \
  --local-dir data/eval/spatiolm_depth
```

Confirm that the dataset can be loaded:

```bash
python - <<'PY'
from datasets import load_from_disk

dataset = load_from_disk("data/eval/spatiolm_depth")
print(dataset)
PY
```

The task YAML files use `dataset_path: ./data/eval/spatiolm_depth` and
`load_from_disk: True`. Run evaluation from the repository root, or update
`dataset_path` if the benchmark is stored elsewhere. Do not use `load_dataset`
for this repository because it intentionally preserves the saved Arrow layout.

### Splits and tasks

| Task | Split | Examples | Evaluation |
| --- | --- | ---: | --- |
| `spatiolm_depth_sv` | `single_view` | 1,600 | Single-image metric depth on SUN RGB-D, NYUv2, and Waymo-derived samples. |
| `spatiolm_depth_mv` | `multi_view` | 1,492 | Multi-view metric depth on NRGBD, ScanNet v2, and KITTI-derived samples. |
| `spatiolm_depth_mt` | `relate_task` | 2,800 | Speed, travel time, two-point distance, camera motion, and cross-view distance. |

The single-view and multi-view tasks report per-domain delta accuracy under the
`max(pred / gt, gt / pred) < 1.25` criterion together with absolute relative
error. The relation task reports the same quantities per relation type and an
overall delta accuracy.

### Data fields

- `single_view`: `image`, `depth`, `points`, `z_distance`, and `type`.
- `multi_view`: paired `image_*`, `depth_*`, `points_*`, `z_distance_*`, and
  `type` fields.
- `relate_task`: paired images and depth maps plus `intrinsic_*`, `pose_*`,
  sampled points, camera-space depth, Euclidean distance, and relation `type`.

Images and depth maps are embedded in the Arrow files. Evaluation draws a red
marker at the selected point before sending the image or image pair to the
model.

## Run evaluation

Set `CHECKPOINT` to a downloaded SpatioLM checkpoint or a Hugging Face model ID:

```bash
export CHECKPOINT=xiaomi-research/SpatioLM-Perception-InternVL3.5

slm_eval \
  --model spatiolm \
  --model_args "pretrained=${CHECKPOINT},modality=image" \
  --tasks spatiolm_depth_sv,spatiolm_depth_mv,spatiolm_depth_mt \
  --batch_size 1 \
  --log_samples \
  --output_path work_dirs/eval/spatiolm_depth
```

For multi-GPU evaluation:

```bash
accelerate launch --multi_gpu --num_processes 8 -m slm_eval \
  --model spatiolm \
  --model_args "pretrained=${CHECKPOINT},modality=image" \
  --tasks spatiolm_depth_sv,spatiolm_depth_mv,spatiolm_depth_mt \
  --batch_size 1 \
  --log_samples \
  --output_path work_dirs/eval/spatiolm_depth
```

The adapter detects image inputs automatically. `modality=image` is retained as
the fallback for malformed or unsupported visual inputs.

## Task implementation

The task definitions and metrics are located in
`src/slm_eval/tasks/spatiolm_depth/`:

- `spatiolm_sv.yaml`: single-view prompt and split selection.
- `spatiolm_mv.yaml`: paired-view prompt and split selection.
- `spatiolm_mt.yaml`: depth-related numerical reasoning tasks.
- `utils_md.py`: point rendering and metric-depth aggregation.
- `utils_mt.py`: relation prompts, geometric targets, and aggregation.

Other spatial tasks registered by this package are listed in the root
[`README.md`](../../README.md).
