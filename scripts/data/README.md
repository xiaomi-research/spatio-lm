# VLM Training Data Preparation

This directory contains utilities for preparing image/video instruction data for
SpatioLM training. The workflow is compatible with VSI-590K-style annotations,
but does not depend on any private server path or private dataset layout.

## Download VSI-590K

The public VSI-590K dataset is hosted as `nyu-visionx/VSI-590K` on Hugging
Face. Install the Hugging Face Hub CLI first if needed:

```bash
pip install -U huggingface_hub
```

Download the annotation file first to inspect the data format:

```bash
hf download nyu-visionx/VSI-590K \
  --repo-type dataset \
  --include vsi_590k.jsonl \
  --local-dir data/raw/VSI-590K
```

The annotation file references images and videos stored in source archives.
Download all archives when preparing the complete training set:

```bash
hf download nyu-visionx/VSI-590K \
  --repo-type dataset \
  --include vsi_590k.jsonl \
  --include '*.tar.gz' \
  --local-dir data/raw/VSI-590K
```

The complete download is very large, so selective downloads are recommended
when only part of the data is needed:

```bash
hf download nyu-visionx/VSI-590K \
  --repo-type dataset \
  --include vsi_590k.jsonl \
  --include scannet.tar.gz \
  --include scannetppv2.tar.gz \
  --local-dir data/raw/VSI-590K
```

If direct access is unavailable, configure a proxy and rerun the same command:

```bash
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export ALL_PROXY=http://127.0.0.1:7890
```

Download archives should be extracted under the same dataset root so the
relative media paths in the annotation file remain resolvable:

```bash
find data/raw/VSI-590K -maxdepth 1 -name '*.tar.gz' -print0 \
  | xargs -0 -n1 tar -xzf - -C data/raw/VSI-590K
```

After extraction, validate and convert the annotations with the command below.

## Recommended layout

Keep annotation files and media under one portable dataset directory:

```text
data/raw/VSI-590K/
├── annotations/
│   └── train.jsonl
├── images/
│   └── sample-000001.jpg
└── videos/
    └── sample-000001.mp4
```

Use relative paths in annotations whenever possible:

```json
{
  "messages": [
    {"role": "user", "content": "<image>Which object is closer?"},
    {"role": "assistant", "content": "The chair."}
  ],
  "images": ["images/sample-000001.jpg"]
}
```

VSI-590K source records commonly use `conversations`, singular `image`/`video`,
and `question_type` fields:

```json
{
  "conversations": [
    {"from": "human", "value": "<image>\\nWhich object is closer to the camera?"},
    {"from": "gpt", "value": "The chair."}
  ],
  "question_type": "relative_distance",
  "image": "images/sample-000001.jpg"
}
```

The converter produces an MS-SWIFT-compatible record and preserves
`question_type`:

```json
{
  "messages": [
    {"role": "user", "content": "<image>\\nWhich object is closer to the camera?"},
    {"role": "assistant", "content": "The chair."}
  ],
  "images": ["images/sample-000001.jpg"],
  "question_type": "relative_distance"
}
```

For a video sample, replace the media field and placeholder with `video`:

```json
{
  "conversations": [
    {"from": "human", "value": "<video>\\nDescribe the spatial relationship."},
    {"from": "gpt", "value": "The table is to the left of the chair."}
  ],
  "video": "videos/room-tour.mp4"
}
```

The script also accepts standard `messages` and simple `question`/`answer`
records. Each record must contain at least one user message, one assistant
message, and one image or video. Media path counts must match the corresponding
`<image>` and `<video>` placeholder counts.

## Convert and validate

For a directory input, the media root defaults to the input directory. No
machine-specific path needs to be configured:

```bash
python scripts/data/prepare_vlm_data.py \
  --input data/raw/VSI-590K \
  --output data/train-vlm/vlm-3d/VSI-590K
```

For a single annotation file, relative media paths default to that file's
parent directory:

```bash
python scripts/data/prepare_vlm_data.py \
  --input data/raw/VSI-590K/annotations/train.jsonl \
  --output data/train-vlm/vlm-3d/VSI-590K/train.jsonl
```

Use `--media-root` only when media are stored outside the annotation directory:

```bash
python scripts/data/prepare_vlm_data.py \
  --input data/raw/VSI-590K/annotations/train.jsonl \
  --media-root data/raw/VSI-590K \
  --output data/train-vlm/vlm-3d/VSI-590K/train.jsonl
```

Before training, validate an existing JSONL without writing a new file:

```bash
python scripts/data/prepare_vlm_data.py \
  --input data/train-vlm/vlm-3d/VSI-590K \
  --check-only
```

The command checks JSON syntax, supported roles, non-empty conversations,
media existence, and the exact number of `<image>`/`<video>` placeholders.

## Combine and sample

Pass `--input` more than once to combine independent training sources. Do not
pass `VSI-Bench`, `ScanQA`, or any other evaluation split here: evaluation
samples must remain isolated from training data to avoid contamination or data
leakage. The output keeps the original record order unless sampling is
requested:

```bash
python scripts/data/prepare_vlm_data.py \
  --input data/raw/VSI-590K \
  --input data/raw/other-train-set \
  --output data/train-vlm/vlm-3d/mixed.jsonl \
  --deduplicate \
  --sample 100000 \
  --seed 42
```

Sampling is performed after normalization and validation, and `--seed` makes it
reproducible.

## Training

Use the generated file with either training entry point:

```bash
swift sft --dataset data/train-vlm/vlm-3d/VSI-590K/prepared.jsonl ...
```

```bash
spatiolm sft3d --dataset data/train-vlm/vlm-3d/VSI-590K/prepared.jsonl ...
```

The converter only prepares existing instruction annotations. It does not
generate new spatial questions from raw 3D geometry; use the registered
`cvlm3d/depthlm` dataset for RGB/depth/camera-supervised data described in the
main README.

## Exit status

- `0`: all records are valid; conversion or validation completed successfully.
- `1`: one or more records failed validation; details are printed to stderr.
- `2`: command-line or input configuration error.
