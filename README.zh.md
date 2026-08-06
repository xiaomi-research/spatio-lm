# SpatioLM：迈向视觉语言模型的通用物理空间智能

<p align="center">
  <a href="https://openreview.net/forum?id=CHavqrN1X9"><img src="https://img.shields.io/badge/Paper-OpenReview-b31b1b?logo=openreview&amp;logoColor=white" alt="Paper: OpenReview"></a>
  <a href="https://arxiv.org/abs/2608.01899"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b?logo=arxiv&amp;logoColor=red" alt="Paper: arXiv"></a>
  <a href="https://icml.cc/virtual/2026/poster/65576"><img src="https://img.shields.io/badge/ICML_2026-Oral-4c78a8" alt="ICML 2026 Oral"></a>
  <a href="https://huggingface.co/collections/xiaomi-research/spatiolm"><img src="https://img.shields.io/badge/Model_Weights-Hugging_Face-ffd21e?logo=huggingface&amp;logoColor=yellow" alt="Model Weights: Hugging Face"></a>
  <a href="https://xiaomi-research.github.io/spatio-lm/"><img src="https://img.shields.io/badge/Project_Page-Website-ff6900" alt="Project Page"></a>
</p>

> **SpatioLM** 官方实现，论文已被第四十三届国际机器学习大会（**ICML 2026**）接收为**口头报告（Oral）**。 <a href="README.md"><img src="https://img.shields.io/badge/Language-English-blue" alt="English"></a> <a href="README.zh.md"><img src="https://img.shields.io/badge/语言-简体中文-red" alt="简体中文"></a>

SpatioLM 是一个用于提升视觉语言模型（VLM）空间智能的参数高效框架。它引入即插即用的空间视觉模块，并通过伪深度和相机监督学习符合物理规律的表征，而在推理阶段无需额外的 3D 输入。同一框架可支持空间感知、空间推理和具身操作任务。

![teaser](docs/assets/teaser.png)

## ✨ 核心亮点

- **非侵入式空间增强：** 在提升空间推理能力的同时，保留基础 VLM 的通用能力。
- **3D 感知监督：** 训练期间从 Depth Anything 3 教师模型中蒸馏 token、深度和相机射线知识。
- **图像与视频支持：** 支持单图、多图以及采样后的视频帧。
- **强大的空间推理能力：** 在 VSI-Bench 上取得 71.6 分，是该基准首个公开超过 70 分的结果。
- **统一训练栈：** 基于 [MS-SWIFT](https://github.com/modelscope/ms-swift) 扩展，并提供专用的 `spatiolm sft3d` 入口。
- **统一评测栈：** 基于 [LMMs-Eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) 扩展，支持空间基准和 OpenAI 兼容 API 评测。

![SpatioLM](docs/assets/framework.png)

## 📦 发布状态

本仓库包含模型实现、3D 蒸馏训练器、数据加载器和评测任务。官方检查点已发布在 Hugging Face 的 [SpatioLM Collection](https://huggingface.co/collections/xiaomi-research/spatiolm) 中。基准数据集不存储在 Git 中，请单独准备，并通过下述命令传入本地路径。

## 🤗 模型库

已发布的检查点覆盖三类互补能力：**Understanding** 面向空间推理和场景理解，**Perception** 面向度量深度和物理空间感知，**Action** 面向使用离散 VLA 动作的具身操作。Understanding 和 Perception 两条路线均提供两个 8B 视觉语言骨干版本。

| 检查点 | 能力 | 骨干模型 | 适用场景 |
| --- | --- | --- | --- |
| [SpatioLM-Understanding-InternVL3.5](https://huggingface.co/xiaomi-research/SpatioLM-Understanding-InternVL3.5) | Understanding | InternVL3.5 8B | 空间推理和场景理解 |
| [SpatioLM-Understanding-SenseNovaSI](https://huggingface.co/xiaomi-research/SpatioLM-Understanding-SenseNovaSI) | Understanding | SenseNova-SI 1.1 / InternVL3 8B | 空间推理和场景理解 |
| [SpatioLM-Perception-InternVL3.5](https://huggingface.co/xiaomi-research/SpatioLM-Perception-InternVL3.5) | Perception | InternVL3.5 8B | 度量深度和物理空间感知 |
| [SpatioLM-Perception-SenseNovaSI](https://huggingface.co/xiaomi-research/SpatioLM-Perception-SenseNovaSI) | Perception | SenseNova-SI 1.1 / InternVL3 8B | 度量深度和物理空间感知 |
| [SpatioLM-Action-VLA0](https://huggingface.co/xiaomi-research/SpatioLM-Action-VLA0) | Action | SenseNova-SI 1.1 / InternVL3 2B | 使用离散 VLA 动作的具身操作 |

所有检查点均使用本仓库中的 SpatioLM 模型实现，推理时无需额外的 3D 输入。通用空间问答请使用 Understanding 检查点，以度量几何为核心的任务请使用 Perception 检查点，具身控制请使用 Action 检查点。

## 🛠️ 安装

<details>
<summary><strong>环境要求</strong></summary>

- Linux
- 推荐使用 Python 3.10
- 支持 CUDA 的 NVIDIA GPU
- 编译 FlashAttention 时需要 CUDA Toolkit

代码已在以下软件栈中完成测试：

| 软件包 | 测试版本 |
| --- | --- |
| Python | 3.10 |
| PyTorch | 2.6.0 + CUDA 12.4 |
| torchvision | 0.21.0 |
| Transformers | 4.57.3 |
| MS-SWIFT | 3.12.1 |
| LMMs-Eval | 0.4.0 |
| Accelerate | 1.12.0 |
| DeepSpeed | 0.18.4 |
| FlashAttention | 2.7.4.post1 |

</details>

<details>
<summary><strong>创建环境</strong></summary>

在仓库根目录运行以下命令。如果你的 CUDA 运行时与 CUDA 12.4 不兼容，请更换 PyTorch wheel 索引。

```bash
conda create -n spatiolm python=3.10 -y
conda activate spatiolm

pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124

pip install \
  "transformers>=4.56.1,<5" \
  "accelerate>=1.10,<2" \
  "deepspeed>=0.14,<0.20" \
  "ms-swift[all]>=3.8,<4" \
  "lmms-eval>=0.4,<1" \
  "qwen-vl-utils<0.0.12" \
  decord h5py einops

pip install "flash-attn<2.8" --no-build-isolation
pip install -e .
```

训练时推荐使用 FlashAttention，但基础推理并不依赖它。如果系统无法完成编译，可跳过安装，并在训练命令中将 `--attn_impl flash_attention_2` 替换为 `--attn_impl sdpa`。

验证安装：

```bash
spatiolm sft3d --help
slm_eval --help
```

</details>

## 🗂️ 模型与数据准备

典型的本地目录结构如下：

```text
data/
├── ckpts/
│   ├── spatiolm-init-or-checkpoint/
│   └── DA3-Large/
├── train/
│   └── spatial_train.jsonl
├── train-v3r/                  # Optional raw DepthLM data
│   └── scannet/
└── eval/
    ├── DA-2K/
    ├── site_bench/
    └── spatiolm_depth/
work_dirs/
```

`MODEL_PATH` 必须指向与 SpatioLM 兼容的 InternVL 检查点。进行 3D 蒸馏时，其配置必须包含 SpatioLM 视觉条件模块和 DPT Head 配置。`TEACHER3D_PATH` 必须指向可由 `AutoModelForDepthEstimation` 加载、兼容 Hugging Face 格式的 Depth Anything 3 教师检查点。

> [!NOTE]
> SpatioLM 的初始化检查点和教师检查点来源于以下公开模型：
>
> | 用途 | 公开源模型 |
> | --- | --- |
> | InternVL3.5 初始化 | [`OpenGVLab/InternVL3_5-8B`](https://huggingface.co/OpenGVLab/InternVL3_5-8B) |
> | SenseNova-SI 初始化 | [`sensenova/SenseNova-SI-1.1-InternVL3-8B`](https://huggingface.co/sensenova/SenseNova-SI-1.1-InternVL3-8B) |
> | VLA0 初始化 | [`sensenova/SenseNova-SI-1.1-InternVL3-8B`](https://huggingface.co/sensenova/SenseNova-SI-1.1-InternVL3-8B) |
> | 深度教师模型来源 | [`depth-anything/DA3-LARGE`](https://huggingface.co/depth-anything/DA3-LARGE) |
>
> 前三项表示用于构建相应初始化检查点的上游 VLM 权重，不能直接替代 `MODEL_PATH`。运行 `spatiolm sft3d` 前，请准备兼容 SpatioLM 的检查点，并确保其 `config.json` 包含所需的 `vision_condition_config` 和 `vision_dpt_config` 字段。同样，公开 DA3 仓库采用官方 Depth Anything 3 检查点结构，而当前 `TEACHER3D_PATH` 需要使用项目转换后的 Hugging Face 格式：注册为 `DA3Model`，并可通过 `AutoModelForDepthEstimation` 加载。

<details>
<summary><strong>监督训练 JSONL 格式</strong></summary>

训练数据遵循 MS-SWIFT 多模态 JSONL 格式。每一行包含一段对话，以及 `images` 或 `videos` 字段：

```json
{"messages":[{"role":"user","content":"<image>Which marked point is closer to the camera?"},{"role":"assistant","content":"A"}],"images":["/path/to/image.jpg"]}
```

```json
{"messages":[{"role":"user","content":"<video>Describe the spatial relationship between the objects."},{"role":"assistant","content":"The chair is to the left of the table."}],"videos":["/path/to/video.mp4"]}
```

媒体路径可以是绝对路径，也可以是相对于训练启动目录的路径。`<image>` 或 `<video>` 占位符的数量必须与提供的媒体数量一致。

准备 VSI-590K 类指令数据时，可以使用 [`scripts/data/README.md`](scripts/data/README.md) 中的脚本和说明。脚本默认将相对媒体路径解析到 `--input` 目录，不依赖远程机器路径，并会在训练前检查 JSON、媒体文件和占位符数量。

```bash
python scripts/data/prepare_vlm_data.py \
  --input data/raw/VSI-590K \
  --output data/train-vlm/vlm-3d/VSI-590K
```

同一份说明还包含公开 VSI-590K 标注和媒体归档的下载方法，包括按需下载以及 HTTP 代理配置。

<details>
<summary><strong>下载 VSI-590K</strong></summary>

先安装 Hugging Face Hub CLI：

```bash
pip install -U huggingface_hub
```

只下载标注文件：

```bash
hf download nyu-visionx/VSI-590K \
  --repo-type dataset \
  --include vsi_590k.jsonl \
  --local-dir data/raw/VSI-590K
```

下载标注文件和全部媒体归档：

```bash
hf download nyu-visionx/VSI-590K \
  --repo-type dataset \
  --include vsi_590k.jsonl \
  --include '*.tar.gz' \
  --local-dir data/raw/VSI-590K
```

只下载指定数据源：

```bash
hf download nyu-visionx/VSI-590K \
  --repo-type dataset \
  --include vsi_590k.jsonl \
  --include scannet.tar.gz \
  --include scannetppv2.tar.gz \
  --local-dir data/raw/VSI-590K
```

如果无法直连 Hugging Face，可以先设置代理：

```bash
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export ALL_PROXY=http://127.0.0.1:7890
```

然后执行上面的任意下载命令。下载完成后，将归档解压到同一个数据集目录：

```bash
find data/raw/VSI-590K -maxdepth 1 -name '*.tar.gz' -print0 \
  | xargs -0 -n1 tar -xzf - -C data/raw/VSI-590K
```

</details>

</details>

<details>
<summary><strong>可选的 DepthLM 原始数据</strong></summary>

内置 DepthLM 数据集根据 RGB 视频、度量深度、相机内参和相机位姿生成空间问题。每个视频都必须配有同名的 HDF5 文件：

```text
data/train-v3r/scannet/
├── scene0000_00.mp4
└── scene0000_00.h5
```

HDF5 文件必须包含：

| 键 | 形状 | 说明 |
| --- | --- | --- |
| `depth` | `(T, H, W)` | 度量深度图；数据集还会读取其 `invalid` 属性 |
| `intrinsic` | `(T, 3, 3)` | 相机内参矩阵 |
| `pos` | `(T, 4, 4)` | 相机到世界坐标系的位姿 |

为保持向后兼容，注册的数据集标识符仍为 `cvlm3d/depthlm`。例如：

```bash
DEPTH_DATASET='torch::cvlm3d/depthlm::data_root="data/train-v3r",video_folders=["scannet"]'
```

在标识符后追加 `#N`，可采样或重复至 `N` 个样本，例如 `torch::cvlm3d/depthlm#1000::...`。

</details>

## 🚀 训练

<details>
<summary><strong>3D 监督训练</strong></summary>

主训练入口将语言模型损失与 token、深度和相机射线蒸馏损失结合。

```bash
export MODEL_PATH=/path/to/spatiolm-init
export TEACHER3D_PATH=/path/to/DA3-Large
export TRAIN_JSONL=/path/to/spatial_train.jsonl
export OUTPUT_DIR=work_dirs/spatiolm-sft3d

export VIDEO_SEGMENTS=12
export LOSS_LM_WEIGHT=0.6
export LOSS_TOKEN_WEIGHT=0.2
export LOSS_DEPTH_WEIGHT=0.1
export LOSS_RAY_WEIGHT=0.1
export DPT_LR_SCALE=10.0

NPROC_PER_NODE=8 spatiolm sft3d \
  --model "$MODEL_PATH" \
  --teacher3d "$TEACHER3D_PATH" \
  --use_hf true \
  --train_type full \
  --freeze_parameters_ratio 1 \
  --trainable_parameters_regex "^(language_model.*(vision_3r.*|dpt_head)|condition_proj)" \
  --torch_dtype bfloat16 \
  --dataset "$TRAIN_JSONL" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 2 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.05 \
  --logging_steps 5 \
  --save_strategy epoch \
  --save_total_limit 2 \
  --save_only_model true \
  --remove_unused_columns false \
  --ddp_find_unused_parameters false \
  --split_dataset_ratio 0 \
  --deepspeed zero2 \
  --attn_impl flash_attention_2 \
  --output_dir "$OUTPUT_DIR"
```

单卡训练时设置 `NPROC_PER_NODE=1`。多机任务还需设置 MS-SWIFT 支持的标准变量 `NNODES`、`NODE_RANK`、`MASTER_ADDR` 和 `MASTER_PORT`。

如需混合常规 JSONL 数据和生成式 DepthLM 数据集，请将上述命令中的 `--dataset` 行替换为：

```bash
--dataset "$TRAIN_JSONL" "$DEPTH_DATASET"
```

</details>

<details>
<summary><strong>标准监督微调</strong></summary>

不需要 3D 教师蒸馏时，可直接使用 MS-SWIFT。自定义注册文件会启用 SpatioLM 模型和模板定义。

```bash
swift sft \
  --custom_register_path src/spatiolm/swift/register.py \
  --model "$MODEL_PATH" \
  --use_hf true \
  --train_type full \
  --freeze_parameters_ratio 1 \
  --trainable_parameters_regex "^(language_model.*vision_3r.*|condition_proj)" \
  --torch_dtype bfloat16 \
  --dataset "$TRAIN_JSONL" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 2 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.05 \
  --logging_steps 5 \
  --save_strategy epoch \
  --save_only_model true \
  --split_dataset_ratio 0 \
  --attn_impl flash_attention_2 \
  --output_dir work_dirs/spatiolm-sft
```

有效全局批大小为 `per_device_train_batch_size × gradient_accumulation_steps × GPU 数量 × 节点数量`。

</details>

## 🔍 推理

Understanding 和 Perception 检查点共用相同的图像推理接口。将 `checkpoint` 设置为上表中的 Hugging Face 模型 ID，或已下载的本地路径：

<details>
<summary><strong>图像推理示例</strong></summary>

```python
import torch
from lmms_eval.models.simple.internvl2 import load_image
from PIL import Image
from transformers import AutoTokenizer

from spatiolm.models import InternVL3RChatModel

checkpoint = "xiaomi-research/SpatioLM-Understanding-InternVL3.5"
image_path = "/path/to/image.jpg"

model = InternVL3RChatModel.from_pretrained(
    checkpoint,
    dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(
    checkpoint,
    trust_remote_code=True,
    use_fast=False,
)

image = Image.open(image_path).convert("RGB")
pixel_values = load_image(image, input_size=448).to(
    device="cuda",
    dtype=torch.bfloat16,
)
answer = model.chat(
    tokenizer,
    pixel_values,
    "Which object is closer to the camera?",
    {"max_new_tokens": 128, "do_sample": False},
)
print(answer)
```

</details>

视频推理和基准批量执行请使用下述 `slm_eval` 接口。通过 `VIDEO_SEGMENTS` 控制采样帧数。

## 📊 评测

`slm_eval` 保留 LMMs-Eval CLI，并自动注册额外的 SpatioLM 模型和任务。

以 `load_from_disk` 目录结构下载 SpatioLM 深度基准：

```bash
hf download edatai/spatiolm-depth \
  --repo-type dataset \
  --local-dir data/eval/spatiolm_depth
```

有关深度基准的数据划分、指标和评测命令，请参阅 [`src/slm_eval/README.md`](src/slm_eval/README.md)。

<details>
<summary><strong>评测本地检查点</strong></summary>

```bash
export CHECKPOINT=/path/to/spatiolm-checkpoint
export VIDEO_SEGMENTS=48

slm_eval \
  --model spatiolm \
  --model_args "pretrained=${CHECKPOINT},modality=video" \
  --tasks site_bench_video \
  --batch_size 1 \
  --log_samples \
  --output_path work_dirs/eval/site_bench_video
```

多 GPU 评测：

```bash
accelerate launch --multi_gpu --num_processes 8 -m slm_eval \
  --model spatiolm \
  --model_args "pretrained=${CHECKPOINT},modality=video" \
  --tasks site_bench_video \
  --batch_size 1 \
  --log_samples \
  --output_path work_dirs/eval/site_bench_video
```

</details>

<details>
<summary><strong>评测 OpenAI 兼容 API</strong></summary>

任何实现 OpenAI Chat Completions 接口的端点，都可以通过并行 API 适配器进行评测：

```bash
export OPENAI_API_BASE=https://your-endpoint.example.com/v1
export OPENAI_API_KEY=your-api-key

slm_eval \
  --model parallel_openai_compatible \
  --model_args "model_version=your-model-name,max_workers=16,max_num_frames=16" \
  --tasks cvbench \
  --batch_size 1 \
  --log_samples \
  --output_path work_dirs/eval/api
```

对于 Azure OpenAI，请在 `--model_args` 中设置 `azure_openai=true`，并导出 `AZURE_OPENAI_API_KEY`、`AZURE_OPENAI_API_BASE` 和 `AZURE_OPENAI_API_VERSION`。

</details>

### 已包含的空间任务

本仓库在标准 LMMs-Eval 任务目录之上新增了以下任务配置：

| 数据来源 | 任务 |
| --- | --- |
| Hugging Face 数据集 | `blink`, `cvbench`, `embspatialbench` |
| 本地基准数据 | `da2k`, `mindcube_tiny`, `my_mmsi_bench`, `scanqa`, `site_bench_image`, `site_bench_video`, `sqa3d`, `viewspatialbench`, `myvsibench` |
| SpatioLM 深度基准 | `spatiolm_depth_sv`, `spatiolm_depth_mv`, `spatiolm_depth_mt` |

列出所有已注册任务：

```bash
slm_eval --tasks list
```

依赖本地数据的任务会从 `src/slm_eval/tasks/*/*.yaml` 读取路径。请将处理后的 Hugging Face 数据集放到配置的 `dataset_path`，或在评测前将该字段更新为你的本地路径。

## 🧱 仓库结构

```text
src/spatiolm/
├── cli/          # 兼容 MS-SWIFT 的训练入口
├── datasets/     # RGB、深度、相机及生成式空间问答数据集
├── losses/       # Token、深度和射线蒸馏损失
├── models/       # SpatioLM、Depth Anything 3、InternVL 和 VGGT 组件
├── swift/        # 模型/模板注册与 3D 训练器
└── templates/    # InternVL 和 Qwen-VL 多模态模板
src/slm_eval/
├── simple/       # 本地检查点和 OpenAI 兼容模型适配器
└── tasks/        # 空间基准定义与指标
```

## 🔧 常见问题

- **CUDA 显存不足：** 减小 `VIDEO_SEGMENTS` 或 `per_device_train_batch_size`，再增大 `gradient_accumulation_steps` 以保持全局批大小。
- **FlashAttention 编译或导入失败：** 不使用 FlashAttention，并改用 `--attn_impl sdpa`。
- **找不到数据集：** 检查 `src/slm_eval/tasks` 下的任务 YAML；当设置 `load_from_disk: true` 时，请确保 `dataset_path` 指向通过 Hugging Face `save_to_disk` 保存的数据集。
- **检查点配置错误：** 确认检查点是兼容 SpatioLM 的 InternVL 模型，并且其 `config.json` 包含所选训练模式需要的视觉条件模块和 DPT Head 配置。

## 🙏 致谢

本工作基于并受益于多个优秀的开源项目，包括 [MS-SWIFT](https://github.com/modelscope/ms-swift)、[Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3)、[SenseNova-SI](https://github.com/OpenSenseNova/SenseNova-SI)、[InternVL](https://github.com/OpenGVLab/InternVL)、[VGGT](https://github.com/facebookresearch/vggt) 和 [LMMs-Eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)。我们衷心感谢这些项目的作者和贡献者与社区分享成果。

## 📝 引用

如果本工作对你有所帮助，请引用：

```bibtex
@inproceedings{wu2026spatiolm,
  title={SpatioLM: Towards General Physical Spatial Intelligence in Vision-Language Models},
  author={Wu, Jing and Wu, Jianhua and Guan, Jiayi and Chen, Jiahong and Lu, Jinghui and Ye, Hangjun and Gao, Bingzhao and Chen, Long},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026},
  note={To appear},
  eprint={2608.01899},
  archivePrefix={arXiv},
  url={https://arxiv.org/abs/2608.01899}
}
```
