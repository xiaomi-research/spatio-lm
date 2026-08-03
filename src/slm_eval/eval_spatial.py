import os
from pathlib import Path

import click
from lmms_eval.evaluator import simple_evaluate
from lmms_eval.tasks import TaskManager


def create_model(model_name: str, model_type: str, **kwargs):
    """Create model based on type and configuration.

    Args:
        pretrain: Model path or name
        model_type: Type of model to create
        **kwargs: Additional model parameters (modality, num_frames, etc.)
    """
    if model_type == "openai_api":
        from slm_eval.simple.parallel_openai_api import (
            ParallelOpenAICompatible as OpenAICompatible,
        )

        assert "OPENAI_API_KEY" in os.environ, "OPENAI_API_KEY must be set"
        assert "OPENAI_API_BASE" in os.environ, "OPENAI_API_BASE must be set"
        print(f"User API BASE: {os.environ['OPENAI_API_BASE']}")
        print(f"User API KEY: {os.environ['OPENAI_API_KEY']}")

        default_headers = None
        if "gpt" in model_name:
            default_headers = {"X-Model-Provider-Id": "azure_openai"}
        elif "doubao" in model_name:
            default_headers = {"X-Model-Provider-Id": "volcengine_maas"}
        elif "qwen3-max" in model_name or "qwen3-vl-plus" in model_name:
            default_headers = {"X-Model-Provider-Id": "tongyi"}
        elif "gemini" in model_name:
            default_headers = {"X-Model-Provider-Id": "vertex_ai"}

        model_kwgs = dict(
            max_workers=kwargs.pop("max_workers", 12),
            max_num_frames=kwargs.pop("num_frames", 16),
            default_headers=kwargs.pop("default_headers", default_headers),
            **kwargs,
        )
        return OpenAICompatible(model_version=model_name, **model_kwgs)

    elif model_type == "internvl" or "internvl" in model_name.lower():
        # Just register custom model to Transformers
        from spatiolm.models import InternVL3RChatModel  # noqa: E402
        from lmms_eval.models.simple.internvl2 import InternVL2

        model_kwgs = dict(
            modality=kwargs.pop("modality", "image"),
            num_frame=kwargs.pop("num_frames", 16),
            **kwargs,
        )
        return InternVL2(pretrained=model_name, **model_kwgs)

    elif model_type == "qwen2_5_vl" or "qwen2.5-vl" in model_name.lower():
        from lmms_eval.models.simple.qwen2_5_vl import Qwen2_5_VL

        model_kwgs = dict(max_num_frames=kwargs.pop("num_frames", 16), **kwargs)
        return Qwen2_5_VL(pretrained=model_name, **model_kwgs)

    elif model_type == "llava_hf" or "llava-hf" in model_name.lower():
        from lmms_eval.models.simple.llava_hf import LlavaHf

        return LlavaHf(pretrained=model_name, dtype="bfloat16")

    else:
        raise ValueError(
            f"Unspported model type: {model_type} or model name: {model_name}"
        )


@click.command()
@click.option(
    "-m",
    "--model_name_or_path",
    default="data/ckpts/public/InternVL3_5-1B",
    help="pretrained model path",
)
@click.option(
    "-mt",
    "--model_type",
    required=False,
    type=click.Choice(
        [
            "internvl",
            "qwen2_5_vl",
            "qwen3_vl",
            "llava_hf",
            "openai_api",
        ]
    ),
    help="model type",
)
@click.option(
    "-ma",
    "--model_args",
    type=str,
    required=False,
    help="model arguments",
)
@click.option(
    "-t",
    "--tasks",
    type=click.Choice(
        [
            "myvsibench",
            "sqa3d",
            "scanqa",
            "my_mmsi_bench",
            "cvbench",
            "blink",
            "embspatialbench",
            "da2k",  # Relative Depth QA
            "spatiolm_depth_sv",  # Metric Depth QA
            "spatiolm_depth_mv",  # Metric Depth QA
            "spatiolm_depth_mt",  # Metric Depth QA
            "site_bench_image",
            "site_bench_video",
        ]
    ),
    default=["myvsibench", "sqa3d", "scanqa"],
    multiple=True,
    help="tasks to evaluate",
)
@click.option(
    "--num_frames",
    default=16,
    envvar="VIDEO_SEGMENTS",
    help="pretrained model path",
)
@click.option(
    "-n",
    "--eval_samples",
    required=False,
    type=int,
    envvar="EVAL_SAMPLES",
    help="number of samples to generate",
)
@click.option(
    "--save_file",
    required=False,
    type=str,
    envvar="LMMS_EVAL_SAVE_FILE",
    help="file to save results",
)
def cli_main(
    model_name_or_path,
    model_type,
    model_args,
    tasks,
    num_frames,
    eval_samples,
    save_file,
):
    model_kwargs = eval(f"dict({model_args})") if model_args else {}
    model_kwargs.setdefault("num_frames", num_frames)
    model = create_model(model_name_or_path, model_type, **model_kwargs)

    task_path = Path(__file__).parent / "tasks"
    task_manager = TaskManager(include_path=str(task_path))

    results = simple_evaluate(
        model=model,
        tasks=list(tasks),
        # num_fewshot=0,
        task_manager=task_manager,
        limit=eval_samples,
    )

    if results is not None and "results" in results:
        print(results["results"])

        if save_file:
            import json

            with open(f"{save_file}.eval.json", "a") as f:
                json.dump(results["results"], f)


if __name__ == "__main__":
    import sys

    sys.path.append(str(Path(__file__).parent))
    cli_main()
