import argparse
import importlib
import sys
from pathlib import Path

from lmms_eval.api.registry import MODEL_REGISTRY
from lmms_eval.models import AVAILABLE_SIMPLE_MODELS


def setup_env():
    ROOT_DIR = Path(__file__).parent.resolve()

    # Register simple models
    sys.path.append(str(ROOT_DIR))
    for model_file in (ROOT_DIR / "simple").rglob("*.py"):
        if model_file.name != "__init__.py":
            module_name = ".".join(
                model_file.relative_to(ROOT_DIR).with_suffix("").parts
            )
            importlib.import_module(module_name)

    for model_name, model_cls in MODEL_REGISTRY.items():
        AVAILABLE_SIMPLE_MODELS[model_name] = (
            f"{model_cls.__module__}.{model_cls.__name__}"
        )

    return argparse.Namespace(
        # Set default tasks directory
        include_path=str(ROOT_DIR / "tasks"),
        # Add any other default arguments here ...
    )


def cli_main():
    from lmms_eval.__main__ import cli_evaluate

    cli_evaluate(setup_env())


if __name__ == "__main__":
    cli_main()
