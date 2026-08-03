import importlib.util
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional

from swift.cli.main import get_torchrun_args
from swift.utils import get_logger

# Just for register
# import spatiolm.swift.register

# Like swift: swift/cli/main.py
ROUTE_MAPPING: Dict[str, str] = {
    # "pt": "swift.cli.pt",
    "sft": "swift.cli.sft",
    "sft3d": "spatiolm.cli.sft",
    # "rlhf": "swift.cli.rlhf",
}


def cli_main(route_mapping: Optional[Dict[str, str]] = None) -> None:
    route_mapping = route_mapping or ROUTE_MAPPING
    argv = sys.argv[1:]
    # _compat_web_ui(argv)
    method_name = argv[0].replace("_", "-")
    argv = argv[1:]
    file_path = importlib.util.find_spec(route_mapping[method_name]).origin
    torchrun_args = get_torchrun_args()
    # TODO: support config file
    # prepare_config_args(argv)
    python_cmd = sys.executable
    if torchrun_args is None or method_name not in {
        "pt",
        "sft",
        "sft3d",
        "rlhf",
        "infer",
    }:
        args = [python_cmd, file_path, *argv]
    else:
        args = [
            python_cmd,
            "-m",
            "torch.distributed.run",
            *torchrun_args,
            file_path,
            *argv,
        ]
    print(f"run sh: `{' '.join(args)}`", flush=True)
    result = subprocess.run(args)
    if result.returncode != 0:
        sys.exit(result.returncode)


if __name__ == "__main__":
    cli_main()
