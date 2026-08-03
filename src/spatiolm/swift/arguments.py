from dataclasses import dataclass, field
from typing import Optional

from swift.llm import TrainArguments


@dataclass
class Training3DArguments(TrainArguments):
    teacher3d: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the teacher 3D model name or repo_id"},
    )
