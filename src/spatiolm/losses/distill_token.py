from typing import Sequence, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


def matrix_similarity(
    student: torch.Tensor,
    teacher: torch.Tensor,
    normalize=True,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
    if student.ndim > 3:
        student = student.view(student.size(0), -1, student.size(-1))
    if teacher.ndim > 3:
        teacher = teacher.view(teacher.size(0), -1, teacher.size(-1))

    if normalize:
        student = F.normalize(student, dim=-1)  # [B, N, C1]
        teacher = F.normalize(teacher, dim=-1)  # [B, N, C2]

    S = student @ student.transpose(1, 2)  # [B, N, N]
    T = teacher @ teacher.transpose(1, 2)  # [B, N, N]

    mask = ~torch.eye(S.size(1), dtype=torch.bool, device=S.device)
    mask = mask[None].repeat(S.size(0), 1, 1)

    sim = F.l1_loss(S[mask], T[mask], reduction=reduction)

    return sim


class DistillTokenLoss(nn.Module):
    def __init__(
        self,
        normalize: bool = True,
        reduction: Literal["mean", "sum"] = "mean",
    ):
        super().__init__()

        self.normalize = normalize
        self.reduction = reduction

    def forward(
        self,
        preds: Sequence[torch.Tensor],
        trgts: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        if isinstance(preds, torch.Tensor):
            preds = [preds]
        if isinstance(trgts, torch.Tensor):
            trgts = [trgts]

        assert len(preds) == len(trgts), (
            "Prediction and reference must have the same length"
        )

        losses = [
            matrix_similarity(
                pred,
                tgt,
                self.normalize,
                self.reduction,
            )
            for pred, tgt in zip(preds, trgts)
        ]

        return sum(losses) / len(losses)
