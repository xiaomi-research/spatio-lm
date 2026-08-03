from typing import Literal, Optional

import torch


def make_vision_attn_mask(
    seq_len: int,
    frm_len: Optional[int] = None,
    device=None,
    dtype=None,
    attn_type: Literal["frame", "global"] = "global",
):
    """
    给定输入的总长度 seq_len 和 单帧token数量 frm_len，生成对角mask矩阵[1, 1, seq_len, seq_len]。
    其对角块大小为 frm_len x frm_len，对角块的数量为 seq_len // frm_len。
    """
    if attn_type == "global":
        return torch.zeros(
            (1, 1, seq_len, seq_len),
            device=device,
            dtype=dtype,
        )

    assert frm_len is not None
    assert seq_len % frm_len == 0, "seq_len must be a multiple of frm_len"

    # 计算块数量
    n_blocks = seq_len // frm_len
    # 每个块是 frm_len x frm_len 的全 1
    block = torch.ones(frm_len, frm_len, dtype=torch.uint8, device=device)
    # 用 torch.block_diag 拼成大矩阵
    mask = torch.block_diag(*([block] * n_blocks))

    attn_mask = torch.full(
        (seq_len, seq_len),
        float("-inf"),
        device=device,
        dtype=dtype,
    )
    # 将 mask 部分替换为 0
    attn_mask.masked_fill_(mask == 1, 0)
    # 填充成目标形状 [1, 1, seq_len, seq_len]
    return attn_mask.unsqueeze(0).unsqueeze(0)
