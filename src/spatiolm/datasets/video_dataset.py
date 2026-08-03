import os
import random
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Dict, List, Literal, Tuple, Union, Optional

import decord
import h5py
import numpy as np
import torch
import torchvision.transforms.v2 as tvf
from torch.utils.data import Dataset

from .pipelines.depth_camera import (
    CenterCropWithCamera,
    RandomResizedCropWithCamera,
    ResizeWithCamera,
)

__all__ = ["VideoDepthCameraDataset"]


def depthmaps_to_pts3d(depthmaps, cam_intrinsics, cam_poses):
    """
    Convert depthmaps to 3D points in world coordinates.

    Args:
        depthmaps: torch.Tensor of shape (T, H, W) - depth maps for each frame
        cam_intrinsics: torch.Tensor of shape (T, 3, 3) - camera intrinsic matrices
        cam_poses: torch.Tensor of shape (T, 4, 4) - camera poses (camera to world)

    Returns:
        pts3d: torch.Tensor of shape (T, H, W, 3) - 3D points in world coordinates
        valid_mask: torch.Tensor of shape (T, H, W) - boolean mask for valid points
    """
    T, H, W = depthmaps.shape

    # Create pixel coordinate grid
    u, v = torch.meshgrid(
        torch.arange(W, device=depthmaps.device, dtype=depthmaps.dtype),
        torch.arange(H, device=depthmaps.device, dtype=depthmaps.dtype),
        indexing="xy",
    )
    u = u.unsqueeze(0).expand(T, -1, -1)  # (T, H, W)
    v = v.unsqueeze(0).expand(T, -1, -1)  # (T, H, W)

    # Extract camera parameters
    fx = cam_intrinsics[:, 0, 0].view(T, 1, 1)  # (T, 1, 1)
    fy = cam_intrinsics[:, 1, 1].view(T, 1, 1)  # (T, 1, 1)
    cx = cam_intrinsics[:, 0, 2].view(T, 1, 1)  # (T, 1, 1)
    cy = cam_intrinsics[:, 1, 2].view(T, 1, 1)  # (T, 1, 1)

    # Convert to camera coordinates
    z_cam = depthmaps
    x_cam = (u - cx) * z_cam / fx
    y_cam = (v - cy) * z_cam / fy

    # Stack to get camera coordinates
    pts_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)  # (T, H, W, 3)

    # Transform to world coordinates
    R = cam_poses[:, :3, :3]  # (T, 3, 3)
    t = cam_poses[:, :3, 3]  # (T, 3)

    # Apply rotation and translation
    pts_world = torch.einsum("tij,tvwj->tvwi", R, pts_cam) + t.unsqueeze(1).unsqueeze(1)

    return pts_world


def repeat_sequence(seq: List, length: int):
    """Repeats a sequence to match a specified length.

    Args:
        seq: The sequence to repeat.
        length: The desired length of the repeated sequence.

    Returns:
        A list with the repeated sequence to match the specified length.

    Examples:
        seq: [1, 2, 3], length: 7 -> [1, 2, 3, 1, 2, 3, 1]
        seq: [1, 2, 3], length: 2 -> [1, 2] or [2, 3]
    """
    seq_len = len(seq)

    if length <= seq_len:
        # If length is less than or equal to seq length, return a random consecutive subsequence
        start_idx = random.randint(0, seq_len - length)
        return seq[start_idx : start_idx + length]
    else:
        # If length is greater than seq length, repeat integer times and randomly sample remaining
        result = []
        full_repeats = length // seq_len
        remaining = length % seq_len

        # Add full repetitions
        result.extend(seq * full_repeats)

        # Randomly sample remaining elements (with replacement if needed)
        if remaining > 0:
            remaining_elements = random.choices(seq, k=remaining)
            result.extend(remaining_elements)

        return result


class VideoDepthCameraDataset(Dataset):
    """
    VideoDataset class for loading and processing video data.

    This dataset handles:
    - Loading videos from specified folders
    - Frame sampling with random or fixed intervals
    - Resizing frames using different methods (crop, resize, random)
    - Applying custom transformation pipelines

    Args:
        video_folders: List of folders containing video files
        resize_mode: Frame resizing method ("crop", "resize", or "random")
        extensions: List of video file extensions to include
        random_sample_frame: Whether to sample frames randomly
        pipelines: List of transforms to apply to frames
    """

    def __init__(
        self,
        video_folders: Sequence[os.PathLike],
        data_root: Optional[Path] = None,
        extensions: Sequence[str] = (".mp4", ".avi"),
        frame_size: Tuple[int, int] = (448, 448),
        frame_nums: int = 2,
        resize_mode: Literal["crop", "resize", "random"] = "crop",
        sample_mode: Literal["uniform", "random"] = "random",
        max_sample_step: int = 5,
        pipelines: Sequence[tvf.Transform] = None,
    ):
        self.data_root = data_root
        self.video_paths = []
        for folder in video_folders:
            folder_with_nums = folder.split("#")
            folder = Path(folder_with_nums[0])

            if not folder.exists():
                folder = data_root / folder
            else:
                warnings.warn(
                    f"Folder {folder} is found, so data_root: {data_root} is ignored."
                )

            # Use glob to match all supported video formats
            video_files = []
            for ext in extensions:
                video_files.extend(sorted(folder.glob("*" + ext)))

            # Repeat the sequence if a number is specified
            if len(folder_with_nums) > 1:
                nums = int(folder_with_nums[1])
                video_files = repeat_sequence(video_files, nums)

            self.video_paths.extend(video_files)

        # # Ensure video paths are unique and maintain consistent order
        # self.video_paths = sorted(set(self.video_paths))

        if len(self.video_paths) < 1:
            warnings.warn(
                f"No video paths found from given folders: {video_folders}", UserWarning
            )

        self.frame_size = frame_size
        self.frame_nums = frame_nums

        self.sample_mode = sample_mode
        self.resize_mode = resize_mode
        self.max_sample_step = max_sample_step

        self.pipelines = tvf.Compose(pipelines) if pipelines else None

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, index: Union[int, Sequence, Dict]):
        """
        Retrieves and processes video frames based on the given index.

        Args:
            index: Can be one of:
                - int: Simple index of video in dataset
                - Sequence: [index, size, num_frames]
                - Dict: {
                    'index': int (required),
                    'size': tuple (optional, default=(224,224)),
                    'num_frames': int (optional, default=8)
                  }

        Returns:
            torch.Tensor: Processed video frames with shape (T, C, H, W) where:
                T = number of frames
                C = channels (3 for RGB)
                H = height
                W = width

        Raises:
            ValueError: If index is not int, Sequence or dict with 'index' key
        """

        if isinstance(index, int):
            index = {"index": index}
        elif isinstance(index, Sequence):
            assert len(index) >= 3, "Sequence must have at least 3 elements"
            index = {"index": index[0], "size": index[1], "num_frames": index[2]}
        elif isinstance(index, Mapping):
            assert "index" in index, "Dict must have 'index' key"
        else:
            raise ValueError("Index must be an int or a dict with 'index' key")

        idx = index["index"]
        size = index.get("size", self.frame_size)
        num_frames = index.get("num_frames", self.frame_nums)

        video_path = self.video_paths[idx]

        # Read video using decord
        vr = decord.VideoReader(str(video_path))

        # Get total number of frames and frame rate
        total_frames = len(vr)
        frame_indices, frame_repeats = self._sample_frames(total_frames, num_frames)

        # Read specified frames
        frames = vr.get_batch(frame_indices)
        frames = frames.asnumpy().transpose(0, 3, 1, 2)
        frames = torch.from_numpy(frames)

        with h5py.File(video_path.with_suffix(".h5"), "r") as h5f:
            unqiue_frames = (
                frame_indices[:-frame_repeats] if frame_repeats > 0 else frame_indices
            )
            depthmaps = h5f["depth"][unqiue_frames].astype(np.float32)
            cam_ins = h5f["intrinsic"][unqiue_frames].astype(np.float32)
            cam_poses = h5f["pos"][unqiue_frames].astype(np.float32)

            depthmaps = torch.from_numpy(depthmaps)
            cam_ins = torch.from_numpy(cam_ins)
            cam_poses = torch.from_numpy(cam_poses)

            if frame_repeats > 0:
                repeat_ids = [-1] * frame_repeats
                depthmaps = torch.cat([depthmaps, depthmaps[repeat_ids]], dim=0)
                cam_ins = torch.cat([cam_ins, cam_ins[repeat_ids]], dim=0)
                cam_poses = torch.cat([cam_poses, cam_poses[repeat_ids]], dim=0)

            invalid_value = h5f["depth"].attrs["invalid"]
            if not isinstance(invalid_value, (list, tuple, np.ndarray)):
                invalid_value = [invalid_value]

        # Adjust resolution
        frames, depthmaps, cam_ins = self._resize_frames(
            frames, depthmaps, cam_ins, size
        )

        # Get valid_mask by checking if depth values are in invalid range
        valid_mask = ~torch.isin(depthmaps, torch.tensor(invalid_value))
        valid_mask &= depthmaps > 1e-8

        # Convert depthmaps to pts3d
        pts3d = depthmaps_to_pts3d(depthmaps, cam_ins, cam_poses)
        valid_mask &= pts3d.isfinite().all(dim=-1)

        if self.pipelines:
            frames = self.pipelines(frames)

        return dict(
            sample_idx=idx,
            video_path=str(video_path),
            frame_indices=frame_indices,
            images=frames.permute(0, 2, 3, 1),
            depthmap=depthmaps,
            pts3d=pts3d,
            camera_intrinsics=cam_ins,
            camera_pose=cam_poses,
            valid_mask=valid_mask,
        )

    def _sample_frames(self, total_frames, num_frames):
        """
        Sample video frames, supporting random start and step

        Args:
            total_frames: Total number of frames in the video
            num_frames: Number of frames to sample
        """
        frame_repeats = 0
        if total_frames <= num_frames:
            frame_repeats = num_frames - total_frames
            # If there are not enough video frames, repeat the last frame
            frame_indices = (
                list(range(total_frames)) + [total_frames - 1] * frame_repeats
            )
            return frame_indices, frame_repeats

        start = 0
        if self.sample_mode == "random":
            # Random start frame
            start = random.randint(0, total_frames - num_frames)

            # Calculate maximum possible step
            max_step = (total_frames - start) // num_frames
            max_step = max(max_step, 1)  # at least 1
            max_step = min(max_step, self.max_sample_step)

            # Randomly select step
            step = random.randint(1, max_step)
        elif self.sample_mode == "uniform":
            # Fixed sampling step
            step = max((total_frames - start) // num_frames, 1)
        else:
            raise ValueError(f"Unknown sample mode: {self.sample_mode}")

        frame_indices = list(range(start, total_frames, step))[:num_frames]
        # If sampling is insufficient (due to integer division truncation), fill with last frame
        if len(frame_indices) < num_frames:
            frame_repeats = num_frames - len(frame_indices)
            frame_indices += [total_frames - 1] * frame_repeats

        return frame_indices, frame_repeats

    def _resize_frames(self, frames, depthmaps, intrinsics, size):
        if frames.shape[-2:] == tuple(size) and depthmaps.shape[-2:] == tuple(size):
            return frames, depthmaps, intrinsics

        # Resize using torchvision.transforms.v2
        if self.resize_mode == "crop":
            resize = CenterCropWithCamera(size=size)
        elif self.resize_mode == "resize":
            resize = ResizeWithCamera(size=size)
        elif self.resize_mode == "random":
            resize = RandomResizedCropWithCamera(
                size=size,
                scale=(0.5, 1.0),
                ratio=(0.75, 1.33),
            )
        else:
            raise ValueError(f"Unknown resize mode: {self.resize_mode}")

        return resize(frames, depthmaps, intrinsics)
