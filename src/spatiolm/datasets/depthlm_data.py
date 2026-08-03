import os
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms.v2 as tvf
from PIL import Image

from .utils import draw_cross
from .video_dataset import VideoDepthCameraDataset


class BaseMessage:
    IMAGE_TAG = "<image>"
    CONV_TEMPLATES = {"probability": [], "question": [], "answer": []}

    def __init__(self, proba, mark_radius=5, probabilities=None):
        self.proba = proba
        self.mark_radius = mark_radius

        if probabilities is not None:
            for k, v in self.CONV_TEMPLATES.items():
                if k == "probability":
                    self.CONV_TEMPLATES["probability"] = probabilities
                else:
                    self.CONV_TEMPLATES[k] = v[: len(probabilities)]

    def _select_qa_template(self, **kwargs):
        template_idx = random.choices(
            range(len(self.CONV_TEMPLATES["probability"])),
            weights=self.CONV_TEMPLATES["probability"],
            k=1,
        )[0]

        question = self.CONV_TEMPLATES["question"][template_idx]
        answer = self.CONV_TEMPLATES["answer"][template_idx]
        return question.format(**kwargs), answer.format(**kwargs)

    def _make_messages(self, question, answer, img_nums=1):
        img_tags = self.IMAGE_TAG * img_nums

        return [
            {"role": "user", "content": f"{img_tags}{question}"},
            {"role": "assistant", "content": answer},
        ]


class DistanceMessage(BaseMessage):
    CONV_TEMPLATES = {
        "probability": [0.5, 0.2, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
        "question": [
            "Measure the distance for marked point in meters and provide the direct answer.",
            "How many meters is this point from the camera?",
            "What is the distance between the marked point and the camera?",
            "Estimate the depth of the marked location in meters.",
            "Can you tell me how far away the marked point is?",
            "Measure the distance from camera to the indicated point.",
            "What's the depth value at the marked position?",
            "Calculate the distance between the camera and the marked location.",
        ],
        "answer": [
            "{distance:.2f}",
            "The point is around {distance:.2f} meters away from the camera.",
            "The distance from the camera is approximately {distance:.2f} meters.",
            "The depth at the marked location is estimated to be {distance:.2f} meters.",
            "The marked point is about {distance:.2f} meters from the camera.",
            "The measured distance is {distance:.2f} meters.",
            "The depth value is {distance:.2f} meters.",
            "The calculated distance is {distance:.2f} meters.",
        ],
    }

    def __call__(self, data_info: Dict):
        images = data_info["images"]
        valid_mask = data_info["valid_mask"]
        depthmap = data_info["depthmap"]

        frame_idx = random.randint(0, len(images) - 1)
        vaild_coords = torch.nonzero(valid_mask[frame_idx], as_tuple=False)
        coord_idx = random.randint(0, len(vaild_coords) - 1)

        sample_point = vaild_coords[coord_idx].tolist()  # [row, col]
        distance = depthmap[frame_idx][sample_point[0], sample_point[1]].item()

        # Draw a cross on the image at the sampled point
        sample_point = tuple(sample_point[::-1])  # (x, y)
        draw_cross(
            images[frame_idx],
            sample_point,
            color=(255, 0, 0),
            radius=self.mark_radius,
        )

        # Randomly select question and answer based on probability
        question, answer = self._select_qa_template(distance=distance)
        messages = self._make_messages(question, answer, len(images))

        return {
            "messages": messages,
            "images": images,
            "distance": distance,
            "frame_idx": frame_idx,
            "sample_point": sample_point,
        }


class GlobalDistanceMessage(BaseMessage):
    CONV_TEMPLATES = {
        "probability": [0.5, 0.3, 0.1, 0.1],
        "question": [
            "Estimate the 3D spatial distance between the two marked points on the given images and provide the direct answer.",
            "What is the metric 3D distance between the two marked points in the given images?",
            "How far apart are the two annotated points in 3D space, measured in real-world units?",
            "What is the absolute spatial distance separating the two highlighted points in the scene?",
        ],
        "answer": [
            "{distance:.2f}",
            "The metric 3D distance between the two marked points is {distance:.2f} meters.",
            "The two annotated points are {distance:.2f} meters apart in 3D space.",
            "The absolute spatial distance between the two highlighted points is {distance:.2f} meters.",
        ],
    }

    def __call__(self, data_info: Dict):
        images = data_info["images"]
        valid_mask = data_info["valid_mask"]
        depthmap = data_info["depthmap"]

        camera_intrinsics = data_info["camera_intrinsics"]
        camera_pose = data_info["camera_pose"]

        assert len(images) > 1, "GlobalDistanceMessage requires at least 2 frames."

        global_points = []
        frame_ids = random.sample(range(len(images)), 2)
        for frame_idx in frame_ids:
            vaild_coords = torch.nonzero(valid_mask[frame_idx], as_tuple=False)
            coord_idx = random.randint(0, len(vaild_coords) - 1)
            sample_point = vaild_coords[coord_idx].tolist()  # [row, col]

            # Get current frame depth
            current_depth = depthmap[frame_idx][sample_point[0], sample_point[1]].item()

            # Convert image coordinates to camera coordinates (3D) in current frame
            intrinsic = camera_intrinsics[frame_idx]
            fx, fy, cx, cy = intrinsic[[0, 1, 0, 1], [0, 2, 1, 2]].tolist()
            u, v = sample_point[::-1]  # (x, y) in image

            # 3D point in current frame camera coordinate system
            X_loc = (u - cx) * current_depth / fx
            Y_loc = (v - cy) * current_depth / fy
            Z_loc = current_depth

            # Convert to homogeneous coordinates
            local_point = torch.tensor([X_loc, Y_loc, Z_loc, 1.0])
            # Get relative pose from frame_idx to frame 0
            pose_current_to_world = camera_pose[frame_idx]  # 4x4 transformation matrix

            # Transform point from current frame to frame 0 coordinate system
            global_point = pose_current_to_world @ local_point
            global_points.append(global_point[:3])

            # Draw a cross on the image at the sampled point
            sample_point = tuple(sample_point[::-1])  # (x, y)
            draw_cross(
                images[frame_idx],
                sample_point,
                color=(255, 0, 0),
                radius=self.mark_radius,
            )

        distance = (global_points[0] - global_points[1]).norm().item()
        # Randomly select question and answer based on probability
        question, answer = self._select_qa_template(distance=distance)
        messages = self._make_messages(question, answer, len(images))

        return {
            "messages": messages,
            "images": images,
            "distance": distance,
            "sample_point": global_points,
            "frame_idx": frame_ids,
        }


class SpeedMessage(BaseMessage):
    CONV_TEMPLATES = {
        "probability": [0.6, 0.2, 0.1, 0.1],
        "question": [
            "Calculate the required speed(m/s) to reach this point in {second:.2f} seconds and provide the direct answer.",
            "How many meters per second should we move in order to reach this point in exactly {second:.2f} seconds?",
            "If we want to arrive at the marked point within {second:.2f} seconds, what speed is required?",
            "What should our velocity be to reach this destination point in {second:.2f} seconds?",
        ],
        "answer": [
            "{speed:.2f}",
            "The point is around {distance:.2f} meters away. Hence, the speed should be around {distance:.2f} / {second:.2f} = {speed:.2f}m/s.",
            "Since the point is {distance:.2f} meters distant, we need to travel at {speed:.2f}m/s meters per second to reach it in {second:.2f} seconds.",
            "To cover the {distance:.2f} meter distance in {second:.2f} seconds, we must maintain a speed of {speed:.2f}m/s.",
        ],
    }

    def __call__(self, data_info: Dict):
        images = data_info["images"]
        valid_mask = data_info["valid_mask"]
        depthmap = data_info["depthmap"]

        frame_idx = random.randint(0, len(images) - 1)
        vaild_coords = torch.nonzero(valid_mask[frame_idx], as_tuple=False)
        coord_idx = random.randint(0, len(vaild_coords) - 1)

        sample_point = vaild_coords[coord_idx].tolist()  # [row, col]
        distance = depthmap[frame_idx][sample_point[0], sample_point[1]].item()

        # Draw a cross on the image at the sampled point
        sample_point = tuple(sample_point[::-1])  # (x, y)
        draw_cross(
            images[frame_idx],
            sample_point,
            color=(255, 0, 0),
            radius=self.mark_radius,
        )

        distance = round(distance, 2)
        speed = round(random.uniform(1.0, 30.0), 2)  # m/s
        second = distance / speed

        if second < 0.1:
            second = 0.1
            speed = distance * 10

        # Randomly select question and answer based on probability
        question, answer = self._select_qa_template(
            distance=distance, speed=speed, second=second
        )
        messages = self._make_messages(question, answer, len(images))

        return {
            "messages": messages,
            "images": images,
            "distance": distance,
            "frame_idx": frame_idx,
            "sample_point": sample_point,
        }


class TimeMessage(SpeedMessage):
    CONV_TEMPLATES = {
        "probability": [0.6, 0.2, 0.1, 0.1],
        "question": [
            "Calculate the travel time(s) to reach this point at {speed:.2f}m/s speed and provide the direct answer.",
            "How many seconds do we need to reach this point if we move towards it with the speed of {speed:.2f}m/s?",
            "If we travel at {speed:.2f}m/s, how long will it take to arrive at the marked point?",
            "What is the time required to reach this destination moving at {speed:.2f}m/s?",
        ],
        "answer": [
            "{second:.2f}",
            "The point is around {distance:.2f} meters away. Hence, we need around {distance:.2f} / {speed:.2f} = {second:.2f}s",
            "Since the point is {distance:.2f} meters distant and we're moving at {speed:.2f}m/s, the travel time will be {second:.2f} seconds.",
            "To cover the {distance:.2f} meter distance at {speed:.2f}m/s, it will take approximately {second:.2f} seconds.",
        ],
    }


class TwoPointsDistanceMessage(BaseMessage):
    CONV_TEMPLATES = {
        "probability": [0.6, 0.2, 0.1, 0.1],
        "question": [
            "Calculate the distance(m) between these two points and provide the direct answer.",
            "How far are these 2 points from each other?",
            "What is the distance between the two marked points?",
            "How many meters separate the two indicated points?",
        ],
        "answer": [
            "{distance:.2f}",
            "The distance between the two points is {distance:.2f} meters.",
            "The two marked points are {distance:.2f} meters apart.",
            "The separation distance between the points is {distance:.2f} meters.",
        ],
    }

    def __call__(self, data_info: Dict):
        images = data_info["images"]
        valid_mask = data_info["valid_mask"]
        depthmap = data_info["depthmap"]
        camera_intrinsics = data_info["camera_intrinsics"]

        frame_idx = random.randint(0, len(images) - 1)
        vaild_coords = torch.nonzero(valid_mask[frame_idx], as_tuple=False)

        # Sample two different points
        coord_idx1 = random.randint(0, len(vaild_coords) - 1)
        coord_idx2 = random.randint(0, len(vaild_coords) - 1)
        # Ensure they are different points
        while coord_idx2 == coord_idx1:
            coord_idx2 = random.randint(0, len(vaild_coords) - 1)

        sample_point1 = vaild_coords[coord_idx1].tolist()  # [row, col]
        sample_point2 = vaild_coords[coord_idx2].tolist()  # [row, col]

        # Get depth values for both points
        depth1 = depthmap[frame_idx][sample_point1[0], sample_point1[1]].item()
        depth2 = depthmap[frame_idx][sample_point2[0], sample_point2[1]].item()

        # Get camera intrinsics for the current frame
        intrinsic = camera_intrinsics[frame_idx]
        fx, fy, cx, cy = intrinsic[[0, 1, 0, 1], [0, 2, 1, 2]].tolist()

        # Convert image coordinates to camera coordinates (3D)
        # image coordinates: (row, col) = (y, x)
        # camera coordinates: (X, Y, Z) where Z is depth
        u1, v1 = sample_point1[::-1]  # (x, y) in image
        u2, v2 = sample_point2[::-1]  # (x, y) in image

        # 3D points in camera coordinate system
        X1 = (u1 - cx) * depth1 / fx
        Y1 = (v1 - cy) * depth1 / fy
        Z1 = depth1

        X2 = (u2 - cx) * depth2 / fx
        Y2 = (v2 - cy) * depth2 / fy
        Z2 = depth2

        # Calculate Euclidean distance in 3D space
        distance = ((X2 - X1) ** 2 + (Y2 - Y1) ** 2 + (Z2 - Z1) ** 2) ** 0.5

        # Draw crosses on both sampled points
        for sample_point in [sample_point1, sample_point2]:
            draw_cross(
                images[frame_idx],
                sample_point[::-1],
                color=(255, 0, 0),
                radius=self.mark_radius,
            )

        # Randomly select question and answer based on probability
        question, answer = self._select_qa_template(distance=distance)
        messages = self._make_messages(question, answer, len(images))

        return {
            "messages": messages,
            "images": images,
            "distance": distance,
            "frame_idx": frame_idx,
            "sample_point": (sample_point1, sample_point2),
        }


class RelativeDistanceMessage(BaseMessage):
    CONV_TEMPLATES = {
        "probability": [0.6, 0.2, 0.2],
        "question": [
            "In the image, Which marked points {marker1} (red) and {marker2} (green) is {metric} to me? Answer only: '{marker1}' or '{marker2}'.",
            "Which object is {metric} to me in the image, {marker1} (red) or {marker2} (green)? Answer only: '{marker1}' or '{marker2}'.",
            "Looking at the image, decide which point is {metric}: {marker1} (red) or {marker2} (green). Answer only: '{marker1}' or '{marker2}'.",
        ],
        "answer": ["{answer}", "{answer}", "{answer}"],
    }

    def __call__(self, data_info: Dict):
        images = data_info["images"]
        valid_mask = data_info["valid_mask"]
        depthmap = data_info["depthmap"]

        frame_idx = random.randint(0, len(images) - 1)
        vaild_coords = torch.nonzero(valid_mask[frame_idx], as_tuple=False)

        # Sample two different points
        coord_idx1 = random.randint(0, len(vaild_coords) - 1)
        coord_idx2 = random.randint(0, len(vaild_coords) - 1)
        # Ensure they are different points
        while coord_idx2 == coord_idx1:
            coord_idx2 = random.randint(0, len(vaild_coords) - 1)

        sample_point1 = vaild_coords[coord_idx1].tolist()  # [row, col]
        sample_point2 = vaild_coords[coord_idx2].tolist()  # [row, col]

        # Get depth values for both points
        depth1 = depthmap[frame_idx][sample_point1[0], sample_point1[1]].item()
        depth2 = depthmap[frame_idx][sample_point2[0], sample_point2[1]].item()

        # Draw crosses on both sampled points
        colors = [(255, 0, 0), (0, 255, 0)]  # red and green
        for sample_point, color in zip([sample_point1, sample_point2], colors):
            draw_cross(
                images[frame_idx],
                sample_point[::-1],
                color=color,
                radius=self.mark_radius,
            )

        marker1, marker2 = random.choices(
            [
                ("A", "B"),
                ("B", "A"),
                ("X", "Y"),
                ("1", "2"),
                ("point1", "point2"),
                ("point2", "point1"),
            ],
            weights=[0.5, 0.1, 0.1, 0.1, 0.1, 0.1],
        )[0]
        metric = random.choice(["closer", "farther"])
        if metric == "closer":
            anwser = marker1 if depth1 < depth2 else marker2
        else:
            anwser = marker1 if depth1 > depth2 else marker2

        question, answer = self._select_qa_template(
            metric=metric, answer=anwser, marker1=marker1, marker2=marker2
        )
        messages = self._make_messages(question, answer, len(images))

        return {
            "messages": messages,
            "images": images,
            "distance": (depth1, depth2),
            "frame_idx": frame_idx,
            "sample_point": (sample_point1, sample_point2),
        }


class CameraMovementMessage(TwoPointsDistanceMessage):
    CONV_TEMPLATES = {
        "probability": [0.6, 0.2, 0.1, 0.1],
        "question": [
            "Calculate the movement(m) between given two images and provide the direct answer.",
            "How many meters has the camera moved between these 2 images?",
            "What is the distance traveled by the camera between the two frames?",
            "Estimate the camera's displacement between these two consecutive images.",
        ],
        "answer": [
            "{distance:.2f}",
            "The camera has moved approximately {distance:.2f} meters between the two images.",
            "The camera displacement between frames is {distance:.2f} meters.",
            "The estimated movement distance is {distance:.2f} meters.",
        ],
    }

    def __call__(self, data_info: Dict):
        images = data_info["images"]
        camera_pose = data_info["camera_pose"]

        cam_xyz1 = camera_pose[0, :3, ..., -1]
        cam_xyz2 = camera_pose[1, :3, ..., -1]

        distance = np.linalg.norm(cam_xyz2 - cam_xyz1)

        # Randomly select question and answer based on probability
        question, answer = self._select_qa_template(distance=distance)
        messages = self._make_messages(question, answer, len(images))

        return {
            "messages": messages,
            "images": images,
            "distance": distance,
        }


class DepthLmDataset(VideoDepthCameraDataset):
    DISTANCE_PROMPT = "How many meters is this point from the camera?"
    DISTANCE_ANSWER = "The point is around {:.2f} meters away from the camera."

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
        mark_radius: int = 2,
        pipelines: Sequence[tvf.Transform] = None,
        prompt_templates: Sequence[BaseMessage] = [
            DistanceMessage(proba=0.50),
            RelativeDistanceMessage(proba=0.15),
            GlobalDistanceMessage(proba=0.10),
            CameraMovementMessage(proba=0.10),
            SpeedMessage(proba=0.05),
            TimeMessage(proba=0.05),
            TwoPointsDistanceMessage(proba=0.05),
        ],
    ):
        super().__init__(
            video_folders,
            data_root,
            extensions,
            frame_size,
            frame_nums,
            resize_mode,
            sample_mode,
            max_sample_step,
            pipelines,
        )
        self.prompt_templates = prompt_templates
        self._prompt_weights = [t.proba for t in self.prompt_templates]

        if mark_radius is not None:
            for t in self.prompt_templates:
                t.mark_radius = mark_radius

    def _random_prompt(self, data_info: Dict):
        prompt_template = random.choices(self.prompt_templates, self._prompt_weights)[0]
        return prompt_template(data_info)

    def __getitem__(self, index):
        sample_cnt = 0
        while sample_cnt < 20:
            data = super().__getitem__(index)
            if (data["valid_mask"].sum(dim=(1, 2)) > 10).all():
                break

            # Resample if no valid frames
            index = random.randint(0, len(self) - 1)
            sample_cnt += 1
        else:
            raise ValueError("Failed to find a valid frame after 20 attempts.")

        data["images"] = data["images"].numpy()
        conversation = self._random_prompt(data)

        conversation.update(
            dict(
                images=[Image.fromarray(img) for img in conversation["images"]],
                video_path=data["video_path"],
            )
        )

        return conversation
