import torchvision.transforms.v2 as tvf

__all__ = [
    "CenterCropWithCamera",
    "ResizeWithCamera",
    "RandomResizedCropWithCamera",
]


class BaseTransformWithCamera(tvf.Transform):
    def camera_transform(self, intrinsics, params: dict):
        raise NotImplementedError

    def forward(self, images, depths=None, intrinsics=None):
        params = self.make_params(images)

        outputs = [self.transform(images, params)]
        if depths is not None:
            _interpolation = None
            if hasattr(self, "interpolation"):
                _interpolation = self.interpolation
                self.interpolation = tvf.InterpolationMode.NEAREST
            outputs.append(self.transform(depths, params))

            # Restore interpolation mode if changed
            if _interpolation is not None:
                self.interpolation = _interpolation

        if intrinsics is not None:
            params["ori_height"] = images.shape[-2]
            params["ori_width"] = images.shape[-1]
            params["new_height"] = outputs[0].shape[-2]
            params["new_width"] = outputs[0].shape[-1]
            outputs.append(self.camera_transform(intrinsics, params))

        if len(outputs) == 1:
            return outputs[0]
        else:
            return tuple(outputs)


class CenterCropWithCamera(BaseTransformWithCamera, tvf.CenterCrop):
    def camera_transform(self, intrinsics, params):
        # Adjust camera intrinsics for center crop
        # When crop is applied, the principal point (cx, cy) needs to be shifted
        # and the focal lengths (fx, fy) remain the same

        if intrinsics is None:
            return None

        # Calculate crop dimensions
        crop_h, crop_w = self.size

        # Calculate top-left corner for center crop
        top = (params["ori_height"] - crop_h) // 2
        left = (params["ori_width"] - crop_w) // 2

        # Create a copy of intrinsics to avoid modifying the original
        new_intrinsics = intrinsics.clone()

        # Shift principal point by the crop offset
        # Assuming intrinsics shape: (batch, 3, 3) or (3, 3)
        if new_intrinsics.dim() == 3:
            # Batch of intrinsics
            new_intrinsics[:, 0, 2] -= left  # cx -= left
            new_intrinsics[:, 1, 2] -= top  # cy -= top
        else:
            # Single intrinsic matrix
            new_intrinsics[0, 2] -= left  # cx -= left
            new_intrinsics[1, 2] -= top  # cy -= top

        return new_intrinsics


class ResizeWithCamera(BaseTransformWithCamera, tvf.Resize):
    def camera_transform(self, intrinsics, params):
        # Adjust camera intrinsics for resize
        # When image is resized, the focal lengths (fx, fy) need to be scaled
        # based on the resize ratio

        # Get original and new dimensions
        ori_height, ori_width = params["ori_height"], params["ori_width"]
        new_height, new_width = params["new_height"], params["new_width"]

        # Calculate scaling ratios for width and height
        scale_w = new_width / ori_width
        scale_h = new_height / ori_height

        # Create a copy of intrinsics to avoid modifying the original
        new_intrinsics = intrinsics.clone()

        # Scale focal lengths (fx, fy) by the resize ratio
        # Assuming intrinsics shape: (batch, 3, 3) or (3, 3)
        if new_intrinsics.dim() == 3:
            # Batch of intrinsics
            new_intrinsics[:, 0, 0] *= scale_w  # fx *= width_scale
            new_intrinsics[:, 1, 1] *= scale_h  # fy *= height_scale
            new_intrinsics[:, 0, 2] *= scale_w  # cx *= width_scale
            new_intrinsics[:, 1, 2] *= scale_h  # cy *= height_scale
        else:
            # Single intrinsic matrix
            new_intrinsics[0, 0] *= scale_w  # fx *= width_scale
            new_intrinsics[1, 1] *= scale_h  # fy *= height_scale
            new_intrinsics[0, 2] *= scale_w  # cx *= width_scale
            new_intrinsics[1, 2] *= scale_h  # cy *= height_scale

        return new_intrinsics


class RandomResizedCropWithCamera(BaseTransformWithCamera, tvf.RandomResizedCrop):
    def camera_transform(self, intrinsics, params):
        # Adjust camera intrinsics for random resized crop
        # 1. crop by (top, left, height, width)
        # 2. reize to `size`
        # 3. adjust intrinsics based on the crop and resize

        # Extract crop parameters
        top = params["top"]
        left = params["left"]
        crop_height = params["height"]
        crop_width = params["width"]

        # Get final output size
        final_height, final_width = params["new_height"], params["new_width"]

        # Create a copy of intrinsics to avoid modifying the original
        new_intrinsics = intrinsics.clone()

        # Step 1: Apply crop adjustment (shift principal point)
        if new_intrinsics.dim() == 3:
            # Batch of intrinsics
            new_intrinsics[:, 0, 2] -= left  # cx -= left
            new_intrinsics[:, 1, 2] -= top  # cy -= top
        else:
            # Single intrinsic matrix
            new_intrinsics[0, 2] -= left  # cx -= left
            new_intrinsics[1, 2] -= top  # cy -= top

        # Step 2: Apply resize scaling (scale focal lengths and principal point)
        scale_w = final_width / crop_width
        scale_h = final_height / crop_height

        if new_intrinsics.dim() == 3:
            # Batch of intrinsics
            new_intrinsics[:, 0, 0] *= scale_w  # fx *= width_scale
            new_intrinsics[:, 1, 1] *= scale_h  # fy *= height_scale
            new_intrinsics[:, 0, 2] *= scale_w  # cx *= width_scale
            new_intrinsics[:, 1, 2] *= scale_h  # cy *= height_scale
        else:
            # Single intrinsic matrix
            new_intrinsics[0, 0] *= scale_w  # fx *= width_scale
            new_intrinsics[1, 1] *= scale_h  # fy *= height_scale
            new_intrinsics[0, 2] *= scale_w  # cx *= width_scale
            new_intrinsics[1, 2] *= scale_h  # cy *= height_scale

        return new_intrinsics
