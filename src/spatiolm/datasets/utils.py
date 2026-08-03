import numpy as np
import torch


def draw_cross(img, point, color=(255, 0, 0), radius=1):
    """
    Draw a cross (+) on the given image at the specified point.

    Args:
        img: Input image (numpy array)
        point: Center point for the cross (x, y)
        color: Color of the cross (R, G, B) - ignored for pixel value 1
        thickness: Thickness of the cross lines
    """

    # Ensure point coordinates are integers
    x, y = int(point[0]), int(point[1])

    # Get image dimensions
    height, width = img.shape[:2]

    if isinstance(img, torch.Tensor):
        color = torch.tensor(color)
    else:
        color = np.array(color)

    if len(img.shape) != 3:
        color = color.mean()

    # Draw horizontal line with color
    # for i in range(-radius, radius + 1):
    for px in range(x - radius, x + radius + 1):
        if 0 <= px < width and 0 <= y < height:
            img[y, px] = color
    # Draw vertical line with color
    for py in range(y - radius, y + radius + 1):
        if 0 <= x < width and 0 <= py < height:
            img[py, x] = color

    return img
