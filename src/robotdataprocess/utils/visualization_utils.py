from __future__ import annotations

import colorsys
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Union


def build_color_palette(colorList: List[str]) -> List[List[Tuple[float, float, float]]]:
    """
    Converts a list of hex colors to per-color palettes with varying
    lightness.

    Args:
        colorList: Base colors as hex strings starting with #.

    Returns:
        One 20-entry list of (r, g, b) floats (0-1) per input color,
        evenly spaced from dark to light at the input color's hue/saturation.
    """
    paletteList = []
    for c in colorList:
        rgb = mcolors.to_rgb(c)
        h, _, s = colorsys.rgb_to_hls(*rgb)
        lightnesses = np.linspace(0.0, 1.0, 20)
        paletteList.append([colorsys.hls_to_rgb(h, li, s) for li in lightnesses])
    return paletteList

def draw_background_image(ax: plt.Axes, background_image_path: str, background_image_x_edge: Union[float, None],
                           background_image_extent_offsets: Union[Tuple[float, float], None]) -> List[float]:
    """
    Loads and draws a background image onto ax. The image center is
    placed at world (x=0, y=0), unless background_image_extent_offsets
    shifts it elsewhere.

    Args:
        ax: Axes to draw onto.
        background_image_path: Path to the image file.
        background_image_x_edge: Distance in meters from image center to the x edge.
        background_image_extent_offsets: XY location where the image center should be placed,
            or None to leave it at world (x=0, y=0).

    Returns:
        The [x_min, x_max, y_min, y_max] extent (meters) the image was drawn at.

    Raises:
        ValueError: If background_image_x_edge is not provided.
    """
    if not background_image_x_edge:
        raise ValueError("Extent must be provided with Background image size via background_image_x_edge.")

    img = mpimg.imread(background_image_path)
    x_extent_meters = background_image_x_edge
    h, w = img.shape[0], img.shape[1]
    y_extent_meters = x_extent_meters / w * h
    x_offset, y_offset = background_image_extent_offsets if background_image_extent_offsets is not None else (0, 0)
    extent = [-x_extent_meters + x_offset, x_extent_meters + x_offset,
                -y_extent_meters + y_offset, y_extent_meters + y_offset]
    ax.imshow(img, extent=extent, origin="upper", alpha=1.0, zorder=0)
    return extent

def compute_bounds_and_aspect(xy_list: List[np.ndarray], target_ar: float = 1.5) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Computes padded XY bounds across all given positions, then expands
    whichever axis is short so the bounds match target_ar.

    Args:
        xy_list: One (N, 2) array of XY positions per trajectory.
        target_ar: Desired width / height ratio of the bounds.

    Returns:
        ((x_min, x_max), (y_min, y_max))
    """
    all_x = np.concatenate([xy[:, 0] for xy in xy_list])
    all_y = np.concatenate([xy[:, 1] for xy in xy_list])
    padding_x = (all_x.max() - all_x.min()) * 0.05
    padding_y = (all_y.max() - all_y.min()) * 0.05
    x_min, x_max = all_x.min() - padding_x, all_x.max() + padding_x
    y_min, y_max = all_y.min() - padding_y, all_y.max() + padding_y

    current_width = x_max - x_min
    current_height = y_max - y_min
    current_ar = current_width / current_height

    if current_ar > target_ar:
        target_height = current_width / target_ar
        diff = target_height - current_height
        y_min -= diff / 2
        y_max += diff / 2
    elif current_ar < target_ar:
        target_width = current_height * target_ar
        diff = target_width - current_width
        x_min -= diff / 2
        x_max += diff / 2

    return (x_min, x_max), (y_min, y_max)

