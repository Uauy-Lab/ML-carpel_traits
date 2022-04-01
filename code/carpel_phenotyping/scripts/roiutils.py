from roi2json import roi_from_fpath

import skimage.draw
import skimage.morphology


import numpy as np


def polyline_roi_to_mask(roi, dim):

    mask = np.zeros(dim, dtype=np.uint8)

    coords_list = roi['coords']
    for pfrom, pto in zip(coords_list, coords_list[1:]):
        rr, cc = skimage.draw.line(*pfrom, *pto)
        mask[cc, rr] = 255

    selem = skimage.morphology.disk(3)
    return skimage.morphology.dilation(mask, selem=selem)


def freehand_roi_to_filled_shape_mask(roi, dim):
    """Generate filled mask as a 2D numpy array.

    Args:
        roi: ROI as dictionary.
        dim: 2-tuple of intended mask size.

    Returns:
        2D numpy array of uint8, with 255 indicating mask regions.
    """

    assert len(dim) == 2, "Dimensions must be 2D"

    rdim, cdim = dim
    coords = roi["coords"]
    tcoords = [(r, c) for c, r in coords]

    mask = skimage.draw.polygon2mask((rdim, cdim), tcoords)
    
    return mask


def freehand_roi_to_convex_hull_mask(roi, dim):
    
    assert len(dim) == 2, "Dimensions must be 2D"

    mask = np.zeros(dim, dtype=np.uint8)

    coords_list = roi['coords']
    for pfrom, pto in zip(coords_list, coords_list[1:]):
        rr, cc = skimage.draw.line(*pfrom, *pto)
        mask[cc, rr] = 255

    return skimage.morphology.convex_hull_image(mask)


def line_roi_to_mask(roi, dim):

    mask = np.zeros(dim, dtype=np.uint8)

    pfrom = int(roi['x1']), int(roi['y1'])
    pto = int(roi['x2']), int(roi['y2'])
    rr, cc = skimage.draw.line(*pfrom, *pto)
    mask[cc, rr] = 255

    selem = skimage.morphology.disk(3)
    return skimage.morphology.dilation(mask, selem=selem)


roi_to_mask_fn = {
    'polyline': polyline_roi_to_mask,
    'freehand': freehand_roi_to_filled_shape_mask,
    'line': line_roi_to_mask
}


def roi_to_mask(roi, dim):
    roi_type = roi['type']
    fn = roi_to_mask_fn[roi_type]
    return fn(roi, dim)


def roi_list_to_composite_mask(roi_list, dim):
    masks = [roi_to_mask(roi, dim) for roi in roi_list]

    return np.any(masks, axis=0)
