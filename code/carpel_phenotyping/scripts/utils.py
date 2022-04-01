from functools import reduce

import PIL
from PIL import ImageDraw, ImageFont

import parse
import skimage.draw
import skimage.morphology
import skimage.transform
import skimage.segmentation
import dtoolcore
import numpy as np

from dtoolbioimage import Image as dbiImage, scale_to_uint8

from roi2json import roi_from_fpath


def filtered_by_relpath(ds, relpath_filter):
    for idn in ds.identifiers:
        relpath = ds.item_properties(idn)['relpath']
        if relpath_filter(relpath):
            yield idn


def idn_to_image_n(ds, idn, suffix):
    relpath = ds.item_properties(idn)['relpath']
    r = parse.parse("image_{:d}_" + suffix, relpath)
    n, = r.fixed
    return n


def image_numbers_from_suffix(ds, suffix):
    def relpath_filter(relpath): return relpath.endswith(suffix)
    idns = list(filtered_by_relpath(ds, relpath_filter))
    image_numbers = [idn_to_image_n(ds, idn, suffix) for idn in idns]

    return image_numbers


def image_n_to_image_fpath(ds, n):
    jpg_idn = dtoolcore.utils.generate_identifier(f"image_{n}.jpg")
    tif_idn = dtoolcore.utils.generate_identifier(f"image_{n}.tif")

    if jpg_idn in ds.identifiers:
        image_idn = jpg_idn
    elif tif_idn in ds.identifiers:
        image_idn = tif_idn
    else:
        raise KeyError(f"Can't find image for {n}")

    return ds.item_content_abspath(image_idn)


def image_n_to_branch_roi_fpath_list(ds, n):
    idn1 = dtoolcore.utils.generate_identifier(f"image_{n}_branch1.roi")
    idn2 = dtoolcore.utils.generate_identifier(f"image_{n}_branch2.roi")

    idns = [idn1, idn2]

    fpaths = [
        ds.item_content_abspath(idn)
        for idn in idns
        if idn in ds.identifiers
    ]

    return fpaths


def polyline_roi_to_mask(roi, dim):

    mask = np.zeros(dim, dtype=np.uint8)

    coords_list = roi['coords']
    for pfrom, pto in zip(coords_list, coords_list[1:]):
        rr, cc = skimage.draw.line(*pfrom, *pto)
        mask[cc, rr] = 255

    selem = skimage.morphology.disk(3)
    return skimage.morphology.dilation(mask, selem=selem)


def n_to_branch_mask(ds, n, dim):
    fpaths = image_n_to_branch_roi_fpath_list(ds, n)
    rois = [roi_from_fpath(fpath) for fpath in fpaths]
    masks = [polyline_roi_to_mask(roi, dim) for roi in rois]
    return reduce(lambda x, y: x ^ y, masks)


def n_to_image_mask_pair(ds, n):
    im_fpath = image_n_to_image_fpath(ds, n)
    im = dbiImage.from_file(im_fpath)
    rdim, cdim, _ = im.shape
    mask = n_to_branch_mask(ds, n, (rdim, cdim)).view(dbiImage)

    return im, mask


def resize_im_mask_pair(im, mask, dim=(512, 512)):

    resized_im = skimage.transform.resize(im, dim, anti_aliasing=False)
    resized_mask = skimage.transform.resize(mask, dim, anti_aliasing=False)

    return resized_im, resized_mask


def annotate_with_measurement_string(im, display_str):
    pilim = PIL.Image.fromarray(scale_to_uint8(im))
    draw = ImageDraw.ImageDraw(pilim)
    font = PIL.ImageFont.truetype("Microsoft Sans Serif.ttf", size=36)
    draw.text((10, 10),  display_str, font=font, fill=(255, 255, 0))

    return pilim


def get_mask_from_image(im, unetmodel, input_dim=(512, 512)):
    scaled_input = skimage.transform.resize(im, input_dim).astype(np.float32)
    scaled_mask = unetmodel.predict_mask_from_image(scaled_input)
    rdim, cdim = im.shape[0], im.shape[1]
    mask = scale_to_uint8(skimage.transform.resize(scaled_mask, (rdim, cdim)) > 0.5)

    return mask


def create_overlaid_image(im, mask):
    mask_rgb = np.dstack(3 * [scale_to_uint8(mask)])
    merge = 0.7 * scale_to_uint8(im) + 0.3 * mask_rgb
    return merge.view(dbiImage)


def fpath_and_model_to_result(fpath, unetmodel, sq_pixels_per_mm_sq):
    im = dbiImage.from_file(fpath)
    mask = get_mask_from_image(im, unetmodel)
    overlaid_image = create_overlaid_image(im, mask)

    boundary_image = skimage.segmentation.mark_boundaries(
        overlaid_image, mask, color=(255, 0, 0)
    )

    mask_size = int(np.sum(mask) / 255)

    mask_area_mm = mask_size / sq_pixels_per_mm_sq
    display_str = f"{mask_area_mm:.2f} mm sq ({mask_size} sq pixels)"

    pilim = annotate_with_measurement_string(boundary_image, display_str)

    result = {
        "filename": fpath,
        "annotated_pil_image": pilim,
        "mask_area_pixels": mask_size,
        "mask_area_mm": mask_area_mm
    }

    return result


def fpaths_to_resized_image_and_mask(image_fpath, roi_fpath):
    im = dbiImage.from_file(image_fpath)
    rdim, cdim, *_ = im.shape

    coords = roi_from_fpath(roi_fpath)["coords"]
    tcoords = [(r, c) for c, r in coords]

    mask = skimage.draw.polygon2mask((rdim, cdim), tcoords)

    resized_im = skimage.transform.resize(im, (512, 512))
    resized_mask = skimage.transform.resize(mask, (512, 512))

    return resized_im, resized_mask