import click
import parse
import dtoolcore

import numpy as np

from skimage.draw import polygon2mask
from skimage.transform import resize

from dtoolbioimage import Image, scale_to_uint8

from roi2json import roi_from_fpath
from aiutils.data import image_mask_dataset_from_im_mask_iter, ImageMaskDataSet


def filtered_by_relpath(ds, relpath_filter):
    for idn in ds.identifiers:
        relpath = ds.item_properties(idn)['relpath']
        if relpath_filter(relpath):
            yield idn


def idn_to_image_n(ds, idn):
    relpath = ds.item_properties(idn)['relpath']
    r = parse.parse("image_{:d}_stigma_area.roi", relpath)
    n, = r.fixed
    return n


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


def image_n_to_roi_fpath(ds, n):
    idn = dtoolcore.utils.generate_identifier(f"image_{n}_stigma_area.roi")
    return ds.item_content_abspath(idn)


def fpaths_to_resized_image_and_mask(image_fpath, roi_fpath):
    im = Image.from_file(image_fpath)
    rdim, cdim, *_ = im.shape

    coords = roi_from_fpath(roi_fpath)["coords"]
    tcoords = [(r, c) for c, r in coords]

    mask = polygon2mask((rdim, cdim), tcoords)

    resized_im = resize(im, (512, 512))
    resized_mask = resize(mask, (512, 512))

    return resized_im, resized_mask


@click.command()
@click.argument('input_ds_uri')
@click.argument('output_base_uri')
@click.argument('output_name')
def main(input_ds_uri, output_base_uri, output_name):

    ds = dtoolcore.DataSet.from_uri(input_ds_uri)
    def relpath_filter(relpath): return relpath.endswith("stigma_area.roi")
    idns = list(filtered_by_relpath(ds, relpath_filter))
    image_numbers = [idn_to_image_n(ds, idn) for idn in idns]


    image_fpaths = [image_n_to_image_fpath(ds, n) for n in image_numbers]
    roi_fpaths = [image_n_to_roi_fpath(ds, n) for n in image_numbers]

    ims_masks = [
        fpaths_to_resized_image_and_mask(image_fpath, roi_fpath)
        for image_fpath, roi_fpath in zip(image_fpaths, roi_fpaths)
    ]

    image_mask_dataset_from_im_mask_iter(output_base_uri, output_name, ims_masks, ds)


if __name__ == "__main__":
    main()
