import click
import dtoolcore
import skimage
import numpy as np

from aiutils.data import image_mask_dataset_from_im_mask_iter, ImageMaskDataSet


from utils import (
    image_numbers_from_suffix,
    n_to_image_mask_pair,
    resize_im_mask_pair
)


@click.command()
@click.argument('input_ds_uri')
@click.argument('output_base_uri')
@click.argument('output_name')
def main(input_ds_uri, output_base_uri, output_name):
    ds = dtoolcore.DataSet.from_uri(input_ds_uri)

    image_numbers = image_numbers_from_suffix(ds, "branch1.roi")

    im_masks_unscaled = [n_to_image_mask_pair(ds, n) for n in image_numbers]
    im_masks_scaled = [resize_im_mask_pair(im, mask) for im, mask in im_masks_unscaled]

    image_mask_dataset_from_im_mask_iter(
        output_base_uri, output_name, im_masks_scaled, ds
    )


if __name__ == "__main__":
    main()
