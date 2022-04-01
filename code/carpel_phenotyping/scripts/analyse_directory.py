import os
import logging
import pathlib

import PIL
from PIL import ImageDraw, ImageFont
import click

import numpy as np
from skimage.transform import resize
from skimage.segmentation import mark_boundaries

from dtoolbioimage import Image as dbiImage, scale_to_uint8
from aiutils.unetmodel import TrainedUNet


# TODO - config file
MODEL_URI = "/Users/mhartley/models/stigma-area-unet-full"
PIXELS_PER_MM = 143


def create_overlaid_image(im, mask):
    mask_rgb = np.dstack(3 * [scale_to_uint8(mask)])
    merge = 0.7 * scale_to_uint8(im) + 0.3 * mask_rgb
    return merge.view(dbiImage)


def replace_ext(fname, new_ext):
    root, _ = os.path.splitext(fname)
    return root + new_ext


def annotate_with_measurement_string(im, display_str):
    pilim = PIL.Image.fromarray(scale_to_uint8(im))
    draw = ImageDraw.ImageDraw(pilim)
    font = PIL.ImageFont.truetype("Microsoft Sans Serif.ttf", size=36)
    draw.text((10, 10),  display_str, font=font, fill=(255, 255, 0))

    return pilim


def get_mask_from_image(im, unetmodel, input_dim=(512, 512)):
    scaled_input = resize(im, input_dim).astype(np.float32)
    scaled_mask = unetmodel.predict_mask_from_image(scaled_input)
    rdim, cdim = im.shape[0], im.shape[1]
    mask = scale_to_uint8(resize(scaled_mask, (rdim, cdim)) > 0.5)

    return mask


@click.command()
@click.argument('input_dirpath')
@click.argument('output_dirpath')
def main(input_dirpath, output_dirpath):

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("stigma_analysis")

    input_fpath_iter = pathlib.Path(input_dirpath).iterdir()

    sq_pixels_per_mm_sq = PIXELS_PER_MM * PIXELS_PER_MM
    unetmodel = TrainedUNet(MODEL_URI, input_channels=3)

    for fpath in input_fpath_iter:

        logger.info(f"Processing {fpath}")

        im = dbiImage.from_file(fpath)
        mask = get_mask_from_image(im, unetmodel)
        overlaid_image = create_overlaid_image(im, mask)

        output_fname = replace_ext(fpath.name, "-annotated.png")
        output_fpath = pathlib.Path(output_dirpath) / output_fname

        boundary_image = mark_boundaries(overlaid_image, mask, color=(255, 0, 0))


        mask_size = int(np.sum(mask) / 255)
        print(mask.shape    )
        print(mask_size)
        print(len(np.nonzero(mask)[0]))
        mask_area_mm = mask_size / sq_pixels_per_mm_sq
        display_str = f"{mask_area_mm:.2f} mm sq ({mask_size} sq pixels)"

        pilim = annotate_with_measurement_string(boundary_image, display_str)
        pilim.save(output_fpath)


if __name__ == "__main__":
    main()
