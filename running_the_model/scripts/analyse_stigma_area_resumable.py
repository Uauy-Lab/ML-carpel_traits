import os
import json
import logging
import pathlib
import itertools

import click
import numpy as np
import dtoolcore

from dtoolbioimage.annotation import AnnotatedImage
from aiutils.unetmodel import TrainedUNet
from aiutils.postprocess import largest_mask_region_from_model, RegionMask
from mruntools.config import Config
from mruntools.data import IndexedDirtree, ImageDataSetView


logger = logging.getLogger(__file__)


class WorkingDir(object):

    def __init__(self, dirpath):
        self.dirpath = pathlib.Path(dirpath)


def image_to_annotated_image(im, trained_stigma_model):

    stigma_mask = largest_mask_region_from_model(trained_stigma_model, im)

    mask_size = int(np.sum(stigma_mask))

    ann = AnnotatedImage.from_image(im)
    ann.mark_mask(stigma_mask)
    ann.mark_mask(stigma_mask.borders, col=(255, 0, 0))
    display_str = f"{mask_size} sq pixels"
    ann.text_at((10, 10), display_str, color=(255, 255, 0))

    return ann, mask_size


@click.command()
@click.argument('config_fpath')
@click.option('--limit', default=0)
def main(config_fpath, limit):
    logging.basicConfig(level=logging.INFO)

    config = Config.from_fpath(config_fpath)

    logger.info(f"Loading dirtree from {config.input_dirpath}")
    itree = IndexedDirtree(config.input_dirpath)
    ids = ImageDataSetView(itree)

    if limit > 0:
        ids_to_process = itertools.islice(ids, limit)
    else:
        ids_to_process = ids

    # sq_pixels_per_mm_sq = config.pixels_per_mm ** 2
    trained_stigma_model = TrainedUNet(config.stigma_model_uri, input_channels=3)

    wdir = WorkingDir(config.working_dirpath)

    for n, im in enumerate(ids_to_process, start=1):
        basename, _ = os.path.splitext(im.properties['relpath'])
        output_dirpath = wdir.dirpath/basename
        output_dirpath.mkdir(exist_ok=True, parents=True)
        metadata_fpath = output_dirpath/"metadata.json"

        if metadata_fpath.exists():
            logger.info(f"Skipping {basename}")
        else:
            try:
                logger.info(f"Processing {n}/{len(ids)}, {basename}")
                ann, area = image_to_annotated_image(
                    im,
                    trained_stigma_model
                )
                im_fpath = output_dirpath/f"{basename}.png"
                ann.save(im_fpath)
                metadata = {
                    "filename": basename,
                    "stigma_area_pixels": area
                }
                with open(metadata_fpath, "w") as fh:
                    json.dump(metadata, fh, indent=2)
            except ValueError:
                logging.warning(f"Failed on {basename}")



if __name__ == "__main__":
    main()
