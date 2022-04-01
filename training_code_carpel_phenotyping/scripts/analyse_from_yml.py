import logging

import click
import dtoolcore

from aiutils.unetmodel import TrainedUNet
from runtools.config import Config
from runtools.data import IndexedDirtree, ImageDataSetView

from utils import fpath_and_model_to_result


logger = logging.getLogger(__file__)


@click.command()
@click.argument('config_fpath')
def main(config_fpath):
    logging.basicConfig(level=logging.INFO)

    config = Config.from_fpath(config_fpath)

    logger.info(f"Loading dirtree from {config.input_dirpath}")
    itree = IndexedDirtree(config.input_dirpath)
    ids = ImageDataSetView(itree)

    sq_pixels_per_mm_sq = config.pixels_per_mm ** 2
    unetmodel = TrainedUNet(config.model_uri, input_channels=3)

    for relpath in config.relpaths:
        idn = dtoolcore.utils.generate_identifier(relpath)
        fpath = itree.item_content_abspath(idn)
        result = fpath_and_model_to_result(fpath, unetmodel, sq_pixels_per_mm_sq)
        result["annotated_pil_image"].save("foop.png")



if __name__ == "__main__":
    main()
