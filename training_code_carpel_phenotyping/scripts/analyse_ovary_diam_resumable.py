import os
import json
import logging
import pathlib
import itertools

import click
import dtoolcore
import numpy as np
import pandas as pd
import skimage.draw

from dtoolbioimage import Image as dbiImage
from dtoolbioimage.annotation import AnnotatedImage
from aiutils.unetmodel import TrainedUNet
from aiutils.postprocess import largest_mask_region_from_model, RegionMask

from runtools.config import Config
from runtools.data import IndexedDirtree, ImageDataSetView


logger = logging.getLogger(__file__)


COL_RED = [255, 0, 0]
COL_GREEN = [0, 255, 0]
COL_CYAN = [0, 255, 255]


def perpendicular_point_clockwise(p0, p1):
    r0, c0 = p0
    r1, c1 = p1

    R, C = c1 - c0, -(r1 - r0)

    return r1+R, c1+C


def perpendicular_point_counter_clockwise(p0, p1):
    r0, c0 = p0
    r1, c1 = p1

    R, C = c1 - c0, -(r1 - r0)

    return r1-R, c1-C


def intersection_point_from_pp(pp, points_in_border, ovary_mask):
    rr, cc, aa = skimage.draw.line_aa(*pp, *ovary_mask.centroid)
    points_in_branch = set(zip(rr, cc))
    intersection = points_in_border & points_in_branch
    rm, cm = np.array(list(intersection)).mean(axis=0).astype(int)

    return rm, cm


def intersection_points(stigma_mask, ovary_mask):
    points_in_border = set(zip(*np.where(ovary_mask.borders)))

    pp = perpendicular_point_clockwise(
        stigma_mask.centroid,
        ovary_mask.centroid
    )
    p0 = intersection_point_from_pp(pp, points_in_border, ovary_mask)

    pp = perpendicular_point_counter_clockwise(
        stigma_mask.centroid,
        ovary_mask.centroid
    )
    p1 = intersection_point_from_pp(pp, points_in_border, ovary_mask)

    return p0, p1


def points_dist(p0, p1):
    np0 = np.array(p0)
    np1 = np.array(p1)

    return np.linalg.norm(np1-np0)


# FIXME - function name/return
def image_to_annotated_image(im, trained_ovary_model, trained_stigma_model):

    ovary_mask = largest_mask_region_from_model(trained_ovary_model, im)
    stigma_mask = largest_mask_region_from_model(trained_stigma_model, im)

    ann = AnnotatedImage.from_image(im)

    ann.mark_mask(stigma_mask)
    ann.mark_mask(ovary_mask.borders)

    ann.draw_line_aa(stigma_mask.centroid, ovary_mask.centroid, COL_RED)

    pc = perpendicular_point_clockwise(
        stigma_mask.centroid, ovary_mask.centroid)
    pcc = perpendicular_point_counter_clockwise(
        stigma_mask.centroid, ovary_mask.centroid)
    ann.draw_line_aa(ovary_mask.centroid, pc, COL_GREEN)
    ann.draw_line_aa(ovary_mask.centroid, pcc, COL_GREEN)

    p0, p1 = intersection_points(stigma_mask, ovary_mask)
    ann.draw_disk(p0, 5, COL_CYAN)
    ann.draw_disk(p1, 5, COL_CYAN)

    d = points_dist(p0, p1)

    return ann, d


class WorkingDir(object):

    def __init__(self, dirpath):
        self.dirpath = pathlib.Path(dirpath)


@click.command()
@click.argument('config_fpath')
def main(config_fpath):

    limit = 0

    logging.basicConfig(level=logging.INFO)

    config = Config.from_fpath(config_fpath)

    trained_ovary_model = TrainedUNet(config.ovary_model_uri, input_channels=3)
    trained_stigma_model = TrainedUNet(config.stigma_model_uri, input_channels=3)
    
    idt = IndexedDirtree(config.input_dirpath)
    ids = ImageDataSetView(idt)

    if limit != 0:
        ids_to_process = itertools.islice(ids, limit)
    else:
        ids_to_process = ids

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
                ann, d = image_to_annotated_image(
                    im,
                    trained_ovary_model,
                    trained_stigma_model
                )
                im_fpath = output_dirpath/f"{basename}.png"
                ann.save(im_fpath)
                metadata = {
                    "filename": basename,
                    "ovary_diam_pixels": d
                }
                with open(metadata_fpath, "w") as fh:
                    json.dump(metadata, fh, indent=2)
            except TypeError:
                logging.warning(f"Failed on {basename}")



    # measurements = []
    # with dtoolcore.DataSetCreator(config.output_name, config.output_base_uri) as output_ds:
    #     for n, im in enumerate(ids_to_process, start=1):
    #         basename, _ = os.path.splitext(im.properties['relpath'])
    #         logger.info(f"Processing {n}/{len(ids)}, {basename}")
    #         output_relpath = f"{basename}.png"
    #         output_abspath = output_ds.prepare_staging_abspath_promise(output_relpath)

    #         try:
    #             ann, d = image_to_annotated_image(
    #                 im,
    #                 trained_ovary_model,
    #                 trained_stigma_model
    #             )
    #             ann.save(output_abspath)
    #             measurements.append({
    #                 "filename": basename,
    #                 "ovary_diam_pixels": d,
    #             })
    #         except TypeError:
    #             logging.warning(f"Failed on {basename}")
        
    #     df = pd.DataFrame(measurements)
    #     csv_output_relpath = "ovary_measurements.csv"
    #     csv_output_abspath = output_ds.prepare_staging_abspath_promise(csv_output_relpath)
    #     df.to_csv(csv_output_abspath, index=False, float_format="%.2f")



if __name__ == "__main__":
    main()
