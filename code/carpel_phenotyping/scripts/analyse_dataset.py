import os
import logging
import itertools
from collections import defaultdict

import pandas as pd

import click
import dtoolcore

from aiutils.unetmodel import TrainedUNet

from utils import fpath_and_model_to_result
# FIXME - config file

MODEL_URI = "/Users/mhartley/models/stigma-area-unet-full"
PIXELS_PER_MM = 143


def replace_ext(fname, new_ext):
    root, _ = os.path.splitext(fname)
    return root + new_ext


def summarise_by_extension(dataset):

    by_extension = defaultdict(int)

    for idn in dataset.identifiers:
        ext = dataset.item_properties(idn)["relpath"].rsplit(".", 1)[1]
        by_extension[ext] += 1

    for ext, count in by_extension.items():
        print(f"{ext}: {count}")


def iter_identifiers_with_extension(dataset, extension):

    for idn in dataset.identifiers:
        if dataset.item_properties(idn)["relpath"].endswith(extension):
            yield idn


@click.command()
@click.argument('dataset_uri')
@click.argument('output_base_uri')
@click.argument('output_name')
def main(dataset_uri, output_base_uri, output_name):

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("stigma_measurement")

    logger.info(f"Processing all files in {dataset_uri}")

    ds = dtoolcore.DataSet.from_uri(dataset_uri)
    jpg_idn_iter = iter_identifiers_with_extension(ds, ".jpg")

    sq_pixels_per_mm_sq = PIXELS_PER_MM * PIXELS_PER_MM

    unetmodel = TrainedUNet(MODEL_URI, input_channels=3)

    idns_to_process = list(jpg_idn_iter)

    measurements = []
    with dtoolcore.DerivedDataSetCreator(output_name, output_base_uri, ds) as output_ds:
        for n, idn in enumerate(idns_to_process, start=1):
            fpath = ds.item_content_abspath(idn)
            logger.info(f"Processing [{n}/{len(idns_to_process)}]: {fpath}")
            input_relpath = ds.item_properties(idn)["relpath"]
            output_relpath = replace_ext(input_relpath, "-annotated.png")
            output_abspath = output_ds.prepare_staging_abspath_promise(output_relpath)
            result = fpath_and_model_to_result(fpath, unetmodel, sq_pixels_per_mm_sq)

            result["annotated_pil_image"].save(output_abspath)
            result["filename"] = input_relpath
            measurements.append(result)

        df = pd.DataFrame(measurements)

        csv_output_relpath = "all_results.csv"
        csv_output_abspath = output_ds.prepare_staging_abspath_promise(csv_output_relpath)
        columns = ["filename", "mask_area_pixels", "mask_area_mm"]
        df.sort_values(by="filename", inplace=True)
        df[columns].to_csv(csv_output_abspath, index=False, float_format='%.3f')
        



if __name__ == "__main__":
    main()