import random
import logging

import click
import pandas
import dtoolcore

from aiutils.unetmodel import TrainedUNet

from runtools.config import Config
from runtools.run import one_to_one, create_readme
from runtools.data import IndexedDirtree, ImageDataSetView

from utils import fpath_and_model_to_result


@click.command()
@click.argument('config_fpath')
def main(config_fpath):

    logging.basicConfig(level=logging.INFO)

    config = Config.from_fpath(config_fpath)

    unetmodel = TrainedUNet(config.model_uri, input_channels=3)

    itree = IndexedDirtree(config.input_dirpath)
    ids = ImageDataSetView(itree)
    # idns = random.sample(ids.identifiers, 9)
    idns = ids.identifiers

    sq_pixels_per_mm_sq = config.pixels_per_mm ** 2

    def process_func(input_ds, idn, output_ds):
        fpath = input_ds.item_content_abspath(idn)
        result = fpath_and_model_to_result(fpath, unetmodel, sq_pixels_per_mm_sq)
        relpath = input_ds.item_properties(idn)["relpath"]
        result["relpath"] = relpath
        output_relpath = relpath
        output_abspath = output_ds.prepare_staging_abspath_promise(f"{output_relpath}-annotated.png")
        result["annotated_pil_image"].save(output_abspath)
        return result

    with dtoolcore.DataSetCreator(config.output_name, config.output_base_uri) as output_ds:
        results, runstats = one_to_one(itree, idns, process_func, output_ds)
        df = pandas.DataFrame(results)
        csv_output_relpath = "all_results.csv"
        csv_output_abspath = output_ds.prepare_staging_abspath_promise(csv_output_relpath)
        columns = ["relpath", "mask_area_pixels", "mask_area_mm"]
        df.sort_values(by="filename", inplace=True)
        df[columns].to_csv(csv_output_abspath, index=False, float_format='%.3f')

        readme_content = create_readme(config, runstats)

    output_ds.put_readme(readme_content)


if __name__ == "__main__":
    main()