import json
import shutil
import pathlib

import click
import dtoolcore
import pandas as pd

from mruntools.config import Config


class WorkingDir(object):

    def __init__(self, dirpath):
        self.dirpath = pathlib.Path(dirpath)


@click.command()
@click.argument('config_fpath')
def main(config_fpath):

    config = Config.from_fpath(config_fpath)

    wdir = WorkingDir(config.working_dirpath)

    with dtoolcore.DataSetCreator(config.output_name, config.output_base_uri) as output_ds:
        data = []
        for dirpath in wdir.dirpath.iterdir():
            metadata_fpath = dirpath/"metadata.json"
            if metadata_fpath.is_file():
                with open(metadata_fpath) as fh:
                    metadata = json.load(fh)
                stigma_area_pixels = metadata["stigma_area_pixels"]
                filename = metadata["filename"]
                data.append((filename, stigma_area_pixels))
                image_fpath = dirpath/f"{filename}.png"
                image_dst_abspath = output_ds.prepare_staging_abspath_promise(f"{filename}.png")
                shutil.copy(image_fpath, image_dst_abspath)
            else:
                pass

        df = pd.DataFrame(data, columns=["filename","stigma_area_pixels"])
        df_abspath = output_ds.prepare_staging_abspath_promise("results.csv")
        df.to_csv(df_abspath, index=False, float_format="%.2f")


if __name__ == "__main__":
    main()
