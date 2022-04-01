import click
import dtoolcore

import numpy as np

from dtoolbioimage import Image

from skimage.transform import resize

from application import fpath_to_composite_image

from aiutils.unetmodel import TrainedUNet

from itertools import zip_longest


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)


@click.command()
@click.argument('model_uri')
@click.argument('test_ds_uri')
@click.argument('output_fpath')
def main(model_uri, test_ds_uri, output_fpath):

    ds = dtoolcore.DataSet.from_uri(test_ds_uri)
    unetmodel = TrainedUNet(model_uri, input_channels=3)


    fpath_iter = (ds.item_content_abspath(idn) for idn in ds.identifiers)
    def make_thumbnail(im): return resize(im, (256, 256))
    def fpath_to_im(fpath): return fpath_to_composite_image(unetmodel, fpath)
    composite_images = map(fpath_to_im, fpath_iter)
    thumbnails = map(make_thumbnail, composite_images)
    grouped = grouper(thumbnails, 6)
    tiled = np.vstack([np.hstack(group) for group in grouped])

    tiled.view(Image).save(output_fpath)


if __name__ == "__main__":
    main()
