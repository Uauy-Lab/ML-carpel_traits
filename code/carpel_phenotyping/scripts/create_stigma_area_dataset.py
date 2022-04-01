import os
import pathlib

import click

from roi2json import roi_from_fpath

from utils import fpaths_to_resized_image_and_mask

from aiutils.data import image_mask_dataset_from_im_mask_iter


def im_fpath_roi_fpath_iter(im_fpath_iter, roi_dirpath):

    for im_fpath in im_fpath_iter:
        roi_fname = im_fpath.stem + '.roi'
        roi_fpath = roi_dirpath / roi_fname

        yield im_fpath, roi_fpath


@click.command()
@click.argument('images_dirpath')
@click.argument('roifiles_dirpath')
@click.argument('output_base_uri')
@click.argument('output_name')
def main(images_dirpath, roifiles_dirpath, output_base_uri, output_name):

    im_fpath_iter = pathlib.Path(images_dirpath).iterdir()
    roi_dirpath = pathlib.Path(roifiles_dirpath)
    fpaths_iter = im_fpath_roi_fpath_iter(im_fpath_iter, roi_dirpath)

    ims_masks = [
        fpaths_to_resized_image_and_mask(image_fpath, roi_fpath)
        for image_fpath, roi_fpath in fpaths_iter
    ]    

    image_mask_dataset_from_im_mask_iter(output_base_uri, output_name, ims_masks)


if __name__ == "__main__":
    main()