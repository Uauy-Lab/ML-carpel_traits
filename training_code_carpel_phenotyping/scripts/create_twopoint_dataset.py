import click
import dtoolcore

import skimage.transform

from dtoolbioimage import Image

from utils import (
    image_numbers_from_suffix,
    image_n_to_image_fpath
)

from roi2json import roi_from_fpath


def roi_fpath_to_line_coords(roi_fpath):
    roi = roi_from_fpath(roi_fpath)

    r0, c0 = int(roi['y1']), int(roi['x1'])
    r1, c1 = int(roi['y2']), int(roi['x2'])

    return ((r0, c0), (r1, c1))


def image_n_to_coords(ds, n):
    idn = dtoolcore.utils.generate_identifier(f"image_{n}_ovary_diam.roi")
    roi_fpath = ds.item_content_abspath(idn)

    return roi_fpath_to_line_coords(roi_fpath)


def image_n_to_image_and_coords(ds, n):

    dim = (512, 512)
    im_fpath = image_n_to_image_fpath(ds, n)
    im = Image.from_file(im_fpath)
    resized_im = skimage.transform.resize(im, dim)

    coords = image_n_to_coords(ds, n)
    rdim, cdim, _ = im.shape
    (r0, c0), (r1, c1) = coords
    scaled_coords = (r0/rdim, c0/cdim), (r1/rdim, c1/cdim)

    return resized_im.view(Image), scaled_coords



@click.command()
@click.argument('input_ds_uri')
@click.argument('output_base_uri')
@click.argument('output_name')
def main(input_ds_uri, output_base_uri, output_name):

    ds = dtoolcore.DataSet.from_uri(input_ds_uri)
    image_numbers = image_numbers_from_suffix(ds, "ovary_diam.roi")


    source_dataset = ds
    with dtoolcore.DerivedDataSetCreator(
        output_name,
        output_base_uri,
        source_dataset
    ) as output_ds:
        for n in image_numbers:
            im, coords = image_n_to_image_and_coords(ds, n)
            im_relpath = f"image{n:02d}.png"
            im_abspath = output_ds.prepare_staging_abspath_promise(im_relpath)
            im.save(im_abspath)
            output_ds.add_item_metadata(im_relpath, "coords", coords)


if __name__ == "__main__":
    main()
