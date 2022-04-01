import itertools

import click
import parse
import dtoolcore
import ruamel.yaml

from dtoolbioimage import Image as dbiImage
from aiutils.data import image_mask_dataset_from_im_mask_iter
from roi2json import roi_from_fpath

from utils import resize_im_mask_pair
from roiutils import roi_to_mask, roi_list_to_composite_mask


def filtered_by_relpath(ds, relpath_filter):
    for idn in ds.identifiers:
        relpath = ds.item_properties(idn)['relpath']
        if relpath_filter(relpath):
            yield idn


def parse_by_idn(input_ds, idn, parse_template):
    relpath = input_ds.item_properties(idn)["relpath"]
    return parse.parse(parse_template, relpath)


def get_item_specifiers(input_ds, parse_template):
    parse_results = [
        parse_by_idn(input_ds, idn, parse_template)
        for idn in input_ds.identifiers
    ]

    item_specs = [
        result.named
        for result in parse_results
        if result
    ]

    return item_specs



def image_from_item_spec(input_ds, item_spec):

    templates = [
        "image_{n}.jpg",
        "image_{n}.tif"
    ]

    fnames = [template.format(**item_spec) for template in templates]
    idns = [dtoolcore.utils.generate_identifier(fname) for fname in fnames]
    exists = [idn in input_ds.identifiers for idn in idns]

    assert exists.count(True) == 1
    idn = idns[exists.index(True)]

    image_fpath = input_ds.item_content_abspath(idn)

    return dbiImage.from_file(image_fpath)


def roi_from_item_spec(input_ds, item_spec, parse_template):
    fname = parse_template.format(**item_spec)
    idn = dtoolcore.utils.generate_identifier(fname)

    try:
        fpath = input_ds.item_content_abspath(idn)
    except KeyError:
        return None

    # fpath = input_ds.item_content_abspath(idn)
    return roi_from_fpath(fpath)


def im_to_dim_2d(im):
    rdim, cdim, _ = im.shape
    return rdim, cdim


def input_ds_to_im_mask_iter(input_ds, parse_templates):

    item_specs = get_item_specifiers(input_ds, parse_templates[0])

    #FIXME - Calling the function twice is horrible
    rois = [
        [
            roi_from_item_spec(input_ds, item_spec, parse_template)
            for parse_template in parse_templates
            if roi_from_item_spec(input_ds, item_spec, parse_template) is not None
        ]
        for item_spec in item_specs
    ]
    images = [
        image_from_item_spec(input_ds, item_spec)
        for item_spec in item_specs
    ]
    masks = (
        roi_list_to_composite_mask(roi_list, im_to_dim_2d(im))
        for roi_list, im in zip(rois, images)
    )
    resized_ims_masks = (
        resize_im_mask_pair(im, mask)
        for im, mask in zip(images, masks)
    )

    return resized_ims_masks


def im_mask_iter_from_input_dataset_config(input_dataset_config):
    input_ds_uri = input_dataset_config["uri"]
    input_ds = dtoolcore.DataSet.from_uri(input_ds_uri)
    im_mask_iter = input_ds_to_im_mask_iter(input_ds, input_dataset_config["template"])

    return im_mask_iter


@click.command()
@click.argument('yaml_fpath')
def main(yaml_fpath):
    yaml = ruamel.yaml.YAML()
    with open(yaml_fpath) as fh:
        config = yaml.load(fh)

    im_mask_iters = [
        im_mask_iter_from_input_dataset_config(input_dataset_config)
        for input_dataset_config in config["input_datasets"]
    ]

    output_base_uri = config["output_base_uri"]
    output_name = config["output_name"]

    chained_iter = itertools.chain(*im_mask_iters)
    image_mask_dataset_from_im_mask_iter(output_base_uri, output_name, chained_iter)


if __name__ == "__main__":
    main()