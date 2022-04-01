import logging
import itertools
from types import SimpleNamespace

import click
import parse
import dtoolcore

from dtoolbioimage import Image as dbiImage

from aiutils.config import Config
from aiutils.data import image_mask_dataset_from_im_mask_iter

from roi2json import roi_from_fpath

from roiutils import roi_list_to_composite_mask
from utils import resize_im_mask_pair


logger = logging.getLogger(__file__)


class ItemSpec(SimpleNamespace):

    def template_repr(self, template):
        return template.format(**self.__dict__)


def parse_by_idn(ds, idn, parse_template):
    relpath = ds.item_properties(idn)["relpath"]
    return parse.parse(parse_template, relpath)


def ds_to_item_specs(ds, parse_template):
    parse_results = [
        parse_by_idn(ds, idn, parse_template)
        for idn in ds.identifiers
    ]

    item_specs = [
        ItemSpec(**result.named)
        for result in parse_results
        if result
    ]

    return item_specs


def image_from_item_spec(input_ds, item_spec, templates):

    fnames = [item_spec.template_repr(template) for template in templates]
    idns = [dtoolcore.utils.generate_identifier(fname) for fname in fnames]
    exists = [idn in input_ds.identifiers for idn in idns]

    assert exists.count(True) == 1
    idn = idns[exists.index(True)]

    image_fpath = input_ds.item_content_abspath(idn)

    return dbiImage.from_file(image_fpath)


def roi_from_item_spec(input_ds, item_spec, template):
    fname = item_spec.template_repr(template)
    idn = dtoolcore.utils.generate_identifier(fname)

    try:
        fpath = input_ds.item_content_abspath(idn)
    except KeyError:
        return None

    return roi_from_fpath(fpath)


def im_to_dim_2d(im):
    rdim, cdim, _ = im.shape
    return rdim, cdim


def mask_from_item_spec(ds, item_spec, dsconfig):
    roi_template = dsconfig['template']
    im_templates = dsconfig['image_templates']

    im = image_from_item_spec(ds, item_spec, im_templates)
    dim = im_to_dim_2d(im)
    roi = roi_from_item_spec(ds, item_spec, roi_template)
    mask = roi_list_to_composite_mask([roi], dim)

    return mask


def dsconfig_to_im_mask_iter(dsconfig):
    logger.info(f"Loading dataset {dsconfig['uri']}")
    ds = dtoolcore.DataSet.from_uri(dsconfig['uri'])

    specs = ds_to_item_specs(ds, dsconfig['template'])

    logging.info(f"{len(specs)} specs match template pattern")

    im_iter = (image_from_item_spec(ds, spec, dsconfig['image_templates']) for spec in specs)
    mask_iter = (mask_from_item_spec(ds, spec, dsconfig) for spec in specs)
    im_mask_iter = zip(im_iter, mask_iter)
    resized_im_mask_iter = (resize_im_mask_pair(*pair) for pair in im_mask_iter)

    return resized_im_mask_iter


@click.command()
@click.argument('config_fpath')
def main(config_fpath):

    logging.basicConfig(level=logging.INFO)

    config = Config.from_fpath(config_fpath)

    im_mask_iters = [
        dsconfig_to_im_mask_iter(dsconfig)
        for dsconfig in config.input_datasets
    ]
    chained_iter = itertools.chain(*im_mask_iters)
    image_mask_dataset_from_im_mask_iter(
        config.output_base_uri,
        config.output_name,
        chained_iter
    )

if __name__ == "__main__":
    main()
