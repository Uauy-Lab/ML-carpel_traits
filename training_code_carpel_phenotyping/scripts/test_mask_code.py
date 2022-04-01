import click
import dtoolcore

from dtoolbioimage import Image as dbiImage
from aiutils.unetmodel import TrainedUNet
from runtools.config import Config
from runtools.spec import ItemSpec

from maskutils import binarise_mask


# TODO - move to runtools.spec
def spec_to_abspath(ds, spec, template):
    relpath = spec.template_repr(template)
    idn = dtoolcore.utils.generate_identifier(relpath)
    return ds.item_content_abspath(idn)


def im_from_spec(ds, spec):
    image_template = "image_{n}.jpg"
    im_abspath = spec_to_abspath(ds, spec, image_template)
    return dbiImage.from_file(im_abspath)


@click.command()
@click.argument('config_fpath')
def main(config_fpath):

    config = Config.from_fpath(config_fpath)
    model = TrainedUNet(config.stigma_model_uri, input_channels=3)
    ds = dtoolcore.DataSet.from_uri(config.dataset_uri)

    spec = ItemSpec(n=15)

    im = im_from_spec(ds, spec)

    imask = model.scaled_mask_from_image(im)

    bmask = binarise_mask(imask)



if __name__ == "__main__":
    main()
