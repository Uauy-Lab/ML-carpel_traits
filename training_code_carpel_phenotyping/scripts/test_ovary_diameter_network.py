import random
import logging

import skimage.measure

import click

from aiutils.unetmodel import TrainedUNet
from aiutils.vis import visualise_image_and_mask
from runtools.data import IndexedDirtree, ImageDataSetView

from preprocess import mask_to_points, annotate_with_points_and_line, binarise_mask


logger = logging.getLogger("testnetwork")

def test_single_item(ids):
    idn = "2a5844852f04ab0519408b83505a3551b8717967"
    im = ids.item_by_idn(idn)
    pred_mask = trained_model.predict_mask_from_image(im)
    p0, p1 = mask_to_points(pred_mask)
    mask_vis = visualise_image_and_mask(im, pred_mask, mask_weight=0.2)
    ann = annotate_with_points_and_line(mask_vis, p0, p1)
    ann.save(f"ann-good.png")


@click.command()
@click.argument('data_dirpath')
def main(data_dirpath):

    logging.basicConfig(level=logging.INFO)

    idt = IndexedDirtree(data_dirpath)
    ids = ImageDataSetView(idt)

    logging.info(f"Loaded tree with {len(ids)} items")

    model_uri = "/Users/mhartley/models/ovaryareanet/"
    trained_model = TrainedUNet(model_uri, input_channels=3)


    for n, im in enumerate(ims):
        logging.info(f"Processing {n}")
        pred_mask = trained_model.predict_mask_from_image(im)
        binary_mask = binarise_mask(pred_mask)

        p0, p1 = mask_to_points(pred_mask)
        mask_vis = visualise_image_and_mask(im, pred_mask)
        ann = annotate_with_points_and_line(mask_vis, p0, p1)

        ann.save(f"ann{n}.png")

    # idns = random.sample(ids.identifiers, 10)





if __name__ == "__main__":
    main()
