import logging

import PIL
import click
import dtoolcore

import pandas as pd
import numpy as np

from dtoolbioimage import scale_to_uint8, Image as dbiImage
from PIL import ImageDraw, ImageFont

from aiutils.data import ImageMaskDataSet, LimitDataSetWrapper
from aiutils.unetmodel import TrainedUNet


def annotate_with_strings(im, strings):
    pilim = PIL.Image.fromarray(scale_to_uint8(im))
    draw = ImageDraw.ImageDraw(pilim)
    font = PIL.ImageFont.truetype("Microsoft Sans Serif.ttf", size=24)

    for n, string in enumerate(strings):
        draw.text((10, 24 * n),  string, font=font, fill=(255, 255, 255))

    return pilim


def visualise_masks(im, mask, pred_mask, strings):

    im_numpy = np.transpose(im.numpy(), (1, 2, 0))

    mask_uint8 = scale_to_uint8(mask.numpy())
    pred_mask_uint8 = scale_to_uint8(pred_mask)

    rdim, cdim = pred_mask.shape
    mask_vis = np.zeros((rdim, cdim, 3), dtype=np.uint8)

    mask_vis[:,:,0] = pred_mask_uint8
    mask_vis[:,:,1] = mask_uint8

    merged = 0.5 * mask_vis + 0.5 * scale_to_uint8(im_numpy)

    pilim = annotate_with_strings(merged, strings)

    return pilim


@click.command()
@click.argument('model_uri')
@click.argument('evaluation_dataset_uri')
@click.argument('output_base_uri')
@click.argument('output_name')
def main(model_uri, evaluation_dataset_uri, output_base_uri, output_name):

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("ModelEvaluation")

    eval_ds = ImageMaskDataSet(evaluation_dataset_uri)
    unetmodel = TrainedUNet(model_uri, input_channels=3)

    # lds = LimitDataSetWrapper(eval_ds, limit=10)

    results = []
    ds = eval_ds
    with dtoolcore.DerivedDataSetCreator(output_name, output_base_uri, eval_ds) as output_ds:
        for n, (im, mask) in enumerate(ds, 1):
            logger.info(f"Processing {n}/{len(ds)}")
            pred_mask = unetmodel.predict_mask_from_tensor(im)

            mflat = (mask.numpy().flatten() > 0.1)
            pflat = (pred_mask.flatten() > 0.1)

            dice_score = 2 * sum(mflat * pflat) / (sum(mflat) + sum(pflat))
            area_ratio = sum(pflat) / sum(mflat)
            dice_score_str = f"dice score: {dice_score:.2f}"
            area_ratio_str = f"area ratio: {area_ratio:.2f}"

            result = {
                "dice_score": dice_score,
                "area_ratio": area_ratio
            }
            results.append(result)

            vis = visualise_masks(im, mask, pred_mask, [dice_score_str, area_ratio_str])
            output_abspath = output_ds.prepare_staging_abspath_promise(f"eval{n}.png")
            vis.save(output_abspath)

        df = pd.DataFrame(results)
        summary_abspath = output_ds.prepare_staging_abspath_promise("metrics.csv")
        df.to_csv(summary_abspath, index=False, float_format='%.2f')


if __name__ == "__main__":
    main()