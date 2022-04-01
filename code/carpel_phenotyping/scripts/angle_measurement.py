import dtoolcore
import numpy as np

from dtoolbioimage import Image as dbiImage

from aiutils.maskutils import binarise_mask, binary_mask_borders
from roi2json import roi_from_fpath


def spec_to_abspath(ds, spec, template):
    relpath = spec.template_repr(template)
    idn = dtoolcore.utils.generate_identifier(relpath)
    return ds.item_content_abspath(idn)


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def angle_between_points(points):
    p0, p1, p2 = points
    angle_rad = angle_between(p0-p1, p2-p1)
    angle_deg = 360 * angle_rad / (2 * np.pi)
    return angle_deg


def im_from_spec(ds, spec):
    image_template = "image_{n}.jpg"
    im_abspath = spec_to_abspath(ds, spec, image_template)
    return dbiImage.from_file(im_abspath)


def roi_from_spec(ds, spec):
    angle_roi_template = "image_{n}_angle.roi"
    angle_roi_abspath = spec_to_abspath(ds, spec, angle_roi_template)
    roi = roi_from_fpath(angle_roi_abspath)
    return roi


def points_from_roi(roi):
    raw_coords = roi['coords']

    return [np.array((r, c)) for c, r in raw_coords]


def measure_from_known_roi(ds, spec):
    roi = roi_from_spec(ds, spec)
    points = points_from_roi(roi)

    angle = angle_between_points(points)

    measurement = {
        "angle": angle,
        "points": points
    }

    return measurement


def imask_to_points(mask):
    binary_mask = binarise_mask(mask)
    mask_perimeter = binary_mask_borders(binary_mask)
    return np.array(np.where(mask_perimeter)).T


def point_with_lowest_metric(points, metric):
    by_metric = {metric(p): p for p in points}
    min_metric = sorted(by_metric)[0]
    return by_metric[min_metric]


def point_from_ovary_area(ds, spec, ov_model):
    im = im_from_spec(ds, spec)
    imask = ov_model.scaled_mask_from_image(im)

    border_points = imask_to_points(imask)

    def upness(p): return p[0]

    p = point_with_lowest_metric(border_points, upness)

    return p


def points_from_branch_area(ds, spec, branch_model):
    im = im_from_spec(ds, spec)
    imask = branch_model.scaled_mask_from_image(im)

    border_points = imask_to_points(imask)

    def leftness(p): return p[1]
    def rightness(p): return -p[1]

    p0 = point_with_lowest_metric(border_points, leftness)
    p2 = point_with_lowest_metric(border_points, rightness)

    return p0, p2


def measure_from_models(ds, spec, ov_model, branch_model):
    p1 = point_from_ovary_area(ds, spec, ov_model)
    p0, p2 = points_from_branch_area(ds, spec, branch_model)

    points = p0, p1, p2
    angle = angle_between_points(points)

    measurement = {
        "angle": angle,
        "points": points
    }

    return measurement
