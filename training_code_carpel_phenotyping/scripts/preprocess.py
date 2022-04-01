import skimage.draw
import skimage.filters
import skimage.measure
import skimage.morphology

import numpy as np


def binarise_mask(mask):
    thresh = skimage.filters.threshold_otsu(mask)
    binary_mask = mask > thresh

    return binary_mask


def binary_mask_borders(binary_mask):
    eroded_mask = skimage.morphology.erosion(binary_mask)
    border_mask = eroded_mask ^ binary_mask
    return border_mask


def closest_point(available_points, p):
    v = np.array(list(available_points)) - p
    sq_dists = np.sum(v * v, axis=1)
    closest_index = np.argmin(sq_dists)
    sq_d = sq_dists[closest_index]
    return list(available_points)[closest_index], sq_d


def furthest_points(point_array):
    d_max = 0

    for p in point_array.T:
        d = point_array.T - p
        sq_dists = np.sum(d * d, axis=1)
        furthest_index = np.argmax(sq_dists)
        if sq_dists[furthest_index] > d_max:
            d_max = sq_dists[furthest_index]
            p0 = p
            p1 = point_array[:, furthest_index]

    return p0, p1


def annotate_with_points_and_line(im, p0, p1, colour=(255, 0, 0)):
    rr, cc = skimage.draw.circle(*p0, 5)
    im[rr, cc] = colour
    rr, cc = skimage.draw.circle(*p1, 5)
    im[rr, cc] = colour
    rr, cc = skimage.draw.line(*p0, *p1)
    im[rr, cc] = colour

    return im


def largest_mask_region(mask):

    mask_label = skimage.measure.label(mask)
    rprops = skimage.measure.regionprops(mask_label)
    
    by_area = {r.area: r.label for r in rprops}
    largest_area = sorted(by_area, reverse=True)[0]

    return mask_label == by_area[largest_area]


def mask_to_points(mask):

    binary_mask = binarise_mask(mask)
    largest_region = largest_mask_region(binary_mask)
    border_mask = binary_mask_borders(largest_region)
    point_array = np.array(np.where(border_mask))
    p0, p1 = furthest_points(point_array)

    return p0, p1
