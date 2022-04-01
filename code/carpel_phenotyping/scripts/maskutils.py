import skimage.filters
import skimage.morphology


class BinaryMask(object):
    pass


def binarise_mask(mask: BinaryMask) -> BinaryMask:
    thresh = skimage.filters.threshold_otsu(mask)
    binary_mask = mask > thresh

    return binary_mask


def binary_mask_borders(binary_mask: BinaryMask) -> BinaryMask:
    eroded_mask = skimage.morphology.erosion(binary_mask)
    border_mask = eroded_mask ^ binary_mask
    
    return border_mask

