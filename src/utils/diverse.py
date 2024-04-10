import os
import random
import shutil

import cv2
import matplotlib.colors as mcolors
import numpy as np


def filter_predictions_by_tile(bboxes, tile):
    filtered_idx = []
    x0, y0, x1, y1 = tile
    for idx, box in enumerate(bboxes):
        if box[0] >= x0:
            if box[1] >= y0:
                if box[2] <= x1:
                    if box[3] <= y1:
                        filtered_idx.append(idx)
    return filtered_idx


def random_color_generator() -> (int, int, int):
    color = random.choice(list(mcolors.CSS4_COLORS.values()))
    return color


def nms_python(bboxes: np.array, psocres: np.array, threshold: int):
    '''
    NMS: first sort the bboxes by scores ,
        keep the bbox with highest score as reference,
        iterate through all other bboxes,
        calculate Intersection Over Union (IOU) between reference bbox and other bbox
        if iou is greater than threshold,then discard the bbox and continue.

    Input:
        bboxes(numpy array of tuples) : Bounding Box Proposals in the format (x_min,y_min,x_max,y_max).
        pscores(numpy array of floats) : confidance scores for each bbox in bboxes.
        threshold(float): Overlapping threshold above which proposals will be discarded.

    Output:
        filtered_bboxes(numpy array) :selected bboxes for which IOU is less than threshold.
    '''
    # Unstacking Bounding Box Coordinates
    bboxes = bboxes.astype('float')
    x_min = bboxes[:, 0]
    y_min = bboxes[:, 1]
    x_max = bboxes[:, 2]
    y_max = bboxes[:, 3]

    # Sorting the pscores in descending order and keeping respective indices.
    sorted_idx = psocres.argsort()[::-1]
    # Calculating areas of all bboxes.Adding 1 to the side values to avoid zero area bboxes.
    bbox_areas = (x_max - x_min + 1) * (y_max - y_min + 1)

    # list to keep filtered bboxes.
    filtered = []
    idx_to_rm = []
    while len(sorted_idx) > 0:
        # Keeping highest pscore bbox as reference.
        rbbox_i = sorted_idx[0]
        # Appending the reference bbox index to filtered list.
        filtered.append(rbbox_i)

        # Calculating (xmin,ymin,xmax,ymax) coordinates of all bboxes w.r.t to reference bbox
        overlap_xmins = np.maximum(x_min[rbbox_i], x_min[sorted_idx[1:]])
        overlap_ymins = np.maximum(y_min[rbbox_i], y_min[sorted_idx[1:]])
        overlap_xmaxs = np.minimum(x_max[rbbox_i], x_max[sorted_idx[1:]])
        overlap_ymaxs = np.minimum(y_max[rbbox_i], y_max[sorted_idx[1:]])

        # Calculating overlap bbox widths,heights and there by areas.
        overlap_widths = np.maximum(0, (overlap_xmaxs - overlap_xmins + 1))
        overlap_heights = np.maximum(0, (overlap_ymaxs - overlap_ymins + 1))
        overlap_areas = overlap_widths * overlap_heights

        # Calculating IOUs for all bboxes except reference bbox
        ious = overlap_areas / (bbox_areas[rbbox_i] + bbox_areas[sorted_idx[1:]] - overlap_areas)

        # select indices for which IOU is greather than threshold
        delete_idx = np.where(ious > threshold)[0] + 1

        if len(delete_idx):
            idx_to_rm.extend(list(delete_idx))

        delete_idx = np.concatenate(([0], delete_idx))

        # delete the above indices
        sorted_idx = np.delete(sorted_idx, delete_idx)

    # Return filtered bboxes
    return bboxes[filtered].astype('int'), idx_to_rm


def draw_tiles(image: np.array, tiles: np.array) -> np.array:
    color = (255, 0, 0)
    thickness = 2

    for box in tiles:
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, thickness)
    return image


def rectangle_translation(rectangle, translation_x, translation_y) -> [int, int, int, int]:
    if len(rectangle) == 2:
        pt1 = rectangle[0]
        pt2 = rectangle[1]

    elif len(rectangle) == 4 or len(rectangle) == 5:
        pt1 = [rectangle[0], rectangle[1]]
        pt2 = [rectangle[2], rectangle[3]]

    x1, y1 = pt1
    x2, y2 = pt2
    return [x1 + translation_x, y1 + translation_y, x2 + translation_x, y2 + translation_y]


def flatten_rectangle(rectangle):
    return [rectangle[0][0], rectangle[0][1], rectangle[1][0], rectangle[1][1]]


def clean_directory(directory_path):
    for file in os.listdir(directory_path):
        if os.path.isfile(file):
            os.remove(os.path.join(directory_path, file))
        else:
            shutil.rmtree(file, ignore_errors=True)
