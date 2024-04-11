from typing import List, Union

import cv2
import numpy as np


class TileCutter:
    def __init__(self, nb_cuts: int, image_shape: Union[List, np.array], marge: int):
        """
        The aim of this class is to create a mosaic from an image. It will cut the picture in several tiles.
        :param nb_cuts: square root of number of tiles
        :param image_shape: cut picture shape
        :param marge: overlap portion between tiles in pixels
        """
        self.marge = marge
        self.nb_cuts = nb_cuts
        self.image_shape = image_shape

        self.tiles = self.cut_image()

    def cut_image(self) -> List:
        """
        Create list of tiles in function of the number of cuts, the margin between tiles and the cut picture shape.
        :return: list of tiles coordinates in format [[xmin,ymin, xmax, ymax], ...]
        """
        tile_height = self.image_shape[0] // self.nb_cuts
        tile_width = self.image_shape[1] // self.nb_cuts

        boxes = []
        for i in range(0, self.image_shape[0], tile_height):
            for j in range(0, self.image_shape[1], tile_width):
                box = [j - self.marge, i - self.marge, j + tile_width + self.marge, i + tile_height + self.marge]
                if box[0] < 0:
                    box[0] = 0
                if box[1] < 0:
                    box[1] = 0
                if box[2] > self.image_shape[0]:
                    box[2] = self.image_shape[0]
                if box[3] > self.image_shape[1]:
                    box[3] = self.image_shape[1]

                boxes.append(box)

        return boxes
