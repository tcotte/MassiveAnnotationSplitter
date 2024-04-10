from typing import List

import cv2
import numpy as np
from imutils.paths import list_images

import diverse
from tile_cutter import TileCutter


class TileUnifier:
    def __init__(self, path_images: List, margin: int):
        self.margin = margin
        self.path_images = path_images
        self.images_list = sorted(list(list_images(self.path_images)))
        self.tile_number = self.get_tile_number()
        self.original_image_shape = (self.tile_number - 1) * self.margin + self.tile_number

    def get_tile_number(self) -> int:
        return np.sqrt(len(self.images_list)).astype(int)

    def get_position_in_grid(self, image_index: int) -> [int, int]:
        nb_images_in_line = self.get_tile_number()
        position_x = image_index % nb_images_in_line
        position_y = image_index // nb_images_in_line
        return position_x, position_y

    def unify(self) -> None:
        self.recompose_picture()

    def recompose_picture(self):
        gathered_picture = []
        horizontal_picture = []

        for j in range(self.get_tile_number()):
            if j == 0:
                # Horizontal stacking
                horizontal_picture = self.get_horizontal_line_picture(line_number=j)[:-self.margin, :]

            else:
                new_picture = self.get_horizontal_line_picture(line_number=j)

                if not j == self.get_tile_number() - 1:
                    gathered_picture = np.vstack((horizontal_picture, new_picture[self.margin:-self.margin, :]))
                else:
                    gathered_picture = np.vstack((horizontal_picture, new_picture[self.margin:, :]))

                horizontal_picture = gathered_picture

        return gathered_picture

    def get_horizontal_line_picture(self, line_number):
        new_picture = cv2.imread(self.images_list[line_number * self.get_tile_number()])[:, :-self.margin]

        for i in range(1, self.get_tile_number()):
            if not i == self.get_tile_number() - 1:
                new_picture = np.hstack((new_picture,
                                         cv2.imread(self.images_list[i + line_number * self.get_tile_number()])[:,
                                         self.margin:-self.margin]))
            else:
                new_picture = np.hstack((new_picture,
                                         cv2.imread(self.images_list[i + line_number * self.get_tile_number()])[:,
                                         self.margin:]))

        return new_picture

    def get_indexes_of_picture_from_column(self, column_number: int) -> List:
        a = list(range(self.get_tile_number() ** 2))
        my_filter = [i % self.get_tile_number() == column_number for i in a]
        index = [i for i in range(len(my_filter)) if my_filter[i]]
        return [a[i] for i in index]

    def get_vertical_line_picture(self, column_number):
        indexes = self.get_indexes_of_picture_from_column(column_number)

        new_picture = cv2.imread(self.images_list[indexes[0]])[:-self.margin, :]

        for i in indexes[1:]:
            if not i == indexes[-1]:
                new_picture = np.vstack((new_picture, cv2.imread(self.images_list[i])[self.margin:-self.margin, :]))
            else:
                new_picture = np.vstack((new_picture, cv2.imread(self.images_list[i])[self.margin:, :]))

        return new_picture



if __name__ == "__main__":
    tile_unifier = TileUnifier(
        path_images=r"C:\Users\tristan_cotte\Pictures\AMES-positive_controls\positive_controls\29-03-2024_15-09-44.jpg",
        margin=20)

    recomposed_picture = tile_unifier.recompose_picture()
    # cv2.imwrite("new_picture.jpg", tile_unifier.recompose_picture())

    tile_cutter = TileCutter(nb_cuts=tile_unifier.get_tile_number(), marge=20, image_shape=recomposed_picture.shape[:2])
    tiles = tile_cutter.tiles

    cv2.imwrite("new_picture.jpg", diverse.draw_tiles(recomposed_picture, tiles))

    # index = 0
    # cv2.imwrite(f"col_{index}.png", tile_unifier.get_vertical_line_picture(column_number=index))
