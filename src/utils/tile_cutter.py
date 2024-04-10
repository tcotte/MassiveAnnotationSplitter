import cv2


class TileCutter:
    def __init__(self, nb_cuts, image_shape, marge):
        self.marge = marge
        self.nb_cuts = nb_cuts
        self.image_shape = image_shape

        self.tiles = self.cut_image()

    def cut_image(self):

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


def draw_tiles(image, tiles):
    color = (255, 0, 0)
    thickness = 2

    for box in tiles:
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, thickness)
    return image