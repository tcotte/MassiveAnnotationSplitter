from typing import Union, List, Tuple, Dict

import numpy as np


class BoundingBox:
    def __init__(self, x1: Union[int, np.int32], y1: Union[int, np.int32], x2: Union[int, np.int32],
                 y2: Union[int, np.int32]):
        """
        Bounding box representation instantiated in xyxy format.
        :param x1: left top corner abscissa of bounding box
        :param y1: left top corner ordinate of bounding box
        :param x2: right bottom corner abscissa of bounding box
        :param y2: right bottom corner ordinate of bounding box
        """
        self.x1 = int(x1)
        self.y1 = int(y1)
        self.x2 = int(x2)
        self.y2 = int(y2)

    @classmethod
    def from_points(cls, pt1: Tuple, pt2: Tuple) -> "BoundingBox":
        """
        Create bounding box from two points which represent left top and right bottom corners of the bounding box.
        :param pt1: left top corner of bounding box (xy)
        :param pt2: right bottom corner of bounding box (xy)
        :return: Bounding box instanciation
        """
        return cls(pt1[0], pt1[1], pt2[0], pt2[1])

    def pt_format(self) -> list[list[int]]:
        """
        :return: bounding box point format -> shape = (2,2)
        """
        return [[self.x1, self.y1], [self.x2, self.y2]]

    def flatten_format(self):
        """
        :return: flattened bounding box format -> shape = (4,)
        """
        return [self.x1, self.y1, self.x2, self.y2]

    def translation(self, translation_x, translation_y) -> None:
        """
        Transform bounding box coordinates to translated coordinates. It comes to move the bounding box.
        :param translation_x: translation abscissa offset
        :param translation_y: translation ordinate offset
        """
        self.x1 += translation_x
        self.x2 += translation_x
        self.y1 += translation_y
        self.y2 += translation_y


class Label:
    def __init__(self, rectangle: Union[BoundingBox, List, np.ndarray], cls: str):
        """
        Rectangle label representation with bounding box and class
        :param rectangle: bounding box which represents the label shape
        :param cls: item's class
        """
        if isinstance(rectangle, BoundingBox):
            self.rectangle = rectangle
        # if the shape passed as parameter is not a BoundingBox class -> get the rectangle coordinate to create it
        else:
            if isinstance(rectangle, list):
                rectangle = np.array(rectangle)
                if rectangle.shape == (2, 2):
                    self.rectangle = BoundingBox.from_points(rectangle[0], rectangle[1])
                else:
                    self.rectangle = BoundingBox(*rectangle)

            elif isinstance(rectangle, np.ndarray):
                if rectangle.size == 2:
                    self.rectangle = BoundingBox.from_points(rectangle[0], rectangle[1])
                else:
                    self.rectangle = BoundingBox(*rectangle)

        self.cls = cls

    def labelme_format(self) -> Dict:
        """
        :return: label in dictionary understood by Labelme (in json annotation file).
        """
        return {
            'label': self.cls,
            'points': self.rectangle.pt_format(),
            'group_id': None,
            'shape_type': 'rectangle',
            'flags': {}
        }

    def supervisely_format(self) -> Dict:
        """
        :return: label in dictionary understood by Supervisely (in json annotation file).
        """
        return {
                "geometryType": "rectangle",
                "classTitle": self.cls,
                "points": {
                    "exterior": self.rectangle.pt_format(),
                    "interior": []
                }
        }

