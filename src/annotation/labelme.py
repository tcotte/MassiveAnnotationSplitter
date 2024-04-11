import json
import os
import platform
from typing import Tuple, List, Any

import imagesize

from src.annotation.labels import Label


class LabelMeAnnotation:
    def __init__(self, path_related_image: str, img_size=None, labels=None):
        """
        The aim of this class is to read and write LabelMe annotation file in the simplest way.
        This class supports only rectangles as shape.
        :param path_related_image: path of image related to this annotation file
        :param img_size: size of picture related to this annotation file
        :param labels: declared labels in this imported annotation files
        """
        if labels is None:
            labels = []
        self.path_related_image = path_related_image

        if img_size is None:
            self.img_width, self.img_height = self.get_image_size_from_picture()
        else:
            self.img_width, self.img_height = img_size

        self.labels = labels

    @classmethod
    def from_file(cls, path_annotation_file: str) -> "LabelMeAnnotation":
        """
        Read LabelMe annotation file and transmit its information into a LabelMeAnnotation class
        :param path_annotation_file: path of the annotation file
        :return: LabelMeAnnotation object
        """
        f = open(path_annotation_file)
        data = json.load(f)
        # get labels with shape and class name
        labels = [Label(rectangle=i["points"], cls=i["label"]) for i in data["shapes"]]
        image_path = data["imagePath"]
        image_size = (data["imageWidth"], data["imageHeight"])
        return cls(image_path, image_size, labels)

    def write_ann_file(self, output_path: str) -> None:
        """
        Write .json annotation file gathering all information contained in this class
        :param output_path: path of the .json annotation file
        """
        annotation = {
            "version": "5.2.1",
            "flags": {},
            "shapes": [label.labelme_format() for label in self.labels],
            "imagePath": os.path.split(self.path_related_image)[1],
            "imageData": None,
            "imageHeight": self.img_height,
            "imageWidth": self.img_width
        }

        with open(output_path, 'w') as outfile:
            json.dump(annotation, outfile)

    def add_rectangles(self, rectangles, cls) -> None:
        """
        Bounding boxes in format: x0, y0, x1, y1 to LabelMe labels. These rectangles will be added in labels attribute.
        :param rectangles: Bounding boxes in format: x0, y0, x1, y1
        """
        for rectangle, c in zip(rectangles, cls):
            self.labels.append(Label(rectangle=rectangle[:4], cls=c))

    @staticmethod
    def get_image_size_from_annotation_file(filepath: str) -> list[int]:
        """
        Get image size from annotation file
        :param filepath: path of annotation files
        :return: image width and image height
        """
        f = open(filepath)
        data = json.load(f)

        return [data["imageWidth"], data["imageHeight"]]

    def get_image_size_from_picture(self) -> Tuple[int, int]:
        """
        Get image size from the picture details
        :return: path of picture
        """
        return imagesize.get(self.path_related_image)
