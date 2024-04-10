import json
import os
import platform

import imagesize

from src.annotation.labels import Label


class LabelMeAnnotation:
    def __init__(self, path_related_image: str, img_size=None, labels=None):
        if labels is None:
            labels = []
        self.path_related_image = path_related_image

        if img_size is None:
            self.img_width, self.img_height = self.get_image_size_from_picture()
        else:
            self.img_width, self.img_height = img_size

        self.labels = labels

    @classmethod
    def from_file(cls, path_annotation_file: str):
        f = open(path_annotation_file)
        data = json.load(f)
        # labels = [i["points"] for i in data["shapes"]]
        labels = [Label(rectangle=i["points"], cls=i["label"]) for i in data["shapes"]]
        image_path = data["imagePath"]
        image_size = (data["imageWidth"], data["imageHeight"])
        return cls(image_path, image_size, labels)




    def write_ann_file(self, output_path: str) -> None:
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
            # if len(obj) == 5:
            #     x0, y0, x1, y1, _ = [int(i) for i in obj]
            #
            # elif len(obj) == 4:
            #     x0, y0, x1, y1 = [int(i) for i in obj]
            #
            # self.labels.append({
            #     'label': 'a',
            #     'points': [[x0, y0], [x1, y1]],
            #     'group_id': None,
            #     'shape_type': 'rectangle',
            #     'flags': {}
            # })
            self.labels.append(Label(rectangle=rectangle[:4], cls=c))

    @staticmethod
    def get_image_size_from_annotation_file(filepath: str):
        f = open(filepath)
        data = json.load(f)

        return [data["imageWidth"], data["imageHeight"]]

    def get_image_size_from_picture(self):
        return imagesize.get(self.path_related_image)
