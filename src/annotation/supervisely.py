import json
import os
import platform

import imagesize

from src.annotation.labels import Label


def get_slylabelfile_from_imgfile(img_path: str) -> str:
    """
    Get path of Supervisely annotation file from the related image file path.
    :param img_path: picture path related to this annotation file
    :return: path of the annotation file
    """
    path = os.path.abspath(img_path)
    splitted_path = path.split(os.sep)
    if platform.system() == "Windows":
        label_path = os.path.join("C:\\", *splitted_path[1:-2], "ann", splitted_path[-1] + ".json")

        if not os.path.isdir(os.path.join("C:\\", *splitted_path[1:-2], "ann")):
            os.makedirs(os.path.join("C:\\", *splitted_path[1:-2], "ann"))
        return label_path
    else:
        label_path = os.path.join(*splitted_path[2:-1], "../sly/ann", splitted_path[-1] + ".json")
        return os.path.abspath(label_path)

class SuperviselyAnnotation:
    def __init__(self, img_size=None, labels=None, tags=None):
        """
        The aim of this class is to read and write Supervisely annotation file in the simplest way.
        This class supports only rectangles as shape.
        :param img_size: size of picture related to this annotation file
        :param labels: declared labels in this imported annotation files
        :param tags: declared tags in this imported annotation files
        """
        if tags is None:
            tags = []
        if labels is None:
            labels = []

        self.img_width, self.img_height = img_size

        self.labels = labels
        self.tags = tags

    @classmethod
    def from_file(cls, path_annotation_file: str) -> "SuperviselyAnnotation":
        """
        Read Supervisely annotation file and transmit its information into a SuperviselyAnnotation class
        :param path_annotation_file: path of the annotation file
        :return: SuperviselyAnnotation object
        """
        f = open(path_annotation_file)
        data = json.load(f)
        labels = [Label(rectangle=d["points"]["exterior"], cls=d["classTitle"]) for d in data["objects"]]
        image_size = (data["size"]["width"], data["size"]["height"])
        return cls(image_size, labels)

    @staticmethod
    def get_imgfile_from_slylabelfile(label_path: str) -> str:
        """
        Get path of image file from the related Supervisely annotation file path.
        :param label_path: annotation path related to this picture file
        :return: path of the picture file
        """
        path = os.path.normpath(label_path)
        splitted_path = path.split(os.sep)
        if platform.system() == "Windows":
            label_path = os.path.join("C:\\", *splitted_path[1:-2], "img", splitted_path[-1][:-4])
            return label_path
        else:
            label_path = os.path.join(*splitted_path[2:-2], "img", splitted_path[-1][:-4])
            return os.path.abspath(label_path)

    def get_image_size_from_picture(self) -> [int, int]:
        """
        Get image size from picture details
        :return: image size at format -> [width, height]
        """
        return imagesize.get(self.path_related_image)

    def write_ann_file(self, output_path: str) -> None:
        """
        Write .json annotation file gathering all information contained in this class
        :param output_path: path of the .json annotation file
        """
        annotation = {
            "description": "",
            "size": {
                "height": self.img_height,
                "width": self.img_width
            },
            "objects": [label.supervisely_format() for label in self.labels],
            "tags": self.tags
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
