import json
import os
import platform
from typing import List

import imagesize
import numpy as np

from src.annotation.labelme import LabelMeAnnotation
from src.annotation.supervisely import SuperviselyAnnotation
from diverse import nms_python

SHOW_NMS = False


class AnnotationUnifier:
    def __init__(self, path_annotations: str, margin: int, whole_image_path: str, format="supervisely"):
        """
        The aim of this class is to unify annotations from tiles and create one annotation file.
        :param path_annotations: input annotation tiles folder
        :param margin: margin used to cut whole picture
        :param whole_image_path: image of the recomposed picture
        :param format: format of decomposition and recomposition (could be "supervisely" or "labelme")
        """
        self.format = format
        self.path_annotations = path_annotations
        self.annotations_files_list = [os.path.join(path_annotations, i) for i in sorted(self.get_annotation_files())]
        self.margin = margin
        self.whole_image_path = whole_image_path

    def get_annotation_files(self) -> List:
        """
        Get list of annotation file paths. It can be easily retrieved filtering the .json extension in the input folder.
        :return: list of annotation file paths
        """
        annotations_files = []
        for filename in os.listdir(self.path_annotations):
            if filename.endswith(".json"):
                annotations_files.append(filename)
        return annotations_files

    @staticmethod
    def get_image_size_from_annotation_file(filepath: str) -> [int, int]:
        """
        Get image size opening and reading Labelme annotation file
        :param filepath: annotation filepath
        :return: [image width, image height]
        """
        f = open(filepath)
        data = json.load(f)

        return [data["imageWidth"], data["imageHeight"]]

    def get_tile_number(self) -> int:
        """
        Get number of tiles in the mosaic
        :return: number of tiles in the mosaic
        """
        return np.sqrt(len(self.annotations_files_list)).astype(int)

    def get_position_in_grid(self, image_index: int) -> [int, int]:
        """
        Get position of tile in the mosaic depending on the image index.
        For example, if there are 9 tiles and the image index is 5, then the position return will be [2, 2]
        :param image_index: image index in the mosaic
        :return: position of the image in the mosaic -> [x position, y position]
        """
        nb_images_in_line = self.get_tile_number()
        position_x = image_index % nb_images_in_line
        position_y = image_index // nb_images_in_line
        return position_x, position_y

    @staticmethod
    def get_slylabelfile_from_imgfile(img_path: str) -> None:
        """
        Get path of Supervise.ly label file from path of image file.
        :param img_path: path of image file
        :return: path of Supervise.ly label file
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

    def unify_annotations(self) -> None:
        """
        Unify annotations from tiles into only one annotation file translating the annotations from tiles.
        """
        whole_picture_labels = []

        current_width = 0
        current_height = 0

        for index, annotation_file in enumerate(self.annotations_files_list):
            annotations = []
            if self.format == "labelme":
                annotations = LabelMeAnnotation.from_file(path_annotation_file=annotation_file)
            elif self.format == "supervisely":
                annotations = SuperviselyAnnotation.from_file(path_annotation_file=annotation_file)

            pos_x, pos_y = self.get_position_in_grid(image_index=index)
            img_height = annotations.img_height
            img_width = annotations.img_width

            [label.rectangle.translation(translation_x=current_width, translation_y=current_height)
             for label in annotations.labels]

            for label in annotations.labels:
                whole_picture_labels.append(label)

            if (index + 1) % self.get_tile_number() == 0:
                current_width = 0
                current_height += img_height - 2 * self.margin
            else:
                current_width += img_width - 2 * self.margin

        # Remove duplicate of annotation
        if len(whole_picture_labels) > 0:
            nms_picture_rectangles, idx_removed_by_nms = nms_python(
                bboxes=np.array([label.rectangle.flatten_format() for label in whole_picture_labels]),
                psocres=np.array([0.5] * len(whole_picture_labels)),
                threshold=0.5)
            nms_labels = [i for j, i in enumerate(whole_picture_labels) if j not in idx_removed_by_nms]

        else:
            nms_picture_rectangles = []
            nms_labels = []

        if self.format == "labelme":
            # Write Labelme file
            labelme_ann = LabelMeAnnotation(path_related_image=self.whole_image_path)
            labelme_ann.add_rectangles(rectangles=nms_picture_rectangles, cls=[label.cls for label in nms_labels])
            labelme_ann.write_ann_file(output_path=self.whole_image_path[:-4] + '.json')

        elif self.format == "supervisely":
            # Write Supervise.ly file
            sly_ann = SuperviselyAnnotation(img_size=imagesize.get(self.whole_image_path))
            sly_ann.add_rectangles(rectangles=nms_picture_rectangles, cls=[label.cls for label in nms_labels])
            sly_ann.write_ann_file(output_path=self.get_slylabelfile_from_imgfile(self.whole_image_path))


if __name__ == "__main__":
    au = AnnotationUnifier(
        path_annotations=r"C:\Users\tristan_cotte\Pictures\tile_set\29-03-2024_15-09-44\ann",
        whole_image_path=r"/new_picture.jpg",
        margin=20)
    ann_file = au.annotations_files_list[0]
    # au.get_objects_from_annotation_file(ann_file)
    au.unify_annotations()
