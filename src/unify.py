import argparse
import os
from typing import Final

import cv2

from src.utils.annotation_unifier import AnnotationUnifier
from src.utils.tile_unifier import TileUnifier

FORMAT_DICT: Final = {1: 'labelme', 2: 'supervisely'}


def unify_tiles_and_annotations(input_cut_image_folder_path: str, input_cut_ann_folder_path: str, output_folder: str,
                                margin: int, format: str) -> None:
    """
    Unify tiles and gather the annotations to recompose the whole picture with its annotations.
    :param input_cut_image_folder_path: path of the image tiles folder
    :param input_cut_ann_folder_path: path of the annotation tiles folder
    :param output_folder: path of the folder where the output will be delivered
    :param margin: margin used to divide the image into mosaic
    :param format: input/output annotation format
    :return: None
    """
    # Unify tiles to recompose picture
    tile_unifier = TileUnifier(path_images=input_cut_image_folder_path, margin=margin)
    # Save the recomposed picture
    path_recomposed_image = os.path.join(output_folder, f"{input_cut_image_folder_path.split(os.sep)[-1]}.jpg")
    cv2.imwrite(path_recomposed_image, tile_unifier.recompose_picture())

    # Gather annotation tiles and reframe it in function of their tile position
    au = AnnotationUnifier(
        path_annotations=input_cut_ann_folder_path,
        whole_image_path=path_recomposed_image,
        margin=margin,
        format=format)
    au.unify_annotations()


parser = argparse.ArgumentParser(
    prog='Unifier program',
    description='The aim of this program is fetch all the tiles made by the cutter and recompose the whole image with '
                'the corresponding annotation.',
    epilog='Supervise.ly and Labelme format are supported by this program')

parser.add_argument('-src', '--source', dest='source', type=str, required=True,
                    help='path of the project containing images and labels')
parser.add_argument('-dst', '--destination', dest='destination', type=str, default="output_cutter",
                    help='path of the output project containing Supervise.ly annotation')
parser.add_argument('-mg', '--margin', dest='margin', type=int, required=True,
                    help='Overlapped part between tiles in pixel')
parser.add_argument('-prg', '--program', dest="program", type=int, required=True,
                    help='Annotation format used: \n'
                         '1. LabelMe\n'
                         '2. Supervise.ly'
                    )

if __name__ == "__main__":
    args = parser.parse_args()
    # Transform format argument parser integer to string
    format = FORMAT_DICT[args.program]

    # Create output folder if it does not exist
    if not os.path.exists(args.destination):
        os.makedirs(args.destination)

    # Retrieve datasets which are subdirectories of the input folder
    datasets = [f.path for f in os.scandir(args.source) if f.is_dir()]

    if format == "supervisely":
        supervisely_ann_filepath = os.path.join(args.destination, "ann")
        supervisely_img_filepath = os.path.join(args.destination, "img")
        # Create Supervise.ly image and annotation folders
        if not os.path.exists(supervisely_ann_filepath):
            os.makedirs(supervisely_ann_filepath)

        if not os.path.exists(supervisely_img_filepath):
            os.makedirs(supervisely_img_filepath)

        for ds_path in datasets:
            unify_tiles_and_annotations(input_cut_ann_folder_path=os.path.join(ds_path, "img"),
                                        input_cut_image_folder_path=os.path.join(ds_path, "ann"),
                                        output_folder=args.destination,
                                        margin=args.margin,
                                        format=format)

    elif format == "labelme":
        # if datasets exist, perform unification for all datasets
        if len(datasets):
            for ds_path in datasets:
                unify_tiles_and_annotations(input_cut_image_folder_path=ds_path,
                                            input_cut_ann_folder_path=args.source,
                                            output_folder=args.destination,
                                            margin=args.margin,
                                            format=format)
        # if not any dataset exists, the input folder transmitted as argument parser is the folder which represent only
        # one picture
        else:
            unify_tiles_and_annotations(input_cut_image_folder_path=args.source,
                                        input_cut_ann_folder_path=args.source,
                                        output_folder=args.destination,
                                        margin=args.margin,
                                        format=format)
