import argparse
import os
from typing import Final

import cv2
import imutils
from imutils import paths
from tqdm import tqdm

from src.utils import diverse
from src.annotation.labelme import LabelMeAnnotation
from src.annotation.supervisely import SuperviselyAnnotation, get_slylabelfile_from_imgfile
from src.utils.tile_cutter import TileCutter
from src.utils.diverse import clean_directory

FORMAT_DICT: Final = {1: 'labelme', 2: 'supervisely'}

parser = argparse.ArgumentParser(
    prog='Cutter program',
    description='The aim of this program is to cut an image and its annotation to annotated tiles',
    epilog='Supervise.ly and Labelme format are supported by this program')

parser.add_argument('-src', '--source', dest='source', type=str, required=True,
                    help='path of the project containing images and labels')
parser.add_argument('-dst', '--destination', dest='destination', type=str, default="output_cutter",
                    help='path of the output project containing Supervise.ly annotation')
parser.add_argument('-cut', '--nb_cuts', dest='nb_cuts', type=int, required=True,
                    help='Square root of the tiles number to cut')
parser.add_argument('-mg', '--margin', dest='margin', type=int, required=True,
                    help='Overlapped part between tiles in pixel')
parser.add_argument('-prg', '--program', dest="program", type=int, required=True,
                    help='Annotation format used: \n'
                         '1. LabelMe\n'
                         '2. Supervise.ly'
                    )

if __name__ == "__main__":
    args = parser.parse_args()

    # Transform format from argument parser integer to string
    format = FORMAT_DICT[args.program]

    for image_path in tqdm(list(imutils.paths.list_images(args.source))):
        _, image_name = os.path.split(image_path)
        image_bgr = cv2.imread(image_path)

        # Create dataset_labelme_cut folder to welcome the tiled pictures
        save_dataset_tile_folder = os.path.join(args.destination, image_name[:-4])
        if not os.path.isdir(save_dataset_tile_folder):
            os.makedirs(save_dataset_tile_folder)
        else:
            clean_directory(directory_path=save_dataset_tile_folder)

        # Cut picture in several tiles
        tile_cutter = TileCutter(nb_cuts=args.nb_cuts, marge=args.margin, image_shape=image_bgr.shape[:2])
        tiles = tile_cutter.tiles
        # Get number of decimal to register pictures -> it is used to sort easily the pictures depending on their
        # filename
        decimal_number = len(str(len(tiles)))

        labels = []
        if format == "supervisely":
            # Get Supervise.ly annotation and image path
            supervisely_ann_filepath = os.path.join(save_dataset_tile_folder, "ann")
            supervisely_img_filepath = os.path.join(save_dataset_tile_folder, "img")
            # Get annotation labels from annotation file
            labels = SuperviselyAnnotation.from_file(
                path_annotation_file=get_slylabelfile_from_imgfile(image_path)).labels

        elif format == "labelme":
            # Get annotation labels from annotation file
            labels = LabelMeAnnotation.from_file(path_annotation_file=image_path[:-4] + ".json").labels

        else:
            raise Exception("This format is not supported by this program. Enter either 1 or 2 as program number.")

        # Flatten rectangle -> transform them in xyxy format -> in array like [x1, y1, x2, y2]
        ann_rectangles = [label.rectangle.flatten_format() for label in labels]

        # Variables used to do rectangles translations
        current_width = 0
        current_height = 0

        for idx, tile in tqdm(enumerate(tiles)):
            tile0 = image_bgr[tile[1]: tile[3], tile[0]:tile[2]]
            left_corner = [tile[0], tile[1]]
            img_height, img_width = tile0.shape[:2]

            # Fetch only annotations lying in the tile
            filtered_indexes = diverse.filter_predictions_by_tile(ann_rectangles, tile)
            filtered_labels = [labels[i] for i in filtered_indexes]
            recs_tile0 = [ann_rectangles[i] for i in filtered_indexes]
            cls_tiles0 = [label.cls for label in filtered_labels]

            # Tile filename
            tile_img_name = f"{image_name[:-4]}_{idx + 1:0{decimal_number}d}_over_f{str(len(tiles))}.jpg"

            # Reframe rectangles depending on the position of the current tile
            translated_recs = [diverse.rectangle_translation(
                rectangle=i, translation_y=-current_height, translation_x=-current_width) for i in recs_tile0]

            if format == "labelme":
                tile_img_path = os.path.join(save_dataset_tile_folder, tile_img_name)
                labelme_ann = LabelMeAnnotation(path_related_image=tile_img_path,
                                                img_size=[img_width, img_height])
                labelme_ann.add_rectangles(rectangles=translated_recs, cls=cls_tiles0)
                # Write Labelme annotation file
                labelme_ann.write_ann_file(output_path=tile_img_path[:-4] + '.json')

                # Save picture
                cv2.imwrite(os.path.join(save_dataset_tile_folder, tile_img_name), tile0)

            elif format == "supervisely":
                if not os.path.exists(supervisely_ann_filepath):
                    os.makedirs(supervisely_ann_filepath)
                if not os.path.exists(supervisely_img_filepath):
                    os.makedirs(supervisely_img_filepath)

                sly_ann = SuperviselyAnnotation(img_size=[img_width, img_height])
                sly_ann.add_rectangles(rectangles=translated_recs, cls=cls_tiles0)
                # Write Supervise.ly annotation file
                sly_ann.write_ann_file(output_path=os.path.join(supervisely_ann_filepath, tile_img_name + '.json'))

                # Save picture
                cv2.imwrite(os.path.join(supervisely_img_filepath, tile_img_name), tile0)

            # Rectangle translation update
            if (idx + 1) % tile_cutter.nb_cuts == 0:
                current_width = 0
                current_height += img_height - 2 * tile_cutter.marge
            else:
                current_width += img_width - 2 * tile_cutter.marge
