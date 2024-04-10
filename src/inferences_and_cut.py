import os
from typing import Final

import cv2
import imutils.paths
from tqdm import tqdm

from src.utils import diverse
from src.annotation.labelme import LabelMeAnnotation
from src.annotation.supervisely import SuperviselyAnnotation
from src.detection.detector import HybridDetector, IOU, RADIUS, CENTER_X, CENTER_Y
from src.utils.tile_cutter import TileCutter

OUTPUT_FORMAT: Final = "labelme"
OUTPUT_FOLDER: Final = r"C:\Users\tristan_cotte\PycharmProjects\MassiveAnnotationSplit\labelme_dataset"

MODEL_PATH: Final = r"C:\Users\tristan_cotte\PycharmProjects\Tools-PlatformImage\models\AMES\ames_yolov8.pt"
SOURCE_PATH: Final = r"C:\Users\tristan_cotte\PycharmProjects\MassiveAnnotationSplit\supervisely_dataset\ds0"
CLS: Final = "a"
NB_CUTS: Final = 3
MARGIN: Final = 20


def detect_and_cut(detector, image_path, nb_cuts, margin):
    image_bgr = cv2.imread(image_path)

    predicted_rectangles = detector.detect(image_path)

    # Create dataset_labelme_cut folder to welcome the tiled pictures
    _, image_name = os.path.split(image_path)
    save_dataset_tile_folder = os.path.join(OUTPUT_FOLDER, image_name[:-4])

    # directory cleaner
    if not os.path.exists(save_dataset_tile_folder):
        os.makedirs(save_dataset_tile_folder, exist_ok=False)
    else:
        diverse.clean_directory(save_dataset_tile_folder)

    # Cut picture in some rectangles
    tile_cutter = TileCutter(nb_cuts=nb_cuts, marge=margin, image_shape=image_bgr.shape[:2])
    tiles = tile_cutter.tiles
    decimal_number = len(str(len(tiles)))

    supervisely_ann_filepath = os.path.join(save_dataset_tile_folder, "ann")
    supervisely_img_filepath = os.path.join(save_dataset_tile_folder, "img")

    current_width = 0
    current_height = 0

    for idx, tile in tqdm(enumerate(tiles)):
        tile0 = image_bgr[tile[1]: tile[3], tile[0]:tile[2]]
        left_corner = [tile[0], tile[1]]

        img_height, img_width = tile0.shape[:2]

        filtered_indexes = diverse.filter_predictions_by_tile(predicted_rectangles, tile)
        recs_tile0 = [predicted_rectangles[i] for i in filtered_indexes]

        tile_img_name = f"{image_name[:-4]}_{idx + 1:0{decimal_number}d}_over_f{str(len(tiles))}.jpg"
        tile_img_path = os.path.join(save_dataset_tile_folder, tile_img_name)

        translated_recs = [diverse.rectangle_translation(
            rectangle=i, translation_y=-current_height, translation_x=-current_width) for i in recs_tile0]

        if OUTPUT_FORMAT == "labelme":
            labelme_ann = LabelMeAnnotation(path_related_image=tile_img_path,
                                            img_size=[img_width, img_height])

            labelme_ann.add_rectangles(rectangles=translated_recs, cls=[CLS] * len(translated_recs))
            labelme_ann.write_ann_file(output_path=tile_img_path[:-4] + '.json')

            # Save picture
            cv2.imwrite(os.path.join(save_dataset_tile_folder, tile_img_name), tile0)

        elif OUTPUT_FORMAT == "supervisely":
            if not os.path.exists(supervisely_ann_filepath):
                os.makedirs(supervisely_ann_filepath)
            if not os.path.exists(supervisely_img_filepath):
                os.makedirs(supervisely_img_filepath)

            sly_ann = SuperviselyAnnotation(img_size=[img_width, img_height])
            sly_ann.add_rectangles(rectangles=translated_recs, cls=[CLS] * len(translated_recs))
            sly_ann.write_ann_file(output_path=os.path.join(supervisely_ann_filepath, tile_img_path + '.json'))

            # Save picture
            cv2.imwrite(os.path.join(supervisely_img_filepath, tile_img_name), tile0)

        # Rectangle translation update
        if (idx + 1) % tile_cutter.nb_cuts == 0:
            current_width = 0
            current_height += img_height - 2 * tile_cutter.marge
        else:
            current_width += img_width - 2 * tile_cutter.marge


if __name__ == "__main__":

    # Get annotation using AI YOLO model
    # model = YOLO(model_path)
    # results = model.predict(source=image_bgr, conf=0.3, iou=0.2, max_det=20000, imgsz=3120)
    # predicted_rectangles = results[0].boxes.xyxy.cpu().numpy().astype(int)

    detector = HybridDetector(
        model_path=r"C:\Users\tristan_cotte\PycharmProjects\Tools-PlatformImage\models\AMES\ames_yolov8.pt",
        iou=IOU, radius=RADIUS, center_x=CENTER_X, center_y=CENTER_Y)

    if os.path.isdir(SOURCE_PATH):
        for image_path in list(imutils.paths.list_images(SOURCE_PATH)):
            detect_and_cut(image_path=image_path, detector=detector, nb_cuts=NB_CUTS, margin=MARGIN)

    else:
        detect_and_cut(image_path=SOURCE_PATH, detector=detector, nb_cuts=NB_CUTS, margin=MARGIN)
