from typing import List

import cv2
import numpy as np
import torch
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO

from src.utils import diverse

IOU = 0.3
CONFIDENCE = 0.3

BX = 16
BY = 16
width = 3070
RADIUS = int(width / 2)
CENTER_X = BX + RADIUS
CENTER_Y = BY + RADIUS


def filtered_smaller_bboxes(pred, threshold_area=200) -> List:
    """
    This function filters bounding boxes from SAHI in function of their area. If the area of one box exceeds a threshold value, this box will be remove.
    Otherwise, this box will be transform in x0 y0 x1 y1 format.
    :pred: bounding boxes predicted by SAHI algorithm
    :threshold area: max area for bounding box to be kept
    :return: list of bounding boxes in x0 y0 x1 y1 format
    """
    smaller_bboxes = []

    for prediction in pred.object_prediction_list:
        area = prediction.bbox.area
        if area < threshold_area:
            smaller_bboxes.append(np.array(prediction.bbox.to_xyxy(), dtype=int))
    return smaller_bboxes


def filtered_bboxes_with_circle(bboxes: np.array, center_x: int, center_y: int, radius: int) -> np.array:
    """
    Filter which allows to only keep the bounding boxes which are lying in the Plexiglas hole.
    :bboxes: 2D array which gathers all the bounding boxes in x0 y0 x1 y1 format.
    :center_x: center abscissa of the Plexiglas hole
    :center_y: center ordinate of the Plexiglas hole
    :radius: radius of the Plexiglas hole
    :return: filtered bounnding boxes in same format as input
    """
    filtered_circle_bboxes = []

    for xyxy in bboxes:
        xlfc, ylfc = xyxy[:2]
        #  (x - center_x)² + (y - center_y)² < radius²
        if (xlfc - center_x) ** 2 + (ylfc - center_y) ** 2 < radius ** 2:
            filtered_circle_bboxes.append(xyxy)

    return filtered_circle_bboxes

class Yolov8Detector:
    def __init__(self, device: str, model_path: str, iou: float, confidence: float, imgsz: int):
        """
        The aim of this class is to detect objects thanks to YOLOv8 API.
        :param device: device used to predict (could be 'cpu' or 'cuda')
        :param model_path: path of the custom YOLO model (the file extension has to be '.pt')
        """
        super().__init__()
        self.model_path = model_path
        self.device = device
        self.model = self.load_model()
        self.yolo_version = 8
        self.confidence = confidence
        self.iou = iou
        self.imgsz = imgsz

    def load_model(self) -> YOLO:
        """
        Load YOLOv8 model in the device set at initialization
        :return: loaded model
        """
        model = YOLO(self.model_path)
        model.to(self.device)
        return model

    def predict_one_class_only(self, image: np.array, max_det: int = 7000) \
            -> np.ndarray:
        """
        Do predictions with YOLOv8 API. Keep only the first class predicted, transform boxes in array of integer
        and send it to CPU.
        :param image: image in which inferences will be done
        :param max_det: maximum items in output
        :return: 2D array of boxes coordinates
        """
        results = self.model.predict(source=image, conf=self.confidence, iou=self.iou, max_det=max_det)
        return results[0].boxes.xyxy.cpu().numpy().astype(int)


class HybridDetector:
    def __init__(self, model_path, **kwargs):
        """
        Hybrid solution consists in using firstly the YOLOv8 detector to detect the colonies which presents medium
        area and then using the SAHI with YOLOv8 detector to detect the smaller colonies. Once we get the bounding
        boxes from the two detectors, we will gather the results of the detectors. It will be important to avoid the
        duplicates, so we certainly use Non-Maximum Suppression algorithm to avoid this fact.
        :param model_path: YOLO model path
        """
        self.iou = kwargs.get('iou', 0.3)
        self.confidence_yolo = kwargs.get('confidence_yolo', 0.3)
        self.confidence_sahi = kwargs.get('confidence_sahi', 0.3)

        self.center_x = kwargs.get('center_x', 0)
        self.center_y = kwargs.get('center_y', 0)
        self.radius = kwargs.get('center_x', 3000)
        self.image_size = kwargs.get('imgsz', 3000)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.yolo_detector = Yolov8Detector(device=self.device, model_path=model_path, iou=self.iou,
                                            confidence=self.confidence_yolo,
                                  imgsz=self.image_size)

        # SAHI model
        self.sahi_detector = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=model_path,
            confidence_threshold=self.confidence_sahi,
            device=self.device  # if torch.cuda.is_available() else 'cpu'
        )

    def detect(self, image_path: str) -> np.ndarray:
        """
        We are going to enumerate all the steps which have to be done to create this *hybrid algorithm*:
        1. Execute the YOLOv8 algorithm with the parameters currently used in the software in production.
        2. Execute SAHI algorithm embedding YOLOv8 model executed previously. The optimal parameters have to be found ->
        for the YOLOv8 and the sliding window.
        3. (Falcutative) Set a differnent ID ofr the bounding boxes found on the first and the second
        inferences. To do it, we can add a new column of the bounding box annotation. For example, id 0 for the first
        prediction and id 1 for the second. This step seems to be facultative because it is useful only for the
        visualisation (the color of the bbox could be different when we will plot the picture and its annotations).
        4. Gather the annotations resulted from the step 1 and the step 2.
        5. Remove all annotation which are not lying within the define circle. This circle represents the hole made in
        the plexiglas and was computed thanks to ImageJ software.
        :param image_path: path of the image in which we will use the hybrid algorithm
        :return: array of bounding boxes
        """
        image_bgr = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        rectangles_yolo = self.yolo_detector.predict_one_class_only(image=image_bgr, max_det=20000)


        sahi_pred = get_sliced_prediction(image_rgb, self.sahi_detector, slice_height=500, slice_width=500, overlap_height_ratio=0.1, overlap_width_ratio=0.1)

        smaller_bboxes = filtered_smaller_bboxes(sahi_pred, threshold_area=200)

        smaller_bboxes_id = np.hstack((smaller_bboxes, np.zeros((len(smaller_bboxes), 1), dtype=int)))
        rectangles_id = np.hstack((rectangles_yolo, np.ones((len(rectangles_yolo), 1), dtype=int)))

        gathered_bboxes = np.concatenate((smaller_bboxes_id, rectangles_id), axis=0)
        gathered_bboxes.astype(int)

        nms_gathered_bboxes, _ = diverse.nms_python(gathered_bboxes, np.array([0.5] * len(gathered_bboxes)), 0.3)
        filtered_circle_bboxes = filtered_bboxes_with_circle(bboxes=nms_gathered_bboxes, center_x=self.center_x,
                                                             center_y=self.center_y, radius=self.radius)
        return filtered_circle_bboxes

if __name__ == '__main__':


    HybridDetector(model_path=r"C:\Users\tristan_cotte\PycharmProjects\Tools-PlatformImage\models\AMES\ames_yolov8.pt",
                   iou=IOU,
                   radius=RADIUS,
                   center_x=CENTER_X,
                   center_y=CENTER_Y)