# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
import argparse
from distutils.command.config import config
import json
import os
import sys
from typing import Any, Dict, Iterable, NoReturn, Tuple, List
import cv2

import numpy as np
import torch
from tqdm import tqdm
import yaml

sys.path.append(os.path.abspath(__file__ + "/.."))

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device
from utils.augmentations import letterbox

from eg_data_tools.annotation_processing.converters.convert_from_detections_to_labeled_image import detection_to_labeled_image
from eg_data_tools.visualization.media.image import draw_bboxes_on_image

def load_yaml(path: str):
    with open(path, "r") as stream:
        content = yaml.load(stream, Loader=yaml.FullLoader)
    return content


class Yolov5MultilabelDetector:
    def __init__(self, config: Dict):
        self._load_cfg(config)
        self._load_model()

    def _load_cfg(self, config: Dict[str, Any]) -> NoReturn:
        self._weights = config["weights"]
        self._device = select_device(config["device"])
        self._half = config["half"]  # use FP16 half-precision inference
        self._half &= self._device.type != "cpu"  # half precision only supported on CUDA
        self._input_size = config["input_size"]  # inference size h, w
        self._nms_conf_thres = config["nms_conf_thres"]  # confidence threshold
        self._iou_thres = config["iou_thres"]  # NMS IOU threshold
        self._iou_thres_post = config["iou_thres_post"]
        self._max_det = config["max_det"]  # maximum detections per image
        self._classes = config["classes"]  # filter by class: --class 0, or --class 0 2 3
        self._agnostic_nms = config["agnostic_nms"]  # class-agnostic NMS
        self._augment = config["augment"]  # augmented inference
        self._classify = config["classify"]  # False
        self._stride = config["stride"]
        self._auto_letterbox = config["auto_letterbox"]
        self._pad = config['pad']

    def _load_model(self) -> NoReturn:
        # Load model
        self._model = DetectMultiBackend(self._weights, device=self._device)

        pt, jit, onnx, engine = self._model.pt, self._model.jit, self._model.onnx, self._model.engine
        
        # Half
        self._half &= (pt or jit or onnx or engine) and self._device.type != 'cpu'  # FP16 supported on limited backends with CUDA
        if pt or jit:
            self._model.model.half() if self._half else self._model.model.float()

        self._warmup()

    def _warmup(self) -> NoReturn:
        self._model.warmup(imgsz=(1, 3, self._input_size, self._input_size), half=self._half)

    def _preprocess(self, img0: np.ndarray) -> torch.Tensor:
        """
        Preprocesses an image for input into a YOLOv5 object detection model.
        
        Args:
            img0 (np.ndarray): Original 3-dimentional image in BGR format  to be processed. 
        
        Returns:
            im: Preprocessed image as a PyTorch tensor, ready for input into the YOLOv5 model. 
            im0s: Original image, unchanged and stored for display purposes.
        
        Steps:
            1. Resize and pad the input image using letterboxing.
            2. Convert the image from HWC (height, width, channels) format to CHW (channels, height, width) format and from BGR to RGB.
            3. Convert the image to a contiguous array.
            4. Convert the array to a PyTorch tensor and move it to the device specified during initialization.
            5. Convert the tensor to float or half precision, depending on the `_half` attribute.
            6. Scale the pixel values from 0-255 to 0.0-1.0 range.
            7. If the tensor does not have a batch dimension, add one.
            8. Return the preprocessed tensor and the original, unprocessed image for display purposes.
        """

        # Resize and pad
        img = None
        shape0 = img0.shape[:2]
        shape = np.array(shape0, dtype=float) / max(shape0)
        shape = np.ceil(shape * self._input_size / self._stride + self._pad).astype(np.int32) * self._stride
        r = self._input_size / max(shape0)  # ratio
        if r != 1:  # if sizes are not equal
            h0, w0 = shape0
            img = cv2.resize(img0, (int(w0 * r), int(h0 * r)),
                             interpolation=cv2.INTER_LINEAR)
        
        img, ratio, pad = letterbox(img if img is not None else img0, shape, auto=self._auto_letterbox, scaleup=False)

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        im, im0s = img, img0

        im = torch.from_numpy(im).to(self._device)
        im = im.half() if self._half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        return im, im0s, pad
    
    @staticmethod
    def __get_params_for_posprocess(im, im0s, pad):
        shape = im.shape[2:]
        h, w = shape
        w -= 2*pad[0]
        h -= 2*pad[1]        
        base_shape = im0s.shape[:2]
        h0, w0 = base_shape
        params = ((h / h0, w / w0), pad)
        return shape, base_shape, params

    @staticmethod
    def _postprocess_detections(pred, im, im0s, pad):
        detections = list()
        shape, base_shape, params = Yolov5MultilabelDetector.__get_params_for_posprocess(im, im0s, pad)

        for _, det in enumerate(pred):
            result = list()
            if len(det):
                # Rescale boxes from img_size to im0 size
                # det format: np.array[[x1, y1, x2, y2, conf, cls], ...]
                det[:, :4] = scale_coords(shape, det[:, :4], base_shape, params).round()
                result = det.cpu().numpy().tolist()
            detections.append(result) 
        return detections[0]

    @torch.no_grad()
    def forward_image(self, img: np.ndarray) -> List[np.ndarray]:
        """
        Args:
            img (np.ndarray): image in BGR format

        returns: [[x, y, w, h, conf, cls], ...]
        """
        # preprocess
        im, im0s, pad = self._preprocess(img)
        # Inference
        # Here `im` is a torch.Tensor with shape [batch_size, channels, input_size, input_size]. Colorspace is RGB
        pred = self._model(im, augment=self._augment, visualize=False)

        # NMS
        pred = non_max_suppression(pred, self._nms_conf_thres, self._iou_thres, self._classes, self._agnostic_nms,
                                   max_det=self._max_det)

        # Process predictions
        pred = self._postprocess_detections(pred, im, im0s, pad)

        return pred



def run_inference(
    img_dir: str,
    config_path: str,
    weights: str,
    conf_threshold: float,
    input_size: int,
    predictions_dir: str = None,
    img_names_to_detect: List[str] = None,
    visualizations_dir: str = None,
    class_list: List[str] = None,
) -> Dict[str, List[Dict]]:
    """
    Runs object detection on images and saves visualizations and/or predictions.

    Args:
        img_dir (str): Path to the directory containing the images to be processed.
        config_path (str): Path to the yaml file containing the configuration.
        weights (str): Path to the weights file.
        conf_threshold (float): Detection confidence threshold.
        input_size (Tuple[int, int]): Input size of the model.
        predictions_dir (str, optional): Path to the directory where predictions will be saved. If None, predictions not be saved. Default is None. 
        img_names_to_detect (List[str], optional): List of image names to detect. If None, all images in img_dir will be processed. Default is None.
        visualizations_dir (str, optional): Path to the directory where visualizations will be saved. Default is None.
        class_list (List[str], optional): List of class names. Used for visualization. Default is None.

    Returns:
        Dict[str, List[Dict]]: A dictionary of the form {img_name: detection}. Detection is a list of dictionaries, each representing a detection.
    """
    
    detection_id_to_label_mapping = None
    if visualizations_dir is not None:
        os.makedirs(visualizations_dir, exist_ok=True)
        detection_id_to_label_mapping = {i: cls_name for i, cls_name in enumerate(class_list)}        
        
    if class_list is None:
        print("class_lsit is not specified. There will be a class indicies instead of class names on the visualizations")

    config=load_yaml(config_path)

    assert isinstance(input_size, int) and input_size > 0, "Specify input_size in format int"

    config['weights'] = weights
    config['input_size'] = input_size
    config['nms_conf_thres'] = conf_threshold

    print('config', config)
    detector = Yolov5MultilabelDetector(config)

    if img_names_to_detect is None:
        img_names_to_detect = os.listdir(img_dir)

    predictions = dict()
    for img_name in tqdm(img_names_to_detect, desc="Predicting"):
        
        img_base_name = os.path.splitext(img_name)[0]
        img_path = os.path.join(img_dir, img_name)
        try:
            img = cv2.imread(img_path)
        except Exception as e:
            print('img_name', img_name, file = sys.stderr)
            raise(e)
        prediction = detector.forward_image(img)
        predictions[img_base_name] = prediction
        
        if visualizations_dir is not None:
            if len(prediction) > 0:
                limage = detection_to_labeled_image(
                    detections=prediction,
                    img_name=img_name,
                    width=img.shape[1],
                    height=img.shape[0],
                    detection_label_to_class_name=detection_id_to_label_mapping,
                    conf_threshold=conf_threshold,
                )
                img = draw_bboxes_on_image(
                    labeled_image=limage,
                    img=img,
                )
            cv2.imwrite(os.path.join(visualizations_dir, img_name), img)

    if predictions_dir is not None:
        os.makedirs(predictions_dir, exist_ok=True)
        for img_base_name, prediction in predictions.items():
            prediction_path = os.path.join(predictions_dir, f'{img_base_name}.json')
            with open(prediction_path, 'w') as json_file:
                json.dump(prediction, json_file)
    
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--predictions_dir", type=str, required=True)
    parser.add_argument("--input_size", type=int, help='inference size max(h,w)')
    parser.add_argument("--weights", type=str)
    parser.add_argument("--config", type=str)
    parser.add_argument("--tr", type=float)
    parser.add_argument("--viz_dir", type=str, default=None)
    parser.add_argument("--classes", nargs='+', type=str, default=None, 
                        help='Class names. They must be in the same order as the model returns them, because their indexes will be used to map class index to class names in the visualization')    

    args = parser.parse_args()

    run_inference(
        img_dir=args.img_dir,
        predictions_dir=args.predictions_dir,
        input_size=args.input_size, # inference size max(h,w)
        weights=args.weights,
        conf_threshold=args.tr,
        config_path=args.config,
        visualizations_dir=args.viz_dir,
        class_list=args.classes
    )

