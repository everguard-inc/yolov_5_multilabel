# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
import argparse
import json
import os
import sys
from typing import Any, Dict, NoReturn, Tuple, List
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
        self._model.warmup(imgsz=(1, 3, *self._input_size), half=self._half)

    def _preprocess(self, img0: np.ndarray) -> np.ndarray:
        """
        Args:
            img: 3-dimentional image in BGR format
        """

        # Padded resize
        img = letterbox(img0, self._input_size, stride=self._stride, auto=self._auto_letterbox)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        im, im0s = img, img0

        im = torch.from_numpy(im).to(self._device)
        im = im.half() if self._half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        return im, im0s

    @staticmethod
    def _postprocess_detections(pred, im, im0s):
        detections = list()
        for i, det in enumerate(pred):
            result = list()
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0s.shape).round()

                for *xyxy, conf, cls in reversed(det.cpu().numpy().tolist()):
                    
                    x1, y1, x2, y2 = xyxy
                    result.append([
                        x1, 
                        y1, 
                        x2, 
                        y2, 
                        conf, 
                        cls
                    ]) 
            
            detections.append(result) 
        return detections

    @torch.no_grad()
    def forward_image(self, img: np.ndarray) -> List[np.ndarray]:
        """
        returns: [[x, y, w, h, conf, cls], ...]
        """
        # preprocess
        im, im0s = self._preprocess(img)

        # Inference
        pred = self._model(im, augment=self._augment, visualize=False)

        # NMS
        pred = non_max_suppression(pred, self._nms_conf_thres, self._iou_thres, self._classes, self._agnostic_nms,
                                   max_det=self._max_det)

        # Process predictions
        pred = self._postprocess_detections(pred, im, im0s)[0]

        return pred


def get_frames_with_bboxes_from_video(
    videos_dir,
    video_name,
    capturing_frequency,
    output_img_dir,
    output_predictions_dir,
    digits_num,
    detector,
    min_predictions_on_image
):  
    video_base_name, ext = os.path.splitext(video_name)
    video_path = os.path.join(videos_dir, video_name)

    print(f"Getting frames from video {video_path}")
    capture = cv2.VideoCapture(cv2.samples.findFileOrKeep(video_path))

    if not capture.isOpened:
        print(f"Unable to open: {video_path}")
        return

    counter = 0
    while True:

        counter += 1
        
        ret, frame = capture.read()
        
        if not ret:
            break

        if counter % capturing_frequency == 0:
            img_base_name = f"{video_base_name}_frame_{str.zfill(str(counter), digits_num)}"
            img_path = os.path.join(output_img_dir, f"{img_base_name}.jpg")
            prediction_path = os.path.join(output_predictions_dir, f"{img_base_name}.json")

            predictions = detector.forward_image(frame)
            
            if len(predictions) > min_predictions_on_image:
                # save ann
                with open(prediction_path, 'w') as json_file:
                    json.dump(predictions, json_file)
                    
                # save img
                cv2.imwrite(img_path, frame)

def detect_images_and_save_predictions(
    img_dir,
    predictions_dir,
    detector,
    min_predictions_on_image = 0,
):
    os.makedirs(predictions_dir, exist_ok=True)

    for img_name in tqdm(os.listdir(img_dir)):
        img_path = os.path.join(img_dir, img_name)
        img_base_name = os.path.splitext(img_name)[0]
        prediction_path = os.path.join(predictions_dir, f"{img_base_name}.json")

        predictions = detector.forward_image(cv2.imread(img_path))

        if len(predictions) >= min_predictions_on_image:
            # save ann
            with open(prediction_path, 'w') as json_file:
                json.dump(predictions, json_file)

def run_inference(
    img_dir,
    predictions_dir,
    input_size = None, # inference size h,w
    weights = None,
    conf_threshold = None,
):
    os.makedirs(predictions_dir, exist_ok=True)
    
    config=load_yaml('config.yaml')
    
    if weights is not None:
        config['weights'] = weights
    if input_size is not None:
        config['input_size'] = input_size
    if conf_threshold is not None:
        config['nms_conf_thres'] = conf_threshold
        
    detector = Yolov5MultilabelDetector()
    
    for img_name in os.lisrdir(img_dir):
        
        img_base_name = os.path.splitext(img_name)[0]
        img_path = os.path.join(img_dir, img_name)
        prediction_path = os.path.join(predictions_dir, f'{img_base_name}.json')
        
        img = cv2.imread(img_path)
        prediction = detector.forward_image(img)
        
        with open(prediction_path, 'w') as json_file:
            json.dump(prediction, json_file)
                
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--predictions_dir", type=str, required=True)
    parser.add_argument("--input_size", nargs='+', type=int, help='inference size h,w')
    parser.add_argument("--weights", type=str)
    parser.add_argument("--tr", type=float)
    args = parser.parse_args()
    
    run_inference(
        img_dir=args.img_dir,
        predictions_dir=args.predictions,
        input_size=args.input_size, # inference size h,w
        weights=args.weights,
        conf_threshold=args.tr,
    )