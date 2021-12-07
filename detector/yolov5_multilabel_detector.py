# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

from pathlib import Path
from typing import Any, Dict, NoReturn, Tuple

import numpy as np
import onnxruntime
import torch

from models.experimental import attempt_load
from utils.augmentations import letterbox
from utils.general import apply_classifier, check_requirements, check_suffix, load_yaml, non_max_suppression, \
    set_logging, scale_coords
from utils.metrics import predicts_to_multilabel_numpy
from utils.torch_utils import load_classifier, select_device


class Yolov5MultilabelDetector:
    def __init__(self, config_path: str):
        self._load_cfg(load_yaml(config_path))
        self._load_model()

    def _load_cfg(self, config: Dict[str, Any]) -> NoReturn:
        self._weights = config["weights"]
        self._device = select_device(config["device"])
        self._half = config["half"]  # use FP16 half-precision inference
        self._half &= self._device.type != "cpu"  # half precision only supported on CUDA
        self._input_size = config["input_size"]  # inference size h,w
        self._nms_conf_thres = config["nms_conf_thres"]  # confidence threshold
        self._iou_thres = config["iou_thres"]  # NMS IOU threshold
        self._iou_thres_post = config["iou_thres_post"]
        self._max_det = config["max_det"]  # maximum detections per image
        self._classes = config["classes"]  # filter by class: --class 0, or --class 0 2 3
        self._agnostic_nms = config["agnostic_nms"]  # class-agnostic NMS
        self._augment = config["augment"]  # augmented inference
        self._classify = config["classify"]  # False
        self._conf_thres_list = config[
            "conf_thres_list"]  # conf_thres_list = [0.5, 0.5, 0.2, 0.5, 0.5, 0.2, 0.5, 0.5, 0.2, 0.5]
        self._stride = config["stride"]

        suffixes = config["suffixes"]  # ['.pt', '.onnx']
        suffix = Path(self._weights).suffix.lower()
        check_suffix(self._weights, suffixes)  # check weights have acceptable suffix
        self._pt, self._onnx = (suffix == x for x in suffixes)  # backend booleans

        if not any([self._pt, self._onnx]):
            raise RuntimeError(f"Model weights with unsupported extension received: {self._weights}")

    def _load_model(self) -> NoReturn:
        if self._pt:
            self._model = attempt_load(self._weights, map_location=self._device)  # load FP32 model
            if self._half:
                self._model.half()  # to FP16
            if self._classify:  # second-stage classifier
                self._modelc = load_classifier(name="resnet50", n=2)  # initialize
                self._modelc.load_state_dict(torch.load("resnet50.pt", map_location=self._device)["model"]).to(
                    self._device).eval()
        elif self._onnx:
            check_requirements(("onnx", "onnxruntime"))
            self._session = onnxruntime.InferenceSession(self._weights, None)
        self._warmup()

    def _warmup(self) -> NoReturn:
        if self._pt and self._device.type != "cpu":
            self._model(torch.zeros(1, 3, *self._input_size).to(self._device).type_as(
                next(self._model.parameters())))  # run once

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Args:
            img: 3-dimentional image in BGR format
        """

        img = letterbox(img, self._input_size, stride=self._stride)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        if self._onnx:
            img = img.astype("float32")
        else:
            img = torch.from_numpy(img).to(self._device)
            img = img.half() if self._half else img.float()  # uint8 to fp16/32
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        return img

    @staticmethod
    def _postprocess_detections(pred, initial_img_shape: Tuple, input_img_shape: Tuple):
        coords = [bbox[:4] for bbox in pred]
        coords = scale_coords(input_img_shape[2:], np.stack(coords)[:4].astype(np.float64), initial_img_shape).round()
        for i, bbox_coords in enumerate(coords):
            pred[i][:4] = bbox_coords

    @torch.no_grad()
    def forward_image(self, img: np.ndarray) -> np.ndarray:
        # Inference
        preprocessed_img = self._preprocess(img)
        if self._pt:
            pred = self._model(preprocessed_img, augment=self._augment, visualize=False)[0]
        elif self._onnx:
            pred = torch.tensor(
                self._session.run([self._session.get_outputs()[0].name], {self._session.get_inputs()[0].name: img}))
        else:
            raise RuntimeError(f"Model weights with unsupported extention received: {self._weights}")

        # NMS
        pred = non_max_suppression(pred, self._nms_conf_thres, self._iou_thres, self._classes, self._agnostic_nms,
                                   max_det=self._max_det)[0]

        pred = pred.detach().cpu().numpy()
        if pred.shape[0] > 1:
            pred = predicts_to_multilabel_numpy(predicts=pred, iou_th=self._iou_thres_post,
                                                conf_th_list=self._conf_thres_list)
        else:
            pred = np.expand_dims(pred, 0)

        # Second-stage classifier (optional)
        if self._classify:
            pred = apply_classifier(pred, self._modelc, preprocessed_img, img)

        self._postprocess_detections(
            pred,
            initial_img_shape=img.shape,
            input_img_shape=preprocessed_img.detach().cpu().numpy().shape
        )

        return pred
