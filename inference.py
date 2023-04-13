import argparse
from distutils.command.config import config
import json
import os
import sys
from typing import Any, Dict, Iterable, NoReturn, Tuple, List
import cv2
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import yaml

sys.path.append(os.path.abspath(__file__ + "/.."))

from torch.utils.data import Dataset, DataLoader
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


class ImageDataset(Dataset):
    def __init__(self, data_dir, image_names_to_detect=None):
        self.data_dir = data_dir
        if image_names_to_detect is None:
            self.image_pathes = [file_path for file_path in Path(data_dir).iterdir()]
        else:
            self.image_pathes = [file_path for file_path in Path(data_dir).iterdir() if file_path.name in image_names_to_detect]

    def __len__(self):
        return len(self.image_pathes)

    def __getitem__(self, index):
        image_path = self.image_pathes[index]
        image = cv2.imread(str(image_path), cv2.COLOR_BGR2RGB)

        return image, str(image_path)


def collate_fn(batch):
    data = list(zip(*batch))
    images = [np.array(item) for item in data[0]]
    paths = data[1]
    return images, paths


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
        for det in pred:
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0s.shape).round()

                for *xyxy, conf, cls in reversed(det.cpu().numpy().tolist()):
                    
                    x1, y1, x2, y2 = xyxy
                    detections.append([
                        x1, 
                        y1, 
                        x2, 
                        y2, 
                        conf, 
                        cls
                    ])
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
    
    @torch.no_grad()
    def forward_batch(self, imgs: torch.Tensor) -> List[np.ndarray]:
        """
        returns: [[x, y, w, h, conf, cls], ...]
        """
        # preprocess
        im_list, im0s_list = [], []
        # im, im0s = self._preprocess(img)
        for img in imgs:
            im, im0s = self._preprocess(img)
            im_list.append(im)
            im0s_list.append(im0s)
        im_batch = torch.cat(im_list, dim=0)

        # Inference
        pred = self._model(im_batch, augment=self._augment, visualize=False)

        # NMS
        pred = non_max_suppression(pred, self._nms_conf_thres, self._iou_thres, self._classes, self._agnostic_nms,
                                   max_det=self._max_det)

        # Process predictions
        detections = []
        for i, p in enumerate(pred):
            detects = self._postprocess_detections([p], im_list[i], im0s_list[i])
            if len(detects) > 0:
                detections.append(detects)
            else:
                detections.append([])

        return detections


def run_inference(
    img_dir: str,
    config_path: str,
    weights: str,
    conf_threshold: float,
    input_size: Tuple[int, int],
    predictions_dir: str = None,
    img_names_to_detect: List[str] = None,
    visualizations_dir: str = None,
    class_list: List[str] = None,
    batch_size: int = 4,
    num_workers: int = 4
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

    assert isinstance(input_size, Iterable) and len(input_size) == 2, "Specify input_size in format Tuple[int, int]"

    config['weights'] = weights
    config['input_size'] = input_size
    config['nms_conf_thres'] = conf_threshold

    print('config', config)
    detector = Yolov5MultilabelDetector(config)
    
    img_dataset = ImageDataset(img_dir, image_names_to_detect=img_names_to_detect)
    dataloader = DataLoader(img_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    predictions = dict()
    for img_batch, img_pathes_batch in tqdm(dataloader):
        predictions_batch = detector.forward_batch(img_batch)
        
        for i, img_path in enumerate(img_pathes_batch):
            img_path = Path(img_path)
            img_base_name = img_path.stem
            predictions[img_base_name] = predictions_batch[i]
            
            if visualizations_dir is not None:
                img = img_batch[i]
                if len(predictions_batch[i]) > 0:
                    limage = detection_to_labeled_image(
                        detections=predictions_batch[i],
                        img_name=img_path.name,
                        width=img.shape[1],
                        height=img.shape[0],
                        detection_label_to_class_name=detection_id_to_label_mapping,
                        conf_threshold=conf_threshold
                    )
                    img = draw_bboxes_on_image(
                        labeled_image=limage,
                        img=img,
                    )
                cv2.imwrite(os.path.join(visualizations_dir, img_path.name), img)

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
    parser.add_argument("--input_size", nargs='+', type=int, help='inference size h,w')
    parser.add_argument("--weights", type=str)
    parser.add_argument("--config", type=str)
    parser.add_argument("--tr", type=float)
    parser.add_argument("--viz_dir", type=str, default=None)
    parser.add_argument("--classes", nargs='+', type=str, default=None, 
                        help='Class names. They must be in the same order as the model returns them, because their indexes will be used to map class index to class names in the visualization')    
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=10)

    args = parser.parse_args()

    run_inference(
        img_dir=args.img_dir,
        predictions_dir=args.predictions_dir,
        input_size=args.input_size, # inference size h,w
        weights=args.weights,
        conf_threshold=args.tr,
        config_path=args.config,
        visualizations_dir=args.viz_dir,
        class_list=args.classes,
        num_workers=args.num_workers,
        batch_size=args.batch_size
    )

