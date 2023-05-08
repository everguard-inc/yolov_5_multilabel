import os
import cv2
import yaml
import json
import torch
import argparse
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Dict, Any, List, Tuple, NoReturn
from collections import defaultdict
from datetime import time, datetime, timedelta
from collections import defaultdict
from tqdm import tqdm
from tabulate import tabulate


from utils.sort import Sort
from utils.datasets import LoadImages
from utils.torch_utils import select_device
from utils.general import non_max_suppression, scale_coords
from models.common import DetectMultiBackend


from eg_data_tools.data_units.data_units import LabeledImage, BBox
from eg_data_tools.visualization.media.image import draw_bboxes_on_image


def load_yaml(path: str):
    with open(path, "r") as stream:
        content = yaml.load(stream, Loader=yaml.FullLoader)
    return content

def open_csv(names_file_path):
    with open(names_file_path, "r") as text_file:
        rows = text_file.readlines()
    for i, line in enumerate(rows):
        rows[i] = line.replace("\n", "").split(",")
    return rows

def save_json(
    value,
    file_path,
):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as filename:
        json.dump(value, filename, indent=4)


def setup_config(
    config_path: str,
    weights_path: str,
    input_size: Tuple[int, int],
    device_id: int,
    agnostic_nms: bool = True,
    iou_thres: float = 0.3,
    conf_thres: float = 0.5,
) -> Dict[str, Any]:
    
    config = load_yaml(config_path)
    config['input_size'] = input_size
    config['device'] = device_id
    config['weights'] = weights_path
    config['iou_thres'] = iou_thres
    config['nms_conf_thres'] = conf_thres
    config['agnostic_nms'] = agnostic_nms
    
    return config

def preprocess_events(
    events_list_path: str
) -> Dict[str, List[Tuple[time, time]]]:
    """
        Args:
            events_list_path: path to events list csv file with columns
                video_name, start_time, end_time
    """
    raw_events = open_csv(events_list_path)
    events = defaultdict(list)
    
    for video_name, start_time, end_time in raw_events[1:]:
        events[video_name].append(
            (datetime.strptime(start_time, '%H:%M:%S.%f').time(), datetime.strptime(end_time, '%H:%M:%S.%f').time())
        )
    
    return events

def milliseconds_to_time(ms: float) -> time:
    time_delta = timedelta(milliseconds=ms)
    return (datetime.min + time_delta).time()


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
    def forward_image(self, im: np.ndarray, im0s: np.ndarray) -> List[np.ndarray]:
        """
        returns: [[x, y, w, h, conf, cls], ...]
        """
        im = torch.from_numpy(im).to(self._device)
        im = im.half() if self._half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]
        
        # Inference
        pred = self._model(im, augment=self._augment, visualize=False)
        # NMS
        pred = non_max_suppression(pred, self._nms_conf_thres, self._iou_thres, self._classes, self._agnostic_nms,
                                   max_det=self._max_det)
        # Process predictions
        pred = self._postprocess_detections(pred, im, im0s)
        return pred



def run_evaluation(
    model_path: str,
    events_list_path: str,
    input_size: Tuple[int, int],
    class_names: List[str],
    report_path: str,
    video_path: str,
    viz_dir_path: str = None,
    model_config_path: str = None,
    tracker_config_path: str = None,
    device_id: int = 0,
    iou_thres: float = 0.3,
    conf_thres: float = 0.5,
    agnostic_nms: bool = True,
    target_class: str = 'not_in_hardhat',
    false_actions_percent_threshold: float = 0.5,
) -> NoReturn:  
    """
        Args:
            model_path: path to model weights
            events_list_path: path to csv file with labelled events
            input_size: input size for model
            class_names: list of class names
            report_path: path to save report
            video_path: path to video file or directory with videos
            model_config_path: path to model .yaml config
            device_id: id of GPU to use
            iou_thres: iou threshold for NMS
            conf_thres: confidence threshold for NMS
            agnostic_nms: if NMS is agnostic or not. For better tracking results set it to True
            target_class: class to evaluate during event
            false_actions_percent_threshold: threshold for false actions percentage, the higher the threshold, the more false actions are allowed
            video_dir: path to directory with videos 
    """
    print("Loading model...")
    
    if model_config_path is None:
        model_config_path = Path(__file__).parent / "config.yaml"
    
    if tracker_config_path is None:
        tracker_config_path = Path(__file__).parent / "tracker_config.yaml"
        
    events = preprocess_events(events_list_path)
    
    model_config = setup_config(
        config_path=model_config_path,
        weights_path=model_path,
        device_id=device_id,
        iou_thres=iou_thres,
        conf_thres=conf_thres,
        input_size=input_size,
        agnostic_nms=agnostic_nms,
    )
    
    tracker_config = load_yaml(tracker_config_path)
    
    classes_mapping = {i: class_name for i, class_name in enumerate(class_names)}
    
    if Path(video_path).is_file():
        video_path = [Path(video_path)]
    elif Path(video_path).is_dir():
        video_path = [pth for pth in Path(video_path).iterdir() if pth.is_file()]

    events_report = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    
    assert len(video_path) == len(events), "Number of videos and events should be equal"
    
    model = Yolov5MultilabelDetector(config=model_config)
    tracker = Sort(config=tracker_config)
    out_video_writer = None
    

    for video_pth in video_path:
        print(f"Processing video {video_pth.name}")
        
        dataset = LoadImages(path=video_pth, img_size=model_config["input_size"], stride=model_config["stride"], auto=model_config["auto_letterbox"])
        
        current_event_timesteps = events[video_pth.name]
        if viz_dir_path:
            os.makedirs(viz_dir_path, exist_ok=True)
            codec = cv2.VideoWriter_fourcc(*"XVID")
            out_video_writer = cv2.VideoWriter(
            os.path.join(viz_dir_path, video_pth.name), codec, 30, (1280, 720), isColor=True,
        )
        video_base_name = video_pth.name
        
        for path, im, img0, cap, s in tqdm(dataset):
            detections = model.forward_image(im, img0)
            cur_video_time = milliseconds_to_time(cap.get(cv2.CAP_PROP_POS_MSEC))
            
            dets_to_sort = np.zeros((0, 6))
            for det in detections:
                dets_to_sort = np.vstack((dets_to_sort, np.array(det)))
            
            tracker_results, according_index = tracker.update(dets_to_sort)
            new_labels = tracker.update_labels(tracks=tracker_results[:, -1], labels=dets_to_sort[according_index, -1])
            final_ppe_detection = np.concatenate((tracker_results, new_labels.reshape(-1, 1)), axis=1)
            
            for i, (start_time, end_time) in enumerate(current_event_timesteps):
                if start_time < cur_video_time < end_time:
                    events_report[video_base_name][i]['total_number_of_frames'] += 1
                    
                    for box in final_ppe_detection:
                        if classes_mapping.get(int(box[-1])) != target_class:
                            events_report[video_base_name][i]['cls_fn'] += 1
                            break
                        if classes_mapping.get(int(box[-1])) == target_class:
                            events_report[video_base_name][i]['cls_tp'] += 1
                            break
                        
                    if dets_to_sort.shape[0] == 0:
                        events_report[video_base_name][i]['localization_fn'] += 1
            
            
            if out_video_writer:
                limage = LabeledImage(
                    name=Path(path).name,
                    height=img0.shape[0],
                    width=img0.shape[1],
                    bbox_list=[BBox(x1=int(box[0]), y1=int(box[1]), x2=int(box[2]), y2=int(box[3]), label=classes_mapping.get(int(box[-1]), 'hardhat_unrecognized'), track=int(box[-2])) for box in final_ppe_detection],
                )
                img = draw_bboxes_on_image(
                    labeled_image=limage,
                    img=img0,
                    put_text=True,
                    put_track_id=True,
                )
                out_video_writer.write(cv2.resize(img, (1280, 720)))
        if out_video_writer:
            out_video_writer.release()
            
            
    for vid_name, action_ids in events_report.items():
        for action_id in action_ids:
            if 'localization_fn' not in events_report[vid_name][action_id].keys():
                events_report[vid_name][action_id]['localization_fn'] = 0
    
    final_report = dict()
    for vid_name, action_ids in events_report.items():
        final_report[vid_name] = defaultdict(int)
        final_report[vid_name]['total_events'] = len(events[vid_name])
        
        for action_id in action_ids:
            
            total_frames = events_report[vid_name][action_id]['total_number_of_frames']
            localization_fn = events_report[vid_name][action_id]['localization_fn']
            cls_fn = events_report[vid_name][action_id]['cls_fn']
            
            cls_false_percent = (cls_fn / total_frames)
            localization_false_percent = (localization_fn / total_frames)
            total_false_percent = cls_false_percent + localization_false_percent
            
            final_report[vid_name]['TP'] += 1 if total_false_percent < false_actions_percent_threshold else 0
            final_report[vid_name]['classification FN'] += 1 if cls_false_percent > false_actions_percent_threshold else 0
            final_report[vid_name]['localization FN'] += 1 if localization_false_percent > false_actions_percent_threshold else 0
    
    report = pd.DataFrame(final_report).T
    sum_row = pd.DataFrame(report.sum(), columns=['Total']).T
    report = pd.concat([report, sum_row])
    report_dict = report.T.to_dict()
    save_json(report_dict, report_path)
    print(tabulate(report, headers ='keys', tablefmt = 'psql'))
    print(f"Report saved to {Path(report_path).resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--events_list", type=str, required=True)
    parser.add_argument("--input_size", type=int, nargs='+', required=True, help='inference size h,w')
    parser.add_argument("--class_names", type=str, nargs='+', required=True, help='class names should be in the same order as in the model')
    parser.add_argument("--report_path", type=str, required=False, default='report.xlsx')
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--viz_dir_path", type=str, default=None)
    parser.add_argument("--model_config_path", type=str, default=None)
    parser.add_argument("--tracker_config_path", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--iou_thresh", type=float, default=0.3)
    parser.add_argument("--score_thresh", type=float, default=0.5)
    parser.add_argument("--agnostic_nms", action='store_true', help="class-agnostic NMS")
    parser.add_argument("--target_class", type=str, help="target class for evaluation")
    parser.add_argument("--false_actions_percent_threshold", type=float, default=0.5)
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    run_evaluation(
        model_path=args.model_path,
        events_list_path=args.events_list,
        input_size=tuple(args.input_size),
        class_names=args.class_names,
        report_path=args.report_path,
        video_path=args.video_path,
        viz_dir_path=args.viz_dir_path,
        model_config_path=args.model_config_path,
        tracker_config_path=args.tracker_config_path,
        device_id=args.device,
        agnostic_nms=args.agnostic_nms,
        iou_thres=args.iou_thresh,
        conf_thres=args.score_thresh,
        target_class=args.target_class,
        false_actions_percent_threshold=args.false_actions_percent_threshold,
    )
