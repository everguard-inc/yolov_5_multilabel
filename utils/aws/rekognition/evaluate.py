import numpy as np
from copy import deepcopy
from typing import Dict, List
from tqdm import tqdm
import json
import argparse

from eg_data_tools.annotation_processing.coco_utils.coco_read_write import open_coco
from eg_data_tools.data_units.data_units import LabeledImage
from eg_data_tools.data_units.data_units import BBox, LabeledImage
from eg_data_tools.annotation_processing.labeled_data_processing.labels_processing import get_and_count_labels_percentage, get_classes
from eg_data_tools.annotation_processing.labeled_data_processing.labeled_images_processing import make_dict_from_labeled_image_list
from eg_data_tools.sets_utils.sets_processing import match_values
from eg_data_tools.model_evaluation.cocoeval import COCOeval

def get_detected_bboxes_by_types_on_limage(
    gt_limage: LabeledImage, 
    predicted_limage: LabeledImage, 
    iou_threshold: float, 
    score_threshold: float, 
    class_names: List[str]
):
    """
    Counts metrics for single LabeledImage
    :param gt_limage: Ground truth image
    :param predicted_limage: Predicted images with bbox confidence scores
    :param iou_threshold: Minimal IOU value between bboxes for TP
    :param score_threshold: Minimal bbox confidence score to take them into account
    :param class_names: List of class names to count metrics
    :return: Counts of detection types on LabeledImage
    :rtype: Dict[str, Dict[str, int]]
    """
    result: Dict[str, Dict[str, int]] = {class_name: {"tp": list(), "fp": list(), "fn": list()} for class_name in class_names}

    for class_name in class_names:  # Compare bboxes only with bboxes with same class
        # Select gt bboxes with class
        gt_bboxes = gt_limage.get_bboxes_with_class(class_names=[class_name])

        # Select predicted bboxes with class with confidence score more than score_threshold
        predicted_bboxes = list(
            filter(lambda x: x.score > score_threshold, predicted_limage.get_bboxes_with_class(class_names=[class_name]))
        )

        # Get pairs of closest bboxes indicies
        gt_bbox_ids, predicted_bbox_ids = match_values(
            values1=gt_bboxes,
            values2=predicted_bboxes,
            cost_function=lambda bbox1, bbox2: bbox1.iou_with_bbox(bbox2),
        )

        # fn is gt bboxes without pair
        result[class_name]["fn"] = [bbox for i, bbox in enumerate(gt_bboxes) if i not in gt_bbox_ids]

        # fp is predicted bboxes without pair
        result[class_name]["fp"] = [bbox for i, bbox in enumerate(predicted_bboxes) if i not in predicted_bbox_ids]

        for predicted_bbox_id, gt_bbox_id in zip(predicted_bbox_ids, gt_bbox_ids):
            # if boxes in pair
            if gt_bboxes[gt_bbox_id].iou_with_bbox(predicted_bboxes[predicted_bbox_id]) > iou_threshold:
                result[class_name]["tp"].append(predicted_bboxes[predicted_bbox_id])
            else:
                result[class_name]["fn"].append(gt_bboxes[gt_bbox_id])
                result[class_name]["fp"].append(predicted_bboxes[predicted_bbox_id])

    return result

def evaluate_limages(
    gt_limages: List[LabeledImage],
    predicted_limages: List[LabeledImage],
    iou_threshold: float = 0.5,
    score_threshold: float = 0.5,
    disable_progress_bar: bool = False,
):
    gt_class_names = get_classes(gt_limages)
    predicted_class_names = get_classes(predicted_limages)
    class_names = list(set(gt_class_names + predicted_class_names))

    metrics = {class_name: {"tp": 0, "fp": 0, "fn": 0} for class_name in class_names}

    gt_limages_dict = make_dict_from_labeled_image_list(gt_limages)
    predicted_limages_dict = make_dict_from_labeled_image_list(predicted_limages)

    assert set(gt_limages_dict.keys()) == set(predicted_limages_dict.keys()), "Received different images number for gt and prediction"

    for limage_name in tqdm(gt_limages_dict.keys(), disable=disable_progress_bar):
        # {class_name: {'tp': int, 'fp': int, 'fn': int}, ... }

        detected_bboxes_by_classes = get_detected_bboxes_by_types_on_limage(
            gt_limage=gt_limages_dict[limage_name],
            predicted_limage=predicted_limages_dict[limage_name],
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
            class_names=class_names,
        )

        for class_name in detected_bboxes_by_classes.keys():
            tp = len(detected_bboxes_by_classes[class_name]["tp"])
            metrics[class_name]["tp"] += tp
            fp = len(detected_bboxes_by_classes[class_name]["fp"])
            metrics[class_name]["fp"] += fp
            fn = len(detected_bboxes_by_classes[class_name]["fn"])
            metrics[class_name]["fn"] += fn

    for class_name in metrics:
        fp = metrics[class_name]["fp"]
        tp = metrics[class_name]["tp"]
        fn = metrics[class_name]["fn"]

        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1_score = 2 * precision * recall / (precision + recall + 1e-7)

        metrics[class_name]["precision"] = precision
        metrics[class_name]["recall"] = recall
        metrics[class_name]["f1_score"] = f1_score
    print(metrics)
    return metrics

def get_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    parser.add_argument("--gt", type=str, required=True, help="path to gt json")
    parser.add_argument("--yolo", type=str, required=True, help="path to yolo json")
    parser.add_argument("--yolo", type=str, required=True, help="path to aws json")

    return parser.parse_args()

def main(parser: argparse.ArgumentParser) -> None:
    args = get_args(parser)
    with open(args.gt) as json_file:
        gt = json.load(json_file)
    gt_limage = []
    good_images = []
    #a =0
    print(len(gt.keys()))
    for i in gt.keys():
        #good = 1
        bbox_list = []
        for c in gt[i]:
            bbox_list.append(BBox(c[1], c[2], c[3], c[4], c[0], score = 1))
            #if c[3]*c[4] < 6000 or c[3]*c[4] > 10000: #for filtering if needed
                #good = 0
        #if good == 1:
            #good_images.append(i)
            #a = a+len(bbox_list)
        gt_limage.append(LabeledImage(name = i, bbox_list=bbox_list))
    #print(a)
    with open(args.yolo) as json_file:
        yolo = json.load(json_file)
    yolo_limage = []
    for i in yolo.keys():
        #if i in good_images:
            bbox_list = []
            for c in yolo[i]:
                bbox_list.append(BBox(c[1], c[2], c[3], c[4], c[0], score = 1))
            yolo_limage.append(LabeledImage(name = i, bbox_list=bbox_list))

    with open(args.aws) as json_file:
        aws = json.load(json_file)
    aws_limage = []
    for i in aws.keys():
        #if i in good_images:
            bbox_list = []
            for c in aws[i]:
                bbox_list.append(BBox(c[1], c[2], c[3], c[4], c[0], score = 1))
            aws_limage.append(LabeledImage(name = i, bbox_list=bbox_list))  

    print(len(gt_limage))
    print(len(aws_limage))
    evaluate_limages(gt_limage, aws_limage)
    evaluate_limages(gt_limage, yolo_limage)


if __name__ == "__main__":
    main(argparse.ArgumentParser())
