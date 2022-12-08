from detector import run_inference
import json

from eg_data_tools.annotation_processing.coco_utils.coco_read_write import open_coco
from eg_data_tools.model_evaluation.object_detection_metrics import evaluate_limages
from eg_data_tools.annotation_processing.converters.convert_from_detections_to_labeled_image import convert_detections_to_labeled_images
import argparse


def get_classes_from_coco(coco_path):

    with open(coco_path, "r") as jfile:
        coco_ann = json.load(jfile)

    coco_categories = coco_ann["categories"]

    if isinstance(coco_categories[0], list):
        coco_categories = coco_categories[0]

    category_dict = dict()
    for category_info in coco_categories:
        category_dict[int(category_info["id"])] = category_info["name"]

    sorted_classes = list()
    for id in sorted(list(category_dict.keys())):
        sorted_classes.append(category_dict[id])

    return sorted_classes


def evaluate_detector(
    val_ann_coco,
    images_dir,
    config_path,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.5,
):

    val_labeled_images = open_coco(val_ann_coco)
    classes = get_classes_from_coco(val_ann_coco)

    predictions = run_inference(
        img_dir = images_dir,
        config_path = config_path,
        img_names_to_detect = [limage.name for limage in val_labeled_images],
    )
    
    predicted_limages = convert_detections_to_labeled_images(
        detections_by_images=predictions,
        detection_label_to_class_name={i: class_name for i, class_name in enumerate(classes)},
        conf_threshold=score_threshold,
        img_folder_path=images_dir,
    )
    metrics = evaluate_limages(
        gt_limages=val_labeled_images,
        predicted_limages=predicted_limages,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
    )

    print(metrics)

    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--val_ann_coco", type=str, required=True)
    parser.add_argument("--config_path", type=str, default="config.yaml")
    parser.add_argument("--score_threshold", type=float, required=True)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    args = parser.parse_args()

    evaluate_detector(
        val_ann_coco=args.val_ann_coco,
        images_dir=args.images_dir,
        config_path=args.config_path,
        iou_threshold=args.iou_threshold,
        score_threshold=args.score_threshold,
    )
