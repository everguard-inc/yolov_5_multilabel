import os

import cv2

from detector.yolov5_multilabel_detector import Yolov5MultilabelDetector


def run_inference(config_path: str, img_dir: str):
    detector = Yolov5MultilabelDetector(config_path=config_path)
    for i, img_name in enumerate(sorted(os.listdir(img_dir))):
        detection = detector.forward_image(cv2.imread(os.path.join(img_dir, img_name)))
        print(i, img_name, detection)


if __name__ == "__main__":
    run_inference(
        config_path='config.yaml',
        img_dir='/media/data/vv/tasks/2021-12-07_debug_yolov5/img'
    )