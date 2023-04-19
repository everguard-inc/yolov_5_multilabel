## Setup

```
git clone https://github.com/everguard-inc/yolov_5_multilabel.git --recursive
cd yolov_5_multilabel
sudo docker build -t $(whoami)/yolov5 .

sudo docker run \
--rm -it \
--gpus all \
--shm-size 8G \
--hostname $(hostname) \
--mount type=bind,source="$PWD",target=/app \
--mount type=bind,source="/home",target=/home \
--mount type=bind,source="/media",target=/media \
--privileged \
$(whoami)/yolov5
```

# Evaluate

Specify model path and input size in the `config.yaml`

```
python evaluate.py \
--val_ann_coco /path/to/val.json \
--images_dir /path/to/images \
--config_path config.yaml \
--iou_threshold 0.5 \
--score_threshold 0.5
```

# Evaluate Events
```
python evaluate_events.py \
--model_path /path/to/model.py \
--events_list /path/to/events.csv \
--input_size 736 736 \
--class_names in_hardhat not_in_hardhat hardhat_unrecognized 
--video_path /path/to/video \
--target_class not_in_hardhat \
--false_actions_percent_threshold 0.9 \

--viz_dir_path /path/to/viz_dir(optional) \
--model_config_path /path/to/model_config.yaml(optional) \
--device device_id(optional) \
--iou_thresh 0.3(optional) \
--score_thresh 0.5(optional) \
--report_path /path/to/report.xlsx(optional)
```

