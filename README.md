## Setup

```
git clone https://github.com/everguard-inc/yolov_5_multilabel.git --recursive

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


