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


# Visualize or Inference

```
python inference.py \
--img_dir path/to/images \
--predictions_dir path/to/save/json_predictions \
--input_size 736 736 \
--weights models/yolov5m_2023-01-23-11-59-36_dataset_zekelman_person_person_on_truck_3x736x736_d787915d.pt \
--config configs/default_inference_config.yaml \
--tr 0.4 \
--viz_dir path/to/save/viz_images \
--classes on_truck not_on_truck
```
