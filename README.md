## Setup



```
git clone https://github.com/everguard-inc/yolov_5_multilabel.git --recursive
cd yolov_5_multilabel

# Build
docker build -t $USER/yolov5 .

# Run
docker run \
--rm -it \
--gpus all \
--shm-size 8G \
--workdir $(pwd) \
--user $(id -u):$(id -g) \
--mount type=bind,source=$HOME,target=$HOME \
$USER/yolov5
```

# Evaluate

Specify model parameters in the `config.yaml`

```
python evaluate.py \
--val_ann_coco /path/to/val.json \
--images_dir /path/to/images \
--config_path config.yaml
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
