
python inference.py \
--img_dir /media/data/vv/tests_data/dvc_datasets/dataset_for_demo/DemoHardhatDetection/val/images \
--predictions_dir output/exp4/predictions_dir \
--input_size 736 736 \
--weights /media/data/vv/models/yolov5m_2022-12-07-20-29-03_dataset_ppe_seah_hardhat_yolo_3x736x736_5160b832.pt \
--config configs/default_inference_config.yaml \
--tr 0.4 \
--viz_dir output/viz_dir3 \
--classes in_hardhat not_in_hardhat hardhat_unrecognized
