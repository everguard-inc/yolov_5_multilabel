
python inference.py \
--config configs/inference.yaml \
--img_dir some/in \
--predictions_dir some/out/pred \
--viz_dir some/out/viz \
--input_size 736 736 \
--weights /home/st/tasks/3.04_yolo_ppe/yolov5m_2023-03-04-03-18-02_dataset_ppe_seah_hardhat_coco_3x736x736_6b34cc65.pt \
--tr 0.4 \
--classes in_hardhat not_in_hardhat hardhat_unrecognized
