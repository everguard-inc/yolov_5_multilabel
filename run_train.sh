CUDA_VISIBLE_DEVICES=1 python train.py \
--img 736 --batch 4 --epochs 3 \
--data data/demo.yaml \
--weights yolov5m.pt --workers 4