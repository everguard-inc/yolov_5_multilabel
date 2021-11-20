download the model
```
aws s3 cp s3://eg-ukraine-team/rodion/toledo_demo/model10cl/new_data/yolov5_multilabel_736_736_15_11_21.pt models/checkpoints/yolov5_multilabel_736_736_15_11_21.pt
```
To run in docker
```
sh docker/build.sh
sh docker/run.sh
```
Run train with 
```
python3 train.py --img 416 --batch 2 --epochs 30 --data dataset/data.yaml --weights yolov5s.pt
```
Run test with
```
python3 detect.py --weights runs/train/exp/weights/best.pt --img 416 --conf 0.5 --source dataset/test
```
