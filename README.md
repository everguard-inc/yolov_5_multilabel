Run train with 
```
python3 train.py --img 416 --batch 2 --epochs 30 --data dataset/data.yaml --weights yolov5s.pt
```
Run test with
```
python3 detect.py --weights runs/train/exp/weights/best.pt --img 416 --conf 0.5 --source dataset/test
```