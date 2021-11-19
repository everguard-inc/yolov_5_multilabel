# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

import argparse
import sys
from pathlib import Path
import os
import pandas as pd
import cv2
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")
import torch.backends.cudnn as cudnn
from copy import deepcopy
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
import time
from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
from utils.general import apply_classifier, check_img_size, check_imshow, check_requirements, check_suffix, colorstr, \
    increment_path, is_ascii, non_max_suppression, print_args, save_one_box, scale_coords_1d, set_logging, \
    strip_optimizer, xyxy2xywh
from utils.plots import Annotator, colors
from utils.torch_utils import load_classifier, select_device, time_sync
from utils.metrics import predicts_to_multilabel_numpy, iou_batch_numpy, bbox_iou_numpy, bbox_io_mean_numpy
from utils.kalman_tracker import KFTracker, labels_dict_to_list, visualize_custom
from tqdm import tqdm
import joblib

def detection_metrics(predicts,targets):
    pass

@torch.no_grad()
def run(weights='yolov5s.pt',  # model.pt path(s)
        source='data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=1280,  # inference size (pixels)
        conf_thres=0.4,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        iou_thres_post = 0.8,
        max_det=100,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project='runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=True,  # use FP16 half-precision inference
        ):
    name = 'toledo_736_736_buffer_1'
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    half = True
    in_harness_classification_model = joblib.load('buckets_stats/rf_model.joblib')
    df = pd.read_csv("events.csv")
    df["result_1buffer"] = 0

    # Directorieso
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    w = weights[0] if isinstance(weights, list) else weights
    classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '']
    check_suffix(w, suffixes)  # check weights have acceptable suffix
    pt, onnx, tflite, pb, saved_model = (suffix == x for x in suffixes)  # backend booleans

    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    if pt:
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16
        if classify:  # second-stage classifier
            modelc = load_classifier(name='resnet50', n=2)  # initialize
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()
    elif onnx:
        check_requirements(('onnx', 'onnxruntime'))
        import onnxruntime
        session = onnxruntime.InferenceSession(w, None)
    else:  # TensorFlow models
        check_requirements(('tensorflow>=2.4.1',))
        import tensorflow as tf
        if pb:  # https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
            def wrap_frozen_graph(gd, inputs, outputs):
                x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped import
                return x.prune(tf.nest.map_structure(x.graph.as_graph_element, inputs),
                               tf.nest.map_structure(x.graph.as_graph_element, outputs))

            graph_def = tf.Graph().as_graph_def()
            graph_def.ParseFromString(open(w, 'rb').read())
            frozen_func = wrap_frozen_graph(gd=graph_def, inputs="x:0", outputs="Identity:0")
        elif saved_model:
            model = tf.keras.models.load_model(w)
        elif tflite:
            interpreter = tf.lite.Interpreter(model_path=w)  # load TFLite model
            interpreter.allocate_tensors()  # allocate
            input_details = interpreter.get_input_details()  # inputs
            output_details = interpreter.get_output_details()  # outputs
            int8 = input_details[0]['dtype'] == np.uint8  # is TFLite quantized uint8 model
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    ascii = is_ascii(names)  # names are ascii (use PIL for UTF-8)

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
    
    for path, img, im0s, vid_cap, frame_number in tqdm(dataset):
        if frame_number == 1:
            person_tracker = KFTracker(buffer_size = 1)
        video_name = path.split('/')[-1].split('.')[0]
        #print(video_name)
        uid = path.split('/')[-1].split('_')[-1].split('.')[0]
        try:
            event_type = df[df['Event uid']==uid]['Event Name'].iloc[0]
        except:
            print(f"Event absent {video_name} {uid}")
            event_type = "None"
            continue

        if event_type == "PPE - No Safety Vest":
            needed_labels = [3, 4]
        if event_type == "PPE - No Hard Hat":
            needed_labels = [6, 7]
        if event_type == "PPE - No Harness":
            needed_labels = [0, 1]
        if event_type == 'None':
            continue
        if onnx:
            img = img.astype('float32')
        else:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        # Inference
        if pt:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(img, augment=augment, visualize=visualize)[0]
        elif onnx:
            pred = torch.tensor(session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: img}))
        else:  # tensorflow model (tflite, pb, saved_model)
            imn = img.permute(0, 2, 3, 1).cpu().numpy()  # image in numpy
            if pb:
                pred = frozen_func(x=tf.constant(imn)).numpy()
            elif saved_model:
                pred = model(imn, training=False).numpy()
            elif tflite:
                if int8:
                    scale, zero_point = input_details[0]['quantization']
                    imn = (imn / scale + zero_point).astype(np.uint8)  # de-scale
                interpreter.set_tensor(input_details[0]['index'], imn)
                interpreter.invoke()
                pred = interpreter.get_tensor(output_details[0]['index'])
                if int8:
                    scale, zero_point = output_details[0]['quantization']
                    pred = (pred.astype(np.float32) - zero_point) * scale  # re-scale
            pred[..., 0] *= imgsz[1]  # x
            pred[..., 1] *= imgsz[0]  # y
            pred[..., 2] *= imgsz[1]  # w
            pred[..., 3] *= imgsz[0]  # h
            pred = torch.tensor(pred)
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]
        conf_thres_list = [0.5,0.5,0.2,0.5,0.5,0.2,0.5,0.5,0.2,0.4]
        pred = pred.detach().cpu().numpy()
        if pred.shape[0]>1:
            pred = predicts_to_multilabel_numpy(pred,iou_thres_post,conf_thres_list)
        else:
            pred = []
        # Second-stage classifier (optional)
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        if webcam:  # batch_size >= 1
            p, im0, frame = path[i], im0s[i].copy(), dataset.count
        else:
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
        p = Path(p)  # to Path
        save_path = str(save_dir / p.name)  # img.jpg
        clean_image = deepcopy(im0)

        if len(pred)!=0:
            for pred_i,_ in enumerate(pred):
                pred[pred_i][:4] = scale_coords_1d(img.shape[2:], pred[pred_i][:4], im0.shape).round()
                pred[pred_i][4:]= pred[pred_i][4:].astype(int)
            labels_dict,boxes = person_tracker.update(pred)
            if len(boxes)!=0:
                buckets_boxes,buckets_labels,person_boxes,person_labels,extra_boxes,extra_labels = not_in_harness_check(
                    boxes,labels_dict,in_harness_classification_model,video_name
                    )

                all_boxes = buckets_boxes.tolist()+person_boxes.tolist()+extra_boxes.tolist()
                all_labels = buckets_labels+person_labels+extra_labels
                im0 = visualize_custom(im0,all_boxes,all_labels)
                #print(all_labels)
                labels_for_count = sum(all_labels,[])
            
            else:
                labels_for_count = []

            wrong_boxes_per_frame = labels_for_count.count(needed_labels[1])
            try:
                df.loc[df['Event uid']==uid, 'result_1buffer'] += wrong_boxes_per_frame
            except:
                print("Event absent")
            
        save_img = True
        if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[0] != save_path:  # new video
                        vid_path[0] = save_path
                        if isinstance(vid_writer[0], cv2.VideoWriter):
                            vid_writer[0].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = 10
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 10, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[0] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[0].write(im0)
        
        del clean_image
    
    df.to_csv("result_1buffer.csv",index=False)
    # Print results
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


#[array([[     1149.1,       337.6,      1174.9,       407.4]]),
#  array([[     367.92,       398.1,      401.08,       437.9]]), 
# array([[     528.61,      308.71,      553.39,      321.29]])] 
# [{'harness': 1, 'vest': 3, 'hardhat': 6, 'crane': None}, 
# {'harness': None, 'vest': None, 'hardhat': None, 'crane': 9}, 
# {'harness': None, 'vest': None, 'hardhat': None, 'crane': 9}]
def not_in_harness_check(boxes,labels,model,path):
    buckets_boxes = []
    buckets_labels = []
    person_boxes = []
    person_labels = []
    extra_boxes = []
    extra_labels = []
    for i, label_dict in enumerate(labels):
        if 9 in list(label_dict.values()):
            buckets_boxes.append(boxes[i][0].astype(int))
            buckets_labels.append([9])
        elif 1 in list(label_dict.values()):
            person_boxes.append(boxes[i][0].astype(int))
            person_labels.append(list(label_dict.values()))
        else:
            extra_boxes.append(boxes[i][0].astype(int))
            extra_labels.append(list(label_dict.values()))
    buckets_boxes = np.array(buckets_boxes)
    person_boxes = np.array(person_boxes)
    extra_boxes = np.array(extra_boxes)
    if len(person_boxes)==0:
        return buckets_boxes,buckets_labels,person_boxes,person_labels,extra_boxes,extra_labels
    if len(buckets_boxes)==0:
        for index in range(len(person_labels)):
            person_labels[index].remove(1)
        return buckets_boxes,buckets_labels,person_boxes,person_labels,extra_boxes,extra_labels

    iou_matrix = iou_batch_numpy(buckets_boxes,person_boxes)
    iou_matrix = np.triu(iou_matrix,0)
    matched_indices = np.c_[(iou_matrix>0).nonzero()]
    for matched in matched_indices:
        b_i = matched[0]
        p_i = matched[1]
        bucket_box = buckets_boxes[b_i]
        person_box = person_boxes[p_i]
        bucket_box_width = bucket_box[2] - bucket_box[0]
        bucket_box_height = bucket_box[3] - bucket_box[1]
        person_box_width = person_box[2] - person_box[0]
        person_box_height = person_box[3] - person_box[1]
        bucket_box[0] =  (bucket_box[0] + bucket_box_width*0.05).astype(int)
        bucket_box[2] =  (bucket_box[2] - bucket_box_width*0.05).astype(int)
        bucket_area = bucket_box_width*bucket_box_height
        person_area = person_box_width*person_box_height
        person_centr = (person_box[0]+person_box[2])/2
        bucket_centr = (bucket_box[0]+bucket_box[2])/2
        bucket_person_width_union = max(bucket_box[2],person_box[2]) - min(bucket_box[0],person_box[0])
        centr_dist_norm = abs(bucket_centr - person_centr)/bucket_person_width_union
        head_loc = 1 if person_box[1] < bucket_box[1] else 0
        legs_loc = 1 if (person_box[3] < bucket_box[3]) and \
                    (person_box[3] > bucket_box[1]) else 0
        iou = bbox_iou_numpy(bucket_box,person_box)
        iom = bbox_io_mean_numpy(bucket_box,person_box)
        ['area_ratio', 'height_ratio', 'legs_loc', 'head_loc', 'iou', 'io_min',
       'centr_x_distance_norm'],
        test = {
        'area_ratio':float(person_area/bucket_area).__round__(2),
        'height_ratio':float((person_box[3]-person_box[1])/(bucket_box[3]-bucket_box[1])).__round__(2),
        'legs_loc':legs_loc,
        'head_loc':head_loc,
        'iou':float(iou).__round__(2),
        'io_min':float(iom).__round__(2),
        'centr_x_distance_norm': float(centr_dist_norm).__round__(2)
        }
        prediction = model.predict([list(test.values())])[0]
        if prediction==1:
            print(path)
            print("harness")
        if prediction==0:
            if 1 in person_labels[p_i]:
                person_labels[p_i].remove(1)
    return buckets_boxes,buckets_labels,person_boxes,person_labels,extra_boxes,extra_labels
    

        


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[1280], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.4, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)