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
    increment_path, is_ascii, non_max_suppression, print_args, save_one_box, scale_coords, set_logging, \
    strip_optimizer, xyxy2xywh
from utils.plots import Annotator, colors
from utils.torch_utils import load_classifier, select_device, time_sync
from utils.metrics import predicts_to_multilabel_numpy


def detection_metrics(predicts,targets):
    pass

@torch.no_grad()
def run(weights='yolov5s.pt',  # model.pt path(s)
        source='data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=736,  # inference size (pixels)
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
    name = 'processed'
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    half = True
    
    df = pd.read_csv("events.csv")
    df["Model Result"] = 0

    # Directories
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
    time_file = open("runs/detect/time.txt", "a")
    #fp_mean_color_list = []
    #fp_box_height_list = []
    #fn_mean_color_list = []
    #fn_box_height_list = []
    #tp_mean_color_list = []
    #tp_box_height_list = []
    for path, img, im0s, vid_cap, frame_number in dataset:
        print('frame_number  = ',frame_number)
        video_name = path.split('/')[-1].split('.')[0]
        uid = path.split('/')[-1].split('_')[-1].split('.')[0]
        event_type = df[df['Event uid']==uid]['Event Name'].iloc[0]
        val_status = df[df['Event uid']==uid]['Validation status'].iloc[0]
        print(uid,event_type,val_status)
        if event_type == "PPE - No Safety Vest":
            needed_labels = [3, 4]
        if event_type == "PPE - No Hard Hat":
            needed_labels = [6, 7]
        if event_type == "PPE - No Harness":
            needed_labels = [0, 1]
        start = time.time()
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
        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]
        conf_thres_list = [0.5,0.5,0.2,0.5,0.5,0.2,0.5,0.5,0.2,0.5]
        pred = pred.detach().cpu().numpy()
        if pred.shape[0]>1:
            pred = predicts_to_multilabel_numpy(pred,iou_thres_post,conf_thres_list)
        else:
            pred = np.expand_dims(pred, 0)
        # Second-stage classifier (optional)
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        if webcam:  # batch_size >= 1
            p, im0, frame = path[i], im0s[i].copy(), dataset.count
        else:
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
        end = time.time()
        time_file.write(f'{end-start}\n')
        p = Path(p)  # to Path
        save_path = str(save_dir / p.name)  # img.jpg
        txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        imc = im0.copy() if save_crop else im0  # for save_crop
        clean_image = deepcopy(im0)
        
        annotator = Annotator(im0, line_width=line_thickness, pil=not ascii)
        # Process predictions
        for i, det in enumerate(pred):  # per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                s = ''
                det = det.astype(np.float64)
                det[:4] = scale_coords(img.shape[2:], det[:4], im0.shape).round()
                labels = det[0,4:].astype(int)
                wrong_boxes_per_frame = list(labels).count(needed_labels[1])
                df.loc[df['Event uid']==uid, 'Model Result'] += wrong_boxes_per_frame
                # Print results
                for c in labels:
                    s += f"{c};"  # add to string
                # Write results
                xyxy = det[0][:4].astype(int)
                fp = wrong_boxes_per_frame>0 and val_status=='FP'
                fn = wrong_boxes_per_frame>=0 and val_status=='TP'
                if fp and fn:
                    print("\nBOTH\n")
                infraction_box = clean_image[xyxy[1]:xyxy[3],xyxy[0]:xyxy[2]]
                mean_color = np.mean(infraction_box)
                box_height = abs(xyxy[3]-xyxy[2])
                if fp:
                    cv2.imwrite(f'color_height/fp_height/{video_name}_{str(frame_number).zfill(7)}.jpg',infraction_box)
                    cv2.imwrite(f'color_height/fp_color/{video_name}_{str(frame_number).zfill(7)}.jpg',infraction_box)
                    #fp_mean_color_list.append(mean_color)
                    #fp_box_height_list.append(box_height)
                elif fn:
                    cv2.imwrite(f'color_height/fn_height/{video_name}_{str(frame_number).zfill(7)}.jpg',infraction_box)
                    cv2.imwrite(f'color_height/fn_color/{video_name}_{str(frame_number).zfill(7)}.jpg',infraction_box)
                    #fn_mean_color_list.append(mean_color)
                    #fn_box_height_list.append(box_height)
                else:
                    cv2.imwrite(f'color_height/tp_height/{video_name}_{str(frame_number).zfill(7)}.jpg',infraction_box)
                    cv2.imwrite(f'color_height/tp_color/{video_name}_{str(frame_number).zfill(7)}.jpg',infraction_box)
                    #tp_mean_color_list.append(mean_color)
                    #tp_box_height_list.append(box_height)
                label = needed_labels[1] if needed_labels[1] in labels else None
                if save_img or save_crop or view_img:  # Add bbox to image
                        #c = int(cls)  # integer class
                        #label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label_infraction(xyxy, label)
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
            
            # Print time (inference-only)
            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[0] != save_path:  # new video
                        vid_path[0] = save_path
                        if isinstance(vid_writer[0], cv2.VideoWriter):
                            vid_writer[0].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = 20
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        #vid_writer[0] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    #vid_writer[0].write(im0)
        '''
        if infraction:
            predictions_to_save = []
            for i, det in enumerate(pred):
                if len(det):
                    det = det.astype(np.float64)
                    det[:4] = scale_coords(img.shape[2:], det[:4], im0.shape).round()
                    predictions_to_save.append(list(det[0]))
            cv2.imwrite(f'runs/infraction/images/{video_name}_{str(frame_number).zfill(7)}.jpg',clean_image)
            predictions_to_save = np.array(predictions_to_save)
            with open(f'runs/infraction/labels/{video_name}_{str(frame_number).zfill(7)}.npy', 'wb') as f:
                np.save(f, predictions_to_save)
        '''
        del clean_image
    
    '''
    fp_box_height_list = np.array(fp_box_height_list)
    fp_mean_color_list = np.array(fp_mean_color_list)
    fn_box_height_list = np.array(fn_box_height_list)
    fn_mean_color_list = np.array(fn_mean_color_list)
    tp_box_height_list = np.array(tp_box_height_list)
    tp_mean_color_list = np.array(tp_mean_color_list)
    
    with open(f'color_height/fp_box_height_list.npy', 'wb') as f:
        np.save(f, fp_box_height_list)
    with open(f'color_height/fp_mean_color_list.npy', 'wb') as f:
        np.save(f, fp_mean_color_list)
    with open(f'color_height/fn_box_height_list.npy', 'wb') as f:
        np.save(f, fn_box_height_list)
    with open(f'color_height/fn_mean_color_list.npy', 'wb') as f:
        np.save(f, fn_mean_color_list)
    with open(f'color_height/tp_box_height_list.npy', 'wb') as f:
        np.save(f, tp_box_height_list)
    with open(f'color_height/tp_mean_color_list.npy', 'wb') as f:
        np.save(f, tp_mean_color_list)
    '''
    df.to_csv("results.csv",index=False)
    # Print results
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[736], help='inference size h,w')
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
