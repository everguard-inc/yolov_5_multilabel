import numpy as np
from typing import Any, Dict, NoReturn, Optional
from collections import Counter, deque
import cv2
from collections import Counter, deque
from typing import Any, Dict, List, Optional
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

np.random.seed(0)


def labels_dict_to_list(labels : dict) -> List:
    post_labels = []
    for _,value in labels.items():
        if value:
            post_labels.append(value)

    return post_labels     

def visualize_custom(image, boxes, labels_list):
    for i, box in enumerate(boxes):
        box = box
        labels = labels_list[i]
        image = plot_one_box(box,labels,image)
    return image

def plot_one_box(x, labels, track_id, img, line_thickness=None):
    """
    description: Plots one bounding box on image img,
                 this function comes from YoLov5 project.
    param: 
        x:      a box likes [x1,y1,x2,y2]
        img:    a opencv image object
        color:  color to draw rectangle, such as (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return
    """
    label_names = ['in_harness', 'not_in_harness', 'harness_unrecognized', 'in_vest',\
        'not_in_vest','vest_unrecognized','in_hardhat','not_in_hardhat','hardhat_unrecognized','crane_bucket']
        
    labels = labels_dict_to_list(labels)
    tl = (
        line_thickness or round(0.001 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    box_color = (255,0,0)#color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    tf = max(tl - 1, 1)
    cv2.rectangle(img, c1, c2, box_color, thickness=tl)
    if track_id:
        cv2.putText(img,str(track_id),(c1[0], c1[1] + 10),0,tl / 4,box_color, thickness=tf)
    labels_space = 0
    for label in labels:
        text_color = (0,255,0) if label==0 or label==3 or label==6 else (0,0,255)
          # font thickness
        t_size = cv2.getTextSize(label_names[label], 0, fontScale=tl / 6, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.putText(
            img,
            label_names[label],
            (c1[0], c1[1] - labels_space),
            0,
            tl / 4,
            text_color,
            thickness=tf,
            )
        labels_space += 20

    return img



def linear_assignment(cost_matrix):
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
    """
    Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return(o)


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))


class KFBuffer:
    def __init__(self, buffer: float = 5):
        """
        Params:
            buffer: float - time in seconds to store values for making choice
        """
        self._buffer = buffer
        self._history: Dict[deque] = {
            'harness':deque(maxlen=buffer),
            'vest':deque(maxlen=buffer),
            'hardhat':deque(maxlen=buffer),
            'crane':deque(maxlen=buffer)}
        self._history_len = 0

    def _choose_value(self) -> Any:
        results : Dict[int] = {'harness':None, 'vest':None, 'hardhat':None, 'crane':None}
        for key in list(self._history.keys()):
            most_common_label = Counter(self._history[key]).most_common(1)
            if len(most_common_label)>0:
                most_common_label = most_common_label[0][0]
            else:
                most_common_label = None 
            results[key] = most_common_label
        return results
    
    def _get_history_len(self):
        max_len_history = 0
        for key in list(self._history.keys()):
            len_history = len(self._history[key])
            if len_history>max_len_history:
                max_len_history = len_history
        self._history_len = max_len_history
        return self._history_len


    def update(self, labels: List) -> Optional[Any]:
        for label in labels:
            if label in [0,1]:
                self._history['harness'].append(label)
            elif label in [3,4]:
                self._history['vest'].append(label)
            elif label in [6,7]:
                self._history['hardhat'].append(label)
            elif label in [9]:
                self._history['crane'].append(label)
        if self._get_history_len() == self._buffer:
            return self._choose_value()
        else:
            return {'harness':None, 'vest':None, 'hardhat':None, 'crane':None}

    def get_current(self):
        if  self._get_history_len() == self._buffer:
            return self._choose_value()
        else:
            return {'harness':None, 'vest':None, 'hardhat':None, 'crane':None}
        
        

class KFTimeBuffer:
    def __init__(self, buffer: float = 5):
        """
        Params:
            buffer: float - time in seconds to store values for making choice
        """
        self._buffer = buffer
        self._history: Dict[List[Dict[str, Any]]] = {
            'harness':deque(maxlen=buffer),
            'vest':deque(maxlen=buffer),
            'hardhat':deque(maxlen=buffer),
            'crane':deque(maxlen=buffer)}
        self._first_time: Dict[Optional[float]] = {'harness':None,'vest':None,'hardhat':None,'crane':None}
        
        
class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox, buffer_size):
        """
        Initialises a tracker using initial bounding box.
        """
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.buffer = KFBuffer(buffer = buffer_size)
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0],
                              [0, 1, 0, 0, 0, 1, 0],
                              [0, 0, 1, 0, 0, 0, 1],
                              [0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01  # todo - customize kalman parameters

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        Updates buffer with velocity > threshold
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))
        self.buffer.update(bbox[4:])

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if((self.kf.x[6] + self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if(self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)

    def get_labels(self):
        """
        Returns the current bounding box labels.
        """
        return self.buffer.get_current()
    

def associate_detections_to_trackers(dets, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    detections = []
    if len(dets) > 0:
        for detection in dets:
            x1 = detection[0]
            y1 = detection[1]
            x2 = detection[2]
            y2 = detection[3]
            if x1 != x2 and y1 != y2:
                detections.append([x1, y1, x2, y2, 1])
        detections = np.array(detections)
    else:
        detections = np.empty((0, 5))

    if(len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
    iou_matrix = iou_batch(detections, trackers)
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if(d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if(t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if(iou_matrix[m[0], m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if(len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class KFTracker(object):

    def __init__(self, buffer_size):
        """
        Sets key parameters for tracker
        """
        self.max_age = 1
        self.min_hits = 3
        self.iou_threshold = 0.19
        self.trackers = []
        self.frame_count = 0
        self.buffer_size = buffer_size

    def update(self, boxes):
        """
        Params:
        lines - a list of lines
        Requires: this method must be called once for each frame even with no lines
        Returns list in format [line, is_moving]
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        dets = []
        track_id = -1
        if len(boxes) > 0:
            for box in boxes:
                dets.append(box)
            dets = np.array(dets)
        if len(dets)==0:
            dets = np.empty((0, 5)) 
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)
        # update matched trackers with assigned detections
        labels = []
        for m in matched:
            self.trackers[m[1]].update(dets[m[0]])
            labels.append(self.trackers[m[1]].get_labels())
        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            track_id+=1
            trk = KalmanBoxTracker(dets[i],self.buffer_size)
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1))
            i -= 1
            # remove dead tracks
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)
                
        return labels

    def visualize(self, image):
        for trk in self.trackers:
            box = np.int0(trk.get_state())[0]
            labels = trk.get_labels()
            track_id = trk.id
            image = plot_one_box(box,labels,track_id,image)
        return image

    