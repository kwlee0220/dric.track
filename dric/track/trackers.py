import numpy as np
import cv2
import tensorflow as tf

from yolov3_tf2.models import YoloV3
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import convert_boxes

from deep_sort import nn_matching
from deep_sort import preprocessing
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort.track import TrackState

class Track:
    def __init__(self, id, tlbr, label=None, detector_confidence=0, tentative=True):
        self.__dict__['id'] = id
        self.__dict__['label'] = label
        self.__dict__['tlbr'] = tlbr
        self.__dict__['detector_confidence'] = detector_confidence
        self.__dict__['tentative'] = tentative

    @property
    def id(self):
        return self.__dict__['id']

    @property
    def label(self):
        return self.__dict__['label']

    @property
    def tlbr(self):
        return self.__dict__['tlbr']

    @property
    def detector_confidence(self):
        return self.__dict__['detector_confidence']

    @property
    def tentative(self):
        return self.__dict__['tentative']

    @property
    def is_confirmed(self):
        return t.detector_confidence > 0 and not t.tentative

    def __setattr__(self, attr, value):
        if attr == 'tentative':
            self.__dict__['tentative'] = value
        else:
            raise AttributeError(attr + ' not allowed')

    def __repr__(self):
        tent = ',T' if self.__dict__['tentative'] else ''
        return '%s(%d):%.3f%s' % (self.__dict__['label'], self.__dict__['id'],
                                    self.__dict__['detector_confidence'], tent)

class BufferedTracker:
    def __init__(self, tracker, buffer_size=10):
        self.tracker = tracker
        self.buffer = []
        self.buffer_size = buffer_size

    def track(self, image=None):
        tracks = self.tracker.track(image)

        keys = [t.id for t in tracks if not t.tentative]
        for _, ts in self.buffer:
            for t in ts:
                if t.tentative and t.id in keys:
                    t.tentative = False
        self.buffer.append((image, tracks))
        if len(self.buffer) >= self.buffer_size:
            head = self.buffer[0]
            self.buffer = self.buffer[1:]
            return head
        else:
            return (None, list())

"""
        if len(self.buffer) >= self.buffer_size or image is None:
            for i in range(len(self.buffer)-1, 0, -1):
                keys = [t.id for t in self.buffer[i][1] if not t.tentative]
                for j in range(i):
                    print(i, j)
                    for t in self.buffer[j][1]:
                        if t.tentative and t.id in keys:
                            t.tentative = False
            head = self.buffer[0]
            self.buffer = self.buffer[1:]
            return head
        else:
            return (None, list())
"""

class DeepSortTracker:
    def __init__(self, detector, n_init=3, max_age=30):
        self.detector = detector

        max_cosine_distance = 0.5
        nn_budget = None
        self.nms_max_overlap = 0.8

        metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric, n_init=n_init, max_age=max_age)
    
    def track(self, image=None):
        detections = self.detector.detect(image) if image is not None else []

        boxes = [d.tlwh for d in detections]                                    
        boxs = np.array(boxes)
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        self.tracker.predict()
        self.tracker.update(detections)
        return [self._to_output(track) for track in self.tracker.tracks if not track.is_deleted()]

    def _to_output(self, t):
        box = t.to_tlbr().astype(np.int32)
        return Track(id=t.track_id, label=t.class_name, tlbr=box, detector_confidence=t.detector_confidence,
                        tentative=t.is_tentative())