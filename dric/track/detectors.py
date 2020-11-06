import numpy as np
import cv2
import tensorflow as tf

from yolov3_tf2.models import YoloV3
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import convert_boxes
from deep_sort.detection import Detection
from tools import generate_detections as gdet

from abc import ABCMeta, abstractmethod
import logging
class ObjectDetector(metaclass=ABCMeta):
    logger = logging.getLogger("dric.track.detector")
    logger.setLevel(logging.INFO)

    @abstractmethod
    def detect(self, image): pass

    def detect_from_images(self, images):
        return [self.detect(img) for img in images]

    def filter(self, class_names):
        return FilteringDetector(self, class_names)

class FilteringDetector(ObjectDetector):
    def __init__(self, detector, class_names):
        self.detector = detector
        self.class_names = class_names

    def detect_from_images(self, images):
        return [self.__filter_detections(detections) for detections in self.detector.detect_from_images(images)]
        
    def detect(self, image):
        self.__filter_detections(self.detector.detect(image))

    def __filter_detections(self, detections):
        return [det for det in detections if det.class_name in self.class_names]

class YoloV3Detector(ObjectDetector):
    def __init__(self, weights='weights/yolov3.tf', labels_file='data/coco.names', image_size=416,
                    score_threshold=0, filter=None):
        self.class_names = [c.strip() for c in open(labels_file).readlines()]
        self.yolo = YoloV3(classes=len(self.class_names))
        self.yolo.load_weights(weights)
        self.image_size = image_size
        self.score_threshold = score_threshold
        self.encoder = gdet.create_box_encoder('model_data/mars-small128.pb', batch_size=1)

    def detect_from_images(self, images):
        tf_images = [self._to_tf_image(img) for img in images]
        tf_images = tf.concat(tf_images, axis=0)
        tf_images = transform_images(tf_images, self.image_size)

        boxes, scores, classes, nums = self.yolo.predict(tf_images)
        return [self._to_output(image, boxes, scores, classes, nums)
                    for image, boxes, scores, classes, nums in zip(images, boxes, scores, classes, nums)]

    def detect(self, image):
        tf_images = self._to_tf_image(image)
        tf_images = transform_images(tf_images, self.image_size)

        boxes, scores, classes, nums = self.yolo.predict(tf_images)
        return self._to_output(image, boxes[0], scores[0], classes[0], nums[0])

    def _to_output(self, image, boxes, scores, classes, nums):
        scores = scores[:nums]
        classes = classes[:nums].astype(np.int32)
        boxes = boxes[:nums]
        indexes = scores >= self.score_threshold
        scores = scores[indexes]
        classes = classes[indexes]
        boxes = convert_boxes(image, boxes[indexes])
        features = self.encoder(image, boxes)
        return [Detection(bbox, score, self.class_names[clazz], feature) for bbox, score, clazz, feature in
                                                            zip(boxes, scores, classes, features)]

    def _to_tf_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tf_image = tf.expand_dims(image, 0)
        return tf_image