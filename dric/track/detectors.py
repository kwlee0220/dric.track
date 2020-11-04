import numpy as np
import cv2
import tensorflow as tf

from yolov3_tf2.models import YoloV3
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import convert_boxes
from deep_sort.detection import Detection
from tools import generate_detections as gdet

class YoloV3Detector:
    def __init__(self, weights='weights/yolov3.tf', labels_file='data/coco.names', image_size=416,
                    score_threshold=0, filter=None):
        self.class_names = [c.strip() for c in open(labels_file).readlines()]
        self.yolo = YoloV3(classes=len(self.class_names))
        self.yolo.load_weights(weights)
        self.image_size = image_size
        self.score_threshold = score_threshold
        self.filter = None if filter is None   \
                            else [i for i, x in enumerate(self.class_names) if x in filter]
        self.encoder = gdet.create_box_encoder('model_data/mars-small128.pb', batch_size=1)
    
    def detect(self, image):
        img_in = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, 416)

        boxes, scores, classes, nums = self.yolo.predict(img_in)
        
        nums = nums[0]
        scores = scores[0][:nums]
        classes = classes[0][:nums].astype(np.int32)
        boxes = boxes[0][:nums]
        indexes = scores >= self.score_threshold
        if self.filter is not None:
            indexes2 = [c in self.filter for c in list(classes)]
            indexes = np.array([b1 and b2 for b1, b2 in zip(list(indexes), indexes2)])
        scores = scores[indexes]
        classes = classes[indexes]
        boxes = convert_boxes(image, boxes[indexes])
        features = self.encoder(image, boxes)
        return [Detection(bbox, score, self.class_names[clazz], feature) for bbox, score, clazz, feature in
                                                            zip(boxes, scores, classes, features)]