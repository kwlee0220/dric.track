import numpy as np
import cv2

def draw_boxes(image, boxes, color=(0,0,255), thickness=2):
    for box in boxes:
        image = cv2.rectangle(image, tuple(box[:2]), tuple(box[2:]), color, thickness)
    return image

def draw_track_labels(image, tracks, color=(0,255,0), font=cv2.FONT_HERSHEY_COMPLEX_SMALL,
            size=1, thickness=2):
    for t in tracks:
        msg = '%s(%s)' % (t.label, t.id)
        image = cv2.putText(image, msg, tuple(t.tlbr[:2]), font, size, color, thickness)
    return image