from absl import flags
import sys
FLAGS = flags.FLAGS
FLAGS(sys.argv)

import time
import cv2
from dric.track import YoloV3Detector, FilteringDetector
from dric.track import DeepSortTracker, BufferedTracker, draw_boxes, draw_track_labels
from deep_sort.track import Track

detector = YoloV3Detector().filter(['car','truck'])
tracker = DeepSortTracker(detector, max_age=10, n_init=5)
#tracker = BufferedTracker(detector, tracker, buffer_size=5)

times = []
fps = 0

def track_detections(image, detections):
    tracks = tracker.track_detections(detections)
    if len(tracks) > 0:
        confirmeds =  [t for t in tracks if t.detector_confidence > 0 and not t.tentative]
        image = draw_boxes(image, [t.tlbr for t in confirmeds], thickness=2)
        image = draw_track_labels(image, confirmeds, thickness=1)

        tentatives = [t for t in tracks if t.detector_confidence == 0 or t.tentative]
        image = draw_boxes(image, [t.tlbr for t in tentatives], color=(255,255,255), thickness=1)
#        tracked_img = draw_track_labels(image, tentatives)

        fps_msg = 'FPS: %.1f' % (fps)
        image = cv2.putText(image, fps_msg, (5, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255), 2)
    
    return image

batch_size = 1
batch = []
capture_count = 0

vid = cv2.VideoCapture('./data/cam_6.mp4')

for i in range(780):
    vid.read()
    capture_count += 1

while vid is not None:
    for i in range(batch_size):
        _, img = vid.read()
        if img is None:
            break
        batch.append(img)

    started = time.time()
    detections_list = detector.detect_from_images(batch)
    for image, detections in zip(batch, detections_list):
        image = track_detections(image, detections)
        capture_count += 1
        image = cv2.putText(image, "%d"%capture_count, (150, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255), 2)
        cv2.imshow('source', image)
        if cv2.waitKey(10) == ord('q'):
            vid.release()
            vid = None
            break
    batch.clear()

    times.append(time.time() - started)
    if len(times) >= 10:
        fps = 1 / (sum(times) / (10*batch_size))
        times = times[1:]

cv2.destroyAllWindows()