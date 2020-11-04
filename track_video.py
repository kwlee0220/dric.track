from absl import flags
import sys
FLAGS = flags.FLAGS
FLAGS(sys.argv)

import cv2
from dric.track import YoloV3Detector, DeepSortTracker, BufferedTracker, draw_boxes, draw_track_labels
from deep_sort.track import Track

detector = YoloV3Detector(filter=['car','truck'])
tracker = DeepSortTracker(detector, max_age=10, n_init=5)
tracker = BufferedTracker(tracker, buffer_size=5)

times = []
vid = cv2.VideoCapture('./data/cam_11.mp4')
while True:
    _, img = vid.read()

    tracked_img, tracks = tracker.track(img)
    if img is None and len(tracks) == 0:
        break

    if tracks is not None and tracked_img is not None:
        confirmeds =  [t for t in tracks if t.detector_confidence > 0 and not t.tentative]
        tracked_img = draw_boxes(tracked_img, [t.tlbr for t in confirmeds], thickness=1)
        tracked_img = draw_track_labels(tracked_img, confirmeds, thickness=1)

        tentatives = [t for t in tracks if t.detector_confidence == 0 or t.tentative]
        tracked_img = draw_boxes(tracked_img, [t.tlbr for t in tentatives], color=(255,255,255), thickness=1)
#        tracked_img = draw_track_labels(tracked_img, tentatives)

        cv2.imshow('source', tracked_img)
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()