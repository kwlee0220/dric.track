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
tracker = BufferedTracker(detector, tracker, buffer_size=5)

times = []
fps = 0
vid = cv2.VideoCapture('./data/cam_6_short.mp4')
while True:
    _, img = vid.read()

    ts1 = time.time()
    tracked_img, tracks = tracker.track(img)
    if img is None and len(tracks) == 0:
        break
    times.append(time.time() - ts1)
    if len(times) >= 10:
        fps = 1 / (sum(times) / 10)
        times = times[1:]

    if len(tracks) > 0:
        confirmeds =  [t for t in tracks if t.detector_confidence > 0 and not t.tentative]
        tracked_img = draw_boxes(tracked_img, [t.tlbr for t in confirmeds], thickness=2)
        tracked_img = draw_track_labels(tracked_img, confirmeds, thickness=1)

        tentatives = [t for t in tracks if t.detector_confidence == 0 or t.tentative]
        tracked_img = draw_boxes(tracked_img, [t.tlbr for t in tentatives], color=(255,255,255), thickness=1)
#        tracked_img = draw_track_labels(tracked_img, tentatives)
        fps_msg = 'FPS: %.1f' % (fps)
        tracked_img = cv2.putText(tracked_img, fps_msg, (5, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                    (0,0,255), 2)

        cv2.imshow('source', tracked_img)
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()