import argparse
import logging
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import os
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from uuid import getnode
import imutils
import cv2
import numpy
import requests

logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument("addr", default="http://localhost:8000", nargs="?",
                    help="The address of the server.")
parser.add_argument("--fps", type=int, help="Set camera's frames per second.")
parser.add_argument("--detect", default=1, dest="detection_delay", type=float,
                    help="Number of seconds between each detection.")
parser.add_argument("--send", default=0.4, dest="send_delay", type=float,
                    help="Number of seconds a face must visible to be sent.")
parser.add_argument("--size", default=0.1, dest="min_size", type=float,
                    help="Minimum face size as fraction of frame size.")
parser.add_argument("--size_max", default=1, dest="max_size", type=float,
                    help="Maximum face size as fraction of frame size.")
parser.add_argument("--headless", action="store_true", default=False,
                    help="Start the script without GUI.")
parser.add_argument("--video", default=None, type=str,
                    help="Read a video file instead of capturing from camera.")
args = parser.parse_args()

# constants
FAILURE_DELAY = 1            # delay after a failing tracker is deleted
KEY_PID = "pid"              # person id
KEY_BOX = "box"              # face bounding box
KEY_FACE = "face"            # biggest image for a face
KEY_SENT = "sent"            # true if a face is sent
KEY_TRACKER = "trk"          # tracker object
KEY_DETECTION = "detection"  # last face detection time
KEY_CREATION = "creation"    # tracker creation time
O_SIZE = 500                 # resize video to 500px (to speedup processing)

api_url = urllib.parse.urljoin(args.addr, "/v1/events/")
trackers = []
person_id = 0
node_id = getnode()
executor = ThreadPoolExecutor(max_workers=4)
file_path = os.path.dirname(__file__)
xml_path = os.path.join(file_path, "haarcascade_frontalface_alt_tree.xml")
# load the known faces and embeddings along with OpenCV's Haar
# cascade for face detection
logging.info("[INFO] loading face detectors..." + str(node_id))
faceCascade = cv2.CascadeClassifier(xml_path)
ffDetector = cv2.CascadeClassifier("/usr/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml")
pfDetector = cv2.CascadeClassifier("/usr/share/OpenCV/haarcascades/haarcascade_profileface.xml")
ubDetector = cv2.CascadeClassifier("/usr/share/OpenCV/haarcascades/haarcascade_upperbody.xml")
# Opencv pre-trained SVM with HOG people features
HOGdetector = cv2.HOGDescriptor()
HOGdetector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

#video = cv2.VideoCapture(0 if args.video is None else args.video)
video = WebcamVideoStream(src=0 if args.video is None else args.video).start()
#if args.fps is not None:
#    video.set(cv2.CAP_PROP_FPS, args.fps)
#fps = video.get(cv2.CAP_PROP_FPS)
# grab next frame
frame = video.read()
#video_w, video_h = int(video.get(3)), int(video.get(4))
(video_h, video_w) = frame.shape[:2]
min_box = (int(video_w * args.min_size), int(video_h * args.min_size))
max_box = (int(video_w * args.max_size), int(video_h * args.max_size))


def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    facesrects = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=min_box, maxSize=max_box)
        # detect front faces in the grayscale frame
    ffrects = ffDetector.detectMultiScale(gray, scaleFactor=1.05,
            minNeighbors=3, minSize=(20, 20),
            flags= cv2.CASCADE_SCALE_IMAGE)
        # OpenCV returns bounding box coordinates in (x, y, w, h) order
        # but we need them in (top, right, bottom, left) order, so we
        # need to do a bit of reordering
    """ 	ffboxes = [( int(x + w / 2), int(y + h / 2)) for (x, y, w, h) in ffrects]
    # loop over the recognized faces
    for (cx, cy) in ffboxes:
        # draw the predicted face name on the image
        cv2.drawMarker(rgb, (cx + Ux, cy + Uy), (5, 75, 10), markerType=cv2.MARKER_TRIANGLE_DOWN, thickness=2, line_type=cv2.LINE_AA)
    """
    # detect profile faces in the grayscale frame
    pfrects = pfDetector.detectMultiScale(gray, scaleFactor=1.05,
            minNeighbors=2, minSize=(20, 20),
            flags=cv2.CASCADE_SCALE_IMAGE)
        # OpenCV returns bounding box coordinates in (x, y, w, h) order
        # but we need them in (top, right, bottom, left) order, so we
        # need to do a bit of reordering
    """ 	pfboxes = [( int(x + w / 2), int(y + h / 2)) for (x, y, w, h) in pfrects]
    # loop over the recognized faces
    for (cx, cy) in pfboxes:
        # draw the predicted face name on the image
        cv2.drawMarker(rgb, (cx + Ux, cy+ Uy), (5, 125, 10), markerType=cv2.MARKER_TRIANGLE_UP, thickness=2, line_type=cv2.LINE_AA)
    """
    # detect upper bodyes in the grayscale frame
    ubrects = ubDetector.detectMultiScale(gray, scaleFactor=1.1,
            minNeighbors=2, minSize=(20, 20),
            flags=cv2.CASCADE_SCALE_IMAGE)
        # OpenCV returns bounding box coordinates in (x, y, w, h) order
        # but we need them in (top, right, bottom, left) order, so we
        # need to do a bit of reordering
    """ 	ubboxes = [( int(x + w / 2), int(y + h / 2), x + Ux, y + Uy , x + Ux + w, y + Uy + h) for (x, y, w, h) in ubrects]
    # loop over the recognized faces
    for (cx, cy, ux, uy, lx, ly) in ubboxes:
        # draw the predicted face name on the image
        cv2.drawMarker(rgb, (cx + Ux, cy + Uy), (5, 254, 10), markerType=cv2.MARKER_DIAMOND, thickness=2, line_type=cv2.LINE_AA)
        cv2.rectangle(rgb, (ux + 1, uy + 1), (lx , ly), (5, 255, 20), 1)
    """	#return

#def HOGdetect_person(gray, rgb, Ux = 0, Uy = 0):
    #if gray.shape[0] < 128 or gray.shape[1] < 64:
    #	roi = cv2.resize(gray, (int(gray.shape[1] * 8), int(gray.shape[0] * 8)))
    #else:
    #	roi = gray

    #logging.info("Faces detected = " + str(len(hboxes)))

    (rects, weights) = HOGdetector.detectMultiScale(gray, winStride=(4, 4), padding=(8, 8), scale=1.05, useMeanshiftGrouping = True)
                # OpenCV returns bounding box coordinates in (x, y, w, h) order
                # but we need them in (top, right, bottom, left) order, so we
                # need to do a bit of reordering
    """ 	boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]
    #names = weights

    # loop over the recognized faces
    for ((top, right, bottom, left), name) in zip(boxes, weights):
        # draw the predicted face name on the image
        r = top%255
        g = np.clip(200 * name, 0 , 255)
        b = left%255
        #cv2.rectangle(frame, (left+4, top+4), (right-4, bottom-4),
        #	np.array((r, g, b)), 2)
        cx = int((left+right)/2)
        cy = int((top+bottom)/2)
        cv2.drawMarker(rgb, (cx + Ux, cy + Uy),np.array((r, g, b)),markerType=cv2.MARKER_TILTED_CROSS, thickness=2, line_type=cv2.LINE_AA)
        y = top - 15 if top - 15 > 15 else top + 15
        y = cy - 15 if cy - 15 > 15 else cy + 15
        cv2.putText(rgb, str(name), (left + Ux, y + Uy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, np.array((r, g, b)), 1)
    """
    return rects


def track_unknown_faces(faces, frame):
    for (x, y, w, h) in faces:
        unknown = True
        for t in trackers[:]:  # check if the face is inside a tracker's box
            mid_x, mid_y = x + w / 2, y + h / 2
            t_x, t_y, t_w, t_h = t[KEY_BOX]
            if t_x <= mid_x <= t_x + w and t_y <= mid_y <= t_y + t_h:
                unknown = False
                break

        if unknown:
            global person_id
            new_tracker = cv2.Tracker_create("MEDIANFLOW")  # KCF MEDIANFLOW
            new_tracker.init(frame, (x, y, w, h))
            trackers.append({
                            KEY_TRACKER: new_tracker,
                            KEY_BOX: (x, y, w, h),
                            KEY_PID: person_id,
                            KEY_DETECTION: time.time(),
                            KEY_CREATION: time.time(),
                            KEY_FACE: numpy.copy(frame[y:y + h, x:x + w])
                            })
            person_id += 1


def update_trackers(frame):
    for t in trackers[:]:
        located, box = t[KEY_TRACKER].update(frame)
        x, y, w, h = (int(b) for b in box)

        if located:
            t[KEY_BOX] = (x, y, w, h)
            t[KEY_DETECTION] = time.time()
            is_bigger = w > t[KEY_FACE].shape[0] and h > t[KEY_FACE].shape[1]
            out_of_camera = x < 1 or y < 1 or x + w > video_w - 1 \
                or y + h > video_h - 1
            if is_bigger and not out_of_camera:
                fr_x = max(0, min(x, video_w))
                fr_y = max(0, min(y, video_h))
                to_x = min(x + w, video_w)
                to_y = min(y + h, video_h)
                t[KEY_FACE] = numpy.copy(frame[fr_y:to_y, fr_x:to_x])
        elif time.time() - t[KEY_DETECTION] >= FAILURE_DELAY:
            if t[KEY_DETECTION] - t[KEY_CREATION] > args.send_delay \
               and time.time() - t[KEY_CREATION] > args.send_delay:
                executor.submit(send_request, t[KEY_FACE])
            else:
                logging.debug("tracker {} ignored".format(t[KEY_PID]))
            trackers.remove(t)


def draw_rectangles(frame):
    for t in trackers:
        x, y, w, h = t[KEY_BOX]
        colour = (0, 255, 0) if KEY_SENT in t else (255, 0, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
        cv2.putText(frame, str(t[KEY_PID]), (x, y), cv2.FONT_HERSHEY_DUPLEX, 1,
                    colour)


def send_request(face):
    encoded = cv2.imencode(".jpg", face)[1]
    data = {"node_id": node_id}
    files = {"event_image": BytesIO(encoded.tostring())}
    try:
        r = requests.post(api_url, data=data, files=files, timeout=6.05)
        if r.ok:
            logging.info("Face sent. Status code=" + str(r.status_code))
        else:
            logging.error("Face not sent. Status code=" + str(r.status_code))
    except (requests.Timeout, requests.ConnectionError) as e:
        logging.error("Face not sent. Unexpected error." + str(e))


last_detection = time.time()
output_size = (O_SIZE, int(O_SIZE * video_h / video_w))
while True:
    #ret, frame = video.read()
    frame = video.read()
    #if not ret:
    #    break
    frame = imutils.resize(frame, width=O_SIZE)
    #update_trackers(frame)
    if time.time() - last_detection > args.detection_delay:
        faces = detect_faces(frame)
        #track_unknown_faces(faces, frame)
        last_detection = time.time()

    if not args.headless:
        draw_rectangles(frame)
        cv2.imshow("video", frame) #cv2.resize(frame, output_size))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
video.stop()
#video.release()
cv2.destroyAllWindows()
executor.shutdown(wait=True)
