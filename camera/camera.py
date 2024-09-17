from picamera2 import Picamera2
import cv2

# Initialize the Pi Camera
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
picam2.start()

def capture_frame():
    frame = picam2.capture_array()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
    return cv2.rotate(frame_rgb, cv2.ROTATE_90_CLOCKWISE)
