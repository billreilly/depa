from flask import Flask, Response
from picamera2 import Picamera2
import cv2
import numpy as np
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common

app = Flask(__name__)

# Initialize Coral TPU with the MoveNet model
interpreter = make_interpreter("movenet_single_pose_lightning_ptq_edgetpu.tflite")
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

# Initialize Pi Camera
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
picam2.start()

def draw_keypoints(frame, keypoints, confidence_threshold=0.3):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
    
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)

def gen_frames():
    while True:
        # Capture frame from the camera
        frame = picam2.capture_array()
        
        # Convert RGBA to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        
        # Rotate the frame 90 degrees clockwise
        rotated_frame = cv2.rotate(frame_rgb, cv2.ROTATE_90_CLOCKWISE)
        
        # Resize the rotated frame to model input size (192x192)
        model_input_frame = cv2.resize(rotated_frame, (192, 192))

        # Perform inference
        common.set_input(interpreter, model_input_frame)
        interpreter.invoke()
        pose = common.output_tensor(interpreter, 0).copy()

        # Reshape the pose data
        keypoints = pose.reshape(-1, 3)

        # Draw keypoints on the frame
        draw_keypoints(model_input_frame, keypoints)

        # Resize back to original dimensions
        output_frame = cv2.resize(model_input_frame, (480, 640))

        # Convert back to RGBA for display
        output_frame_rgba = cv2.cvtColor(output_frame, cv2.COLOR_RGB2RGBA)

        # Encode frame for streaming
        ret, buffer = cv2.imencode('.jpg', output_frame_rgba)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return "<h1>Raspberry Pi Camera MoveNet Stream (Rotated 90°)</h1><img src='/video_feed'>"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)