from flask import Flask, Response, render_template_string, request, redirect, url_for
from picamera2 import Picamera2
import cv2
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common
import numpy as np

app = Flask(__name__)

# Load the BodyPix MobileNet model
interpreter = make_interpreter('bodypix_mobilenet_v1_075_512_512_16_quant_decoder_edgetpu.tflite')
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]  # Model's input height (512)
width = input_details[0]['shape'][2]   # Model's input width (512)

# Initialize Pi Camera
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
picam2.start()

# Global variables
blackout_human = False  # Control whether to blackout the human or background

@app.route('/toggle_blackout', methods=['POST'])
def toggle_blackout():
    global blackout_human
    blackout_human = not blackout_human  # Toggle the blackout mode
    return redirect(url_for('index'))

def gen_frames():
    global blackout_human
    while True:
        try:
            # Capture frame from the camera
            frame = picam2.capture_array()

            # Convert RGBA to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

            # Resize the frame for model input
            model_input_frame = cv2.resize(frame_rgb, (width, height))

            # Perform segmentation using the BodyPix model
            common.set_input(interpreter, model_input_frame)
            interpreter.invoke()
            segmentation_output = interpreter.get_tensor(output_details[0]['index'])[0]

            # Segmentation mask for body parts (24 body parts)
            person_mask = (segmentation_output == 15).astype(np.uint8)
            person_mask_resized = cv2.resize(person_mask, (640, 480))

            # Apply the mask based on the blackout_human flag
            if blackout_human:
                # Blackout the human (invert the mask)
                blackout_frame = frame_rgb * np.dstack([1 - person_mask_resized] * 3)
            else:
                # Blackout the background (normal behavior)
                blackout_frame = frame_rgb * np.dstack([person_mask_resized] * 3)

            # Convert back to RGBA for display
            blackout_frame_rgba = cv2.cvtColor(blackout_frame, cv2.COLOR_RGB2RGBA)

            # Encode frame for streaming
            ret, buffer = cv2.imencode('.jpg', blackout_frame_rgba)
            if not ret:
                continue  # Skip frame if encoding fails
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        except Exception as e:
            print(f"Error in gen_frames: {str(e)}")
            continue  # Skip frame on error

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    html_content = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>BodyPix MobileNet Pose Estimation and Segmentation</title>
    </head>
    <body>
        <h1>BodyPix MobileNet Pose Estimation and Segmentation</h1>
        <img src="/video_feed" width="640" height="480">
        <form action="/toggle_blackout" method="POST" style="display:inline-block">
            <button type="submit">Toggle Blackout Human/Background</button>
        </form>
    </body>
    </html>
    '''
    return render_template_string(html_content)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
