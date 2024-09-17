from flask import Flask, Response, render_template_string, request, redirect, url_for
from picamera2 import Picamera2
import cv2
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common
import numpy as np

app = Flask(__name__)

# Initialize Coral TPU with the DeepLab model
interpreter = make_interpreter('deeplabv3_mnv2_pascal_quant_edgetpu.tflite')
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

# Initialize Pi Camera
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
picam2.start()

# Global variables
rotation_angle = 0  # Track the rotation angle
blackout_human = False  # Control whether to blackout the human or the background

@app.route('/rotate', methods=['POST'])
def toggle_rotation():
    global rotation_angle
    # Increment the rotation angle by 90 degrees with each click
    rotation_angle = (rotation_angle + 90) % 360  # Ensure it stays within 0-360 degrees
    print(f"Rotation angle set to: {rotation_angle}")  # Debug output for server logs
    return redirect(url_for('index'))  # Redirect back to the index page

@app.route('/toggle_blackout', methods=['POST'])
def toggle_blackout():
    global blackout_human
    blackout_human = not blackout_human  # Toggle the blackout mode
    print(f"Blackout human mode set to: {blackout_human}")  # Debug output for server logs
    return redirect(url_for('index'))

def gen_frames():
    global rotation_angle, blackout_human
    while True:
        try:
            # Capture frame from the camera
            frame = picam2.capture_array()

            # Convert RGBA to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

            # Apply the current rotation angle
            if rotation_angle == 90:
                frame_rgb = cv2.rotate(frame_rgb, cv2.ROTATE_90_CLOCKWISE)
            elif rotation_angle == 180:
                frame_rgb = cv2.rotate(frame_rgb, cv2.ROTATE_180)
            elif rotation_angle == 270:
                frame_rgb = cv2.rotate(frame_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Resize the frame back to original dimensions
            resized_frame = cv2.resize(frame_rgb, (640, 480))
            
            # Resize for the model input
            model_input_frame = cv2.resize(resized_frame, (width, height))

            # Perform segmentation
            common.set_input(interpreter, model_input_frame)
            interpreter.invoke()
            segmentation_mask = interpreter.get_tensor(output_details[0]['index'])

            # The model outputs a 21-class segmentation. Class 15 is 'person'.
            person_mask = (segmentation_mask == 15).astype(np.uint8)
            person_mask_resized = cv2.resize(person_mask[0], (640, 480))

            # Apply the mask based on the blackout_human flag
            if blackout_human:
                # Blackout the human (invert the mask)
                blackout_frame = resized_frame * np.dstack([1 - person_mask_resized] * 3)
            else:
                # Blackout the background (normal behavior)
                blackout_frame = resized_frame * np.dstack([person_mask_resized] * 3)

            # Convert back to RGBA for display
            blackout_frame_rgba = cv2.cvtColor(blackout_frame, cv2.COLOR_RGB2RGBA)

            # Encode frame for streaming
            ret, buffer = cv2.imencode('.jpg', blackout_frame_rgba)

            if not ret:
                print("Failed to encode frame")
                continue  # Skip this frame if encoding fails

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
        <title>Raspberry Pi Camera DeepLab Segmentation Stream</title>
    </head>
    <body>
        <h1>Raspberry Pi Camera DeepLab Segmentation Stream</h1>
        <img src="/video_feed" width="640" height="480">
        <form action="/rotate" method="POST" style="display:inline-block">
            <button type="submit">Rotate Frame</button>
        </form>
        <form action="/toggle_blackout" method="POST" style="display:inline-block">
            <button type="submit">Toggle Blackout Human/Background</button>
        </form>
    </body>
    </html>
    '''
    return render_template_string(html_content)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
