from flask import Flask, Response, render_template_string, request, jsonify
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
        frame = picam2.capture_array()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        rotated_frame = cv2.rotate(frame_rgb, cv2.ROTATE_90_CLOCKWISE)
        model_input_frame = cv2.resize(rotated_frame, (192, 192))

        common.set_input(interpreter, model_input_frame)
        interpreter.invoke()
        pose = common.output_tensor(interpreter, 0).copy()

        keypoints = pose.reshape(-1, 3)
        draw_keypoints(model_input_frame, keypoints)

        output_frame = cv2.resize(model_input_frame, (480, 640))
        output_frame_rgba = cv2.cvtColor(output_frame, cv2.COLOR_RGB2RGBA)

        ret, buffer = cv2.imencode('.jpg', output_frame_rgba)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/update_status', methods=['POST'])
def update_status():
    new_status = request.json.get('status')
    return jsonify({"status": "success", "message": new_status})

@app.route('/')
def index():
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
        <title>AI Camera Stream</title>
        <link rel="manifest" href="/manifest.json">
        <meta name="apple-mobile-web-app-capable" content="yes">
        <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
        <meta name="apple-mobile-web-app-title" content="AI Camera">
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                background: linear-gradient(135deg, #e6f3e6, #c1e6c1);
                color: #333333;
                margin: 0;
                padding: 0;
                min-height: 100vh;
                display: flex;
                flex-direction: column;
            }
            .container {
                flex-grow: 1;
                display: flex;
                flex-direction: column;
                max-width: 100%;
                margin: 0 auto;
                padding: 20px;
                box-sizing: border-box;
            }
            h1 {
                color: #2e7d32;
                text-align: center;
                margin-bottom: 20px;
                text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
            }
            #video-container {
                background-color: #ffffff;
                padding: 10px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin-bottom: 20px;
                flex-grow: 1;
                display: flex;
                justify-content: center;
                align-items: center;
            }
            #video-container img {
                max-width: 100%;
                max-height: 100%;
                object-fit: contain;
            }
            #controls {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 10px;
                width: 180px;
                margin: 0 auto 20px;
            }
            .ptz-btn {
                width: 50px;
                height: 50px;
                font-size: 20px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                display: flex;
                align-items: center;
                justify-content: center;
                user-select: none;
                -webkit-user-select: none;
                -moz-user-select: none;
                -ms-user-select: none;
                -webkit-tap-highlight-color: transparent;
            }
            .ptz-btn:hover, .ptz-btn:active, .ptz-btn:focus {
                outline: none;
                background-color: #4CAF50;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            }
            .ptz-btn:active {
                transform: none;
            }
            #up { grid-column: 2; }
            #left { grid-column: 1; grid-row: 2; }
            #right { grid-column: 3; grid-row: 2; }
            #down { grid-column: 2; grid-row: 3; }
            #zoom-in { grid-column: 1; grid-row: 3; }
            #zoom-out { grid-column: 3; grid-row: 3; }
            #status {
                padding: 15px;
                background-color: #ffffff;
                border-radius: 5px;
                text-align: center;
                font-weight: bold;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>AI Camera Stream</h1>
            <div id="video-container">
                <img src="{{ url_for('video_feed') }}">
            </div>
            <div id="controls">
                <button class="ptz-btn" id="up">↑</button>
                <button class="ptz-btn" id="left">←</button>
                <button class="ptz-btn" id="right">→</button>
                <button class="ptz-btn" id="down">↓</button>
                <button class="ptz-btn" id="zoom-in">+</button>
                <button class="ptz-btn" id="zoom-out">-</button>
            </div>
            <div id="status">Camera Status: standing</div>
        </div>
        <script>
            function updateStatus(status) {
                fetch('/update_status', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({status: status}),
                })
                .then(response => response.json())
                .then(data => {
                    document.getElementById('status').textContent = 'Camera Status: ' + status;
                    if (status === "fall detected") {
                        if ("Notification" in window && Notification.permission === "granted") {
                            new Notification("AI Camera Alert", { body: "Fall detected!" });
                        }
                    }
                })
                .catch((error) => {
                    console.error('Error:', error);
                });
            }

            document.addEventListener('DOMContentLoaded', (event) => {
                if ("Notification" in window) {
                    Notification.requestPermission();
                }
                
                document.querySelectorAll('.ptz-btn').forEach(button => {
                    button.addEventListener('click', () => {
                        updateStatus('fall detected');
                    });
                });
            });
        </script>
    </body>
    </html>
    ''')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)