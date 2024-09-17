from flask import Flask, Response, render_template, redirect, url_for, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_admin import Admin
from flask_admin.contrib.sqla import ModelView
import cv2
from camera.camera import capture_frame  # Camera handling logic
from tpu.tpu_model import get_keypoints  # TPU model keypoints detection
from utils.draw import draw_keypoints, draw_lines_for_joints   # Utility for drawing keypoints
from utils.fall_detection import detect_fall  # Fall detection logic
from servo_control import *  # Servo control logic
from auth.auth import auth_bp, send_fall_notification            # Import the auth Blueprint
from models import db, User              # Import the database and User model from models.py
import time
import threading  # Import threading for parallel execution



# Initialize the Flask application
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Configure SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Register the auth Blueprint
app.register_blueprint(auth_bp, url_prefix='/auth')

# Initialize Flask-Admin
admin = Admin(app, name='Admin Panel', template_mode='bootstrap3')
admin.add_view(ModelView(User, db.session))

# Ensure the database is properly initialized with the Flask app
db.init_app(app)

# Ensure the database is created before the first request
@app.before_first_request
def create_tables():
    db.create_all()

# Global variables to track the servo state and movement time
servo_moved_clockwise = False
servo_moved_counterclockwise = False
last_movement_time = time.time()
auto_servo_movement = False  # Default to enabled

# Thread lock to ensure safe access to shared resources
servo_lock = threading.Lock()

# Function to check if a joint is near the center of the frame
def check_joint_at_frame_center(joint, frame_width, center_threshold=100):
    joint_x = joint[1]  # X-coordinate of the joint
    center_left = frame_width // 2 - center_threshold  # Left boundary of the center region
    center_right = frame_width // 2 + center_threshold  # Right boundary of the center region

    if joint_x < center_left:
        return 'left'
    elif joint_x > center_right:
        return 'right'
    else:
        return 'center'

# Threaded function to control servo movement
def control_servo(joint_position):
    with servo_lock:  # Acquire lock to ensure safe access to the servo
        if joint_position == 'left':
            move_servo_counterclockwise()  # Move the servo counterclockwise
            time.sleep(1)  # Move for 1 second
            stop_servo()  # Stop the PWM signal to disengage the servo
        elif joint_position == 'right':
            move_servo_clockwise()  # Move the servo clockwise
            time.sleep(1)  # Move for 1 second
            stop_servo()  # Stop the PWM signal to disengage the servo


# Global variables for smoothing the bounding box
prev_x_min, prev_y_min, prev_x_max, prev_y_max = None, None, None, None
smooth_factor = 0.6  # Adjust this value to control smoothing

def gen_frames():
    global servo_moved_clockwise, servo_moved_counterclockwise, last_movement_time, auto_servo_movement
    movement_delay = 5  # 5-second delay between movement checks

    while True:
        frame = capture_frame()  # Capture frame from the camera
        frame_width = frame.shape[1]  # Get the frame width
        frame_height = frame.shape[0]  # Get the frame height

        # Get keypoints from the TPU model
        keypoints = get_keypoints(frame, 640, 480)

        # Detect fall based on head and hip angles, also get the calculated angle
        angle, fall_detected = detect_fall(keypoints)

        # Get the current time
        current_time = time.time()

        # Calculate the remaining time for the countdown
        time_remaining = max(0, movement_delay - (current_time - last_movement_time))

        # Display the auto-servo movement status on the frame
        auto_status = "Auto Movement Enabled" if auto_servo_movement else "Auto Movement Disabled"
        cv2.putText(frame, auto_status, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Always display the calculated angle live on the frame
        if angle is not None:
            cv2.putText(frame, f"Angle: {int(angle)} degrees", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Only check for movement if auto-movement is enabled
        if auto_servo_movement and time_remaining == 0:
            joint_position = check_joint_at_frame_center(keypoints[0], frame_width)

            # Start the servo movement in a separate thread
            servo_thread = threading.Thread(target=control_servo, args=(joint_position,))
            servo_thread.start()

            # Update the last movement time to reset the countdown
            last_movement_time = current_time

        # Ensure keypoints are valid before drawing or using them
        if keypoints is not None and keypoints.any():
            # Draw keypoints on the frame
            draw_keypoints(frame, keypoints)

            # Get the bounding box based on the keypoints for censoring
            x_coords = [int(kp[1] * frame_width) for kp in keypoints if kp[2] > 0.5]  # Scale to frame dimensions
            y_coords = [int(kp[0] * frame_height) for kp in keypoints if kp[2] > 0.5]

            if x_coords and y_coords:
                # Get the top-left and bottom-right coordinates for the censoring box
                x_min = max(min(x_coords) - 50, 0)  # Adding padding, ensuring it doesn't go negative
                y_min = max(min(y_coords) - 50, 0)
                x_max = min(max(x_coords) + 50, frame_width)  # Adding padding, ensuring it doesn't go beyond frame width
                y_max = min(max(y_coords) + 150, frame_height)

                # Draw a large box to censor the person
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), -1)  # Red box as censoring overlay

        # Display the countdown on the frame
        cv2.putText(frame, f"Next check in: {int(time_remaining)}s", (50, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # If fall status is True, overlay a message on the frame
        if fall_detected:
            cv2.putText(frame, "Fall Detected!", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "No Fall", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Encode the frame for streaming
        output_frame = cv2.resize(frame, (480, 640))
        ret, buffer = cv2.imencode('.jpg', output_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Clean up servo resources at the end of the process
    cleanup_servo()


# Define available cameras (for now, just one option)
CAMERAS = [{'id': 0, 'name': 'Main Camera'}]

# Camera selection page
@app.route('/')
def index():
    if 'username' not in session:
        return redirect(url_for('auth.login'))  # Redirect to login if not authenticated
    return render_template('camera_selection.html', cameras=CAMERAS)

# Redirect to the camera feed once the camera is selected
@app.route('/select_camera/<int:camera_id>')
def select_camera(camera_id):
    session['camera_id'] = camera_id  # Save the selected camera in session
    return redirect(url_for('camera_view'))

# PTZ Control: Custom clockwise movement
@app.route('/custom_move_right', methods=['POST'])
def custom_move_right():
    custom_move_servo_clockwise()
    return jsonify({"message": "Custom clockwise movement executed."})

# PTZ Control: Custom counterclockwise movement
@app.route('/custom_move_left', methods=['POST'])
def custom_move_left():
    custom_move_servo_counterclockwise()
    return jsonify({"message": "Custom counterclockwise movement executed."})

# Route to display the camera feed
@app.route('/camera_view')
def camera_view():
    if 'camera_id' not in session:
        return redirect(url_for('index'))
    return render_template('index.html', camera_id=session['camera_id'])

# Video feed route that streams frames
@app.route('/video_feed/<int:camera_id>')
def video_feed(camera_id):
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_servo', methods=['POST'])
def toggle_servo():
    global auto_servo_movement
    auto_servo_movement = not auto_servo_movement  # Toggle the servo movement state
    status = "enabled" if auto_servo_movement else "disabled"
    return jsonify({"message": f"Auto servo movement {status}."})


@app.route('/logout')
def logout():
    session.clear()  # Clear all session data
    return redirect(url_for('auth.login'))  # Adjusted to point to 'auth.login'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
