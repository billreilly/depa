import math

def detect_fall(keypoints, angle_threshold=20):
    """
    Detects if the person has fallen based on the angle between the head and hips.
    Returns both the calculated angle and the fall detection result.
    """
    head = keypoints[0]    # Head keypoint
    left_hip = keypoints[11]  # Left hip keypoint
    right_hip = keypoints[12]  # Right hip keypoint

    # Average the position of the left and right hips to get a single hip point
    hip = [(left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2]

    head_x, head_y = head[1], head[0]
    hip_x, hip_y = hip[1], hip[0]

    # Calculate the difference in x and y coordinates between the head and hips
    delta_x = head_x - hip_x
    delta_y = head_y - hip_y

    # Calculate the angle in degrees between the head and hips
    angle = math.degrees(math.atan2(delta_y, delta_x))

    # Adjust the angle so that 0 degrees is horizontal (falling) and 90 degrees is vertical (standing)
    angle = abs(angle)

    # Detect fall if the angle is close to horizontal (near 0 or 180 degrees)
    fall_detected = angle <= angle_threshold or angle >= 180 - angle_threshold

    return angle, fall_detected
