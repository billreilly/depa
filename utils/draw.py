import cv2
import numpy as np

def draw_keypoints(frame, keypoints, confidence_threshold=0.3):
    y, x, c = frame.shape
    shaped = keypoints * [y, x, 1]  # Scale keypoints to match the frame size
    
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)


def draw_lines_for_joints(frame, keypoints):
    """
    Draws lines between the head and hips on the frame for better visualization.
    """
    head = keypoints[0]    # Head keypoint
    left_hip = keypoints[11]  # Left hip keypoint
    right_hip = keypoints[12]  # Right hip keypoint

    # Average the position of the left and right hips to get a single hip point
    hip = [(left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2]

    # Draw a line between the head and the averaged hip point
    if head[2] > 0.5 and left_hip[2] > 0.5 and right_hip[2] > 0.5:  # Ensure confidence is high enough
        cv2.line(frame, (int(head[1]), int(head[0])), (int(hip[1]), int(hip[0])), (0, 255, 0), 2)

    # Optionally, draw lines between the two hips as well
    cv2.line(frame, (int(left_hip[1]), int(left_hip[0])), (int(right_hip[1]), int(right_hip[0])), (255, 0, 0), 2)

    return frame