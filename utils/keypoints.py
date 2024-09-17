def get_keypoints_positions(keypoints):
    """
    Extracts and returns head, left hip, and right hip keypoints from the full keypoints array.
    """
    head = keypoints[0]    # Head keypoint
    left_hip = keypoints[11]  # Left hip keypoint
    right_hip = keypoints[12]  # Right hip keypoint
    return head, left_hip, right_hip

def is_confidence_sufficient(keypoint, threshold=0.5):
    """
    Checks if the keypoint's confidence score is above the threshold.
    """
    return keypoint[2] >= threshold
