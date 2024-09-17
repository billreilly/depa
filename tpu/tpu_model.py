from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common
import cv2

# Initialize Coral TPU with MoveNet model
interpreter = make_interpreter("movenet_single_pose_lightning_ptq_edgetpu.tflite")
interpreter.allocate_tensors()

def get_keypoints(frame, width, height):
    model_input_frame = cv2.resize(frame, (192, 192))
    common.set_input(interpreter, model_input_frame)
    interpreter.invoke()
    pose = common.output_tensor(interpreter, 0).copy()
    return pose.reshape(-1, 3)
    
