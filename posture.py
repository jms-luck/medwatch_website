import cv2
import mediapipe as mp
import math

# -----------------------------
# MediaPipe Initialization
# -----------------------------
def initialize_mediapipe():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return mp_drawing, mp_pose, pose

# -----------------------------
# Angle Calculation Functions
# -----------------------------
def calculate_angle(a, b, c):
    """
    Calculate angle between three points (A-B-C)
    """
    ax, ay = a
    bx, by = b
    cx, cy = c

    ba = (ax - bx, ay - by)
    bc = (cx - bx, cy - by)

    dot_product = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.sqrt(ba[0]**2 + ba[1]**2)
    mag_bc = math.sqrt(bc[0]**2 + bc[1]**2)

    if mag_ba * mag_bc == 0:
        return 0

    angle = math.acos(dot_product / (mag_ba * mag_bc))
    return math.degrees(angle)

def torso_tilt_angle(shoulder, hip):
    """
    Angle of torso with respect to vertical axis
    """
    dx = shoulder[0] - hip[0]
    dy = hip[1] - shoulder[1]  # vertical reference
    angle = math.degrees(math.atan2(abs(dx), abs(dy)))
    return angle

def get_xy(landmark, w, h):
    return (int(landmark.x * w), int(landmark.y * h))

# -----------------------------
# Video Capture
# -----------------------------