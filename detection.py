import cv2
from ultralytics import YOLO

# =========================
# YOLO INIT
# =========================
YOLO_MODEL_PATH = "yolov8l.pt"
model = YOLO(YOLO_MODEL_PATH)

# =========================
# UTILS
# =========================
def resize_keep_aspect(frame, target_width=320):
    h, w = frame.shape[:2]
    scale = target_width / w
    resized = cv2.resize(frame, (target_width, int(h * scale)))
    return resized, scale

# =========================
# YOLO PERSON DETECTION
# =========================
def detect_persons_yolo(frame):
    """
    Returns list of bounding boxes [x1, y1, x2, y2] in original frame scale
    """
    resized, scale = resize_keep_aspect(frame)

    results = model.predict(
        resized,
        classes=[0],      # person class
        conf=0.25,
        imgsz=320,
        verbose=False
    )

    boxes = []
    for r in results:
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
            boxes.append([
                x1 / scale,
                y1 / scale,
                x2 / scale,
                y2 / scale
            ])
    return boxes
