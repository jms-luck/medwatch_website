import cv2
import time
import numpy as np


def initialize_preprocessing():
    """Initialize preprocessing parameters (CLAHE and gamma curves)"""
    global clahe, gamma_cache
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gamma_cache = {}

# -----------------------------
# Core Preprocessing Functions
# -----------------------------
def resize_keep_aspect(frame, target_width=640):
    """Resize frame while maintaining aspect ratio"""
    h, w = frame.shape[:2]
    scale = target_width / w
    return cv2.resize(frame, (target_width, int(h * scale)))


def is_night(frame):
    """Detect if frame is night-time based on brightness"""
    if frame is None or frame.size == 0:
        return False
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Downsample to reduce noise influence and speed up brightness check
    small = cv2.resize(gray, (0, 0), fx=0.25, fy=0.25)
    return small.mean() < 65


def get_image_quality_metrics(frame):
    """Calculate image quality metrics for adaptive preprocessing"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = gray.mean()
    contrast = gray.std()
    return brightness, contrast


def apply_gray_world_white_balance(frame):
    """Reduce color cast using a simple gray-world assumption"""
    b, g, r = cv2.split(frame.astype(np.float32))
    eps = 1e-3
    avg_b, avg_g, avg_r = b.mean(), g.mean(), r.mean()
    gray = (avg_b + avg_g + avg_r) / 3.0
    scale_b = gray / (avg_b + eps)
    scale_g = gray / (avg_g + eps)
    scale_r = gray / (avg_r + eps)
    balanced = cv2.merge((
        np.clip(b * scale_b, 0, 255),
        np.clip(g * scale_g, 0, 255),
        np.clip(r * scale_r, 0, 255)
    ))
    return balanced.astype(np.uint8)


def apply_clahe(frame):
    """Apply Contrast Limited Adaptive Histogram Equalization"""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def apply_bilateral_denoise(frame):
    """Apply bilateral filtering for edge-preserving denoising"""
    return cv2.bilateralFilter(frame, 9, 75, 75)


def apply_histogram_equalization(frame):
    """Apply histogram equalization for better contrast"""
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YCrCb2BGR)


def apply_gamma_correction(frame, gamma_lut):
    """Apply gamma correction using lookup table"""
    return cv2.LUT(frame, gamma_lut)


def apply_sharpening(frame, strength=0.6):
    """Apply unsharp mask for edge enhancement"""
    blurred = cv2.GaussianBlur(frame, (3, 3), 0)
    sharpened = cv2.addWeighted(frame, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def normalize_for_yolo(frame):
    """Normalize frame values for YOLO input (0-1 range)"""
    return frame.astype(np.float32) / 255.0


def get_adaptive_gamma_lut(brightness, night):
    """Build a gamma LUT that brightens dark frames and softens harsh highlights"""
    key = (int(brightness), night)
    if key in gamma_cache:
        return gamma_cache[key]

    target_brightness = 100 if night else 125
    gamma_value = np.clip(target_brightness / max(brightness, 1.0), 0.6, 1.6)
    exponent = 1.0 / gamma_value
    lut = np.array([(i / 255.0) ** exponent * 255 for i in range(256)]).astype("uint8")
    gamma_cache[key] = lut
    return lut


def preprocess_frame_for_yolo(frame, frame_id, night):
    """
    Complete preprocessing pipeline optimized for YOLO detection
    Includes: resizing, denoising, histogram equalization, contrast enhancement, and sharpening
    """
    # Step 1: Resize while maintaining aspect ratio
    frame = resize_keep_aspect(frame, target_width=640)
    
    # Step 2: Quick color balance to remove casts
    if frame_id % 3 == 0:
        frame = apply_gray_world_white_balance(frame)

    # Step 3: Edge-preserving denoising (stronger at night)
    if night:
        frame = apply_bilateral_denoise(frame)
    else:
        frame = cv2.GaussianBlur(frame, (3, 3), 0)

    # Step 4: Adaptive contrast enhancement for dark/flat frames
    brightness, contrast = get_image_quality_metrics(frame)
    if night or contrast < 40:
        frame = apply_clahe(frame)

    # Step 5: Gamma correction driven by measured brightness
    gamma_lut = get_adaptive_gamma_lut(brightness, night)
    frame = apply_gamma_correction(frame, gamma_lut)

    # Step 6: Sharpening to help detection (a bit stronger per request)
    sharpen_strength = 0.65 if night else 0.45
    frame = apply_sharpening(frame, strength=sharpen_strength)

    return frame

# ============================================



def create_video_writer(fps, w, h, output_path):
    return cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

def process_frame(frame, frame_id, night):
    """Legacy function for backward compatibility - uses new preprocessing pipeline"""
    return preprocess_frame_for_yolo(frame, frame_id, night)

def initialize_video(input_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError(f"Unable to read first frame from {input_path}")

    frame = resize_keep_aspect(frame)
    h, w = frame.shape[:2]
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return cap, fps, w, h


# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# gamma_day = np.array([(i / 255.0) ** (1 / 1.1) * 255 for i in range(256)]).astype("uint8")
# gamma_night = np.array([(i / 255.0) ** (1 / 1.4) * 255 for i in range(256)]).astype("uint8")
#     cap, fps, w, h = initialize_video(r"downloads/cam1.avi")
#     out = create_video_writer(fps, w, h, r"downloads/cam1_preprocessed_fast.mp4")
    
#     frame_id = 0
    
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         start = time.time()
#         night = is_night(frame)
#         frame = process_frame(frame, frame_id, night)
        
#         proc_fps = 1 / max(time.time() - start, 0.001)
#         cv2.putText(frame, f"FPS: {proc_fps:.1f}",
#                     (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
#                     0.7, (0, 0, 255), 2)
        
#         out.write(frame)
#         cv2.imshow("Fast Preprocessing", frame)
        
#         if cv2.waitKey(1) & 0xFF == 27:
#             break
        
#         frame_id += 1
    
#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()
#     print("✅ Fast preprocessed video saved")

# if __name__ == "__main__":
#     main()
