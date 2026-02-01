import cv2
import time

# Import required functions from utils.py
from utils import (
    initialize_preprocessing,
    initialize_video,
    is_night,
    process_frame
)
from detection import detect_persons_yolo
from posture import *
from notification import send_fall_notification

def main():
    # Initialize CLAHE & gamma LUTs
    initialize_preprocessing()
    mp_drawing, mp_pose, pose = initialize_mediapipe()
    try:
        cap, fps, w, h = initialize_video(r"C:\Users\meena\Desktop\dock\downloads\test.mp4")
        # cap, fps, w, h = initialize_video(r"rtsp://admin:admin@192.168.1.210:1935")
    except RuntimeError as err:
        print(f"Video init error: {err}")
        return

    frame_id = 0
    last_center = None
    last_timestamp = None
    last_velocity = 0.0
    fall_detected = False
    fall_timestamp = None
    fall_notified = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        start = time.time()

        frame = cv2.flip(frame, 1)

        # Detect lighting condition
        night = is_night(frame)

        # Preprocess frame (resizes internally)
        processed_frame = process_frame(frame, frame_id, night)
        ph, pw, _ = processed_frame.shape

        # Run pose and detection on the same preprocessed frame to keep coordinates consistent
        rgb_proc = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_proc)
        # Detect persons
        boxes = detect_persons_yolo(processed_frame)
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # FPS calculation
        proc_fps = 1 / max(time.time() - start, 0.001)

        status = "No Person"
        angle_value = 0

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Get required landmarks
            shoulder = get_xy(
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
                pw, ph
            )
            hip = get_xy(
                landmarks[mp_pose.PoseLandmark.LEFT_HIP],
                pw, ph
            )
            knee = get_xy(
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE],
                pw, ph
            )

            # Calculate angles
            body_angle = calculate_angle(shoulder, hip, knee)
            torso_angle = torso_tilt_angle(shoulder, hip)
            angle_value = torso_angle

            # Compute velocity using hip movement between frames
            current_center = hip
            now = time.time()
            prev_center = last_center
            if prev_center is not None and last_timestamp is not None:
                dt = max(now - last_timestamp, 1e-3)
                dy = current_center[1] - prev_center[1]
                dx = current_center[0] - prev_center[0]
                dist = (dx * dx + dy * dy) ** 0.5
                last_velocity = dist / dt  # pixels per second
            else:
                dy = 0.0
                last_velocity = 0.0
            last_center = current_center
            last_timestamp = now

            # Posture classification
            if torso_angle < 20:
                status = "STANDING"
                color = (0, 255, 0)
            elif torso_angle < 60:
                status = "SITTING"
                color = (0, 255, 255)
            else:
                status = "FALL / LYING"
                color = (0, 0, 255)

            # Fall decision: lying with recent high-velocity movement downward
            if status == "FALL / LYING" and last_velocity > 150 and dy > 30:
                fall_detected = True
                if fall_timestamp is None:
                    fall_timestamp = now

                if not fall_notified:
                    try:
                        send_fall_notification(
                            title="Fall detected",
                            body=f"Fall detected at {time.strftime('%H:%M:%S')} with velocity {int(last_velocity)} px/s"
                        )
                        fall_notified = True
                    except Exception as notify_err:
                        print(f"Notification error: {notify_err}")

            # Draw landmarks
            mp_drawing.draw_landmarks(
                processed_frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            # Draw torso line
            cv2.line(processed_frame, shoulder, hip, (255, 0, 0), 3)

        # -----------------------------
        # Display Info
        # -----------------------------
        cv2.putText(processed_frame, f"Torso Angle: {int(angle_value)} deg",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)

        cv2.putText(processed_frame, f"Status: {status}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 255), 2)

        cv2.putText(processed_frame, f"Velocity: {int(last_velocity)} px/s",
                    (20, 120), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 0), 2)

        if fall_detected:
            elapsed = time.time() - fall_timestamp if fall_timestamp else 0
            cv2.putText(processed_frame, f"FALL DETECTED ({elapsed:.1f}s)",
                        (20, 160), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 0, 255), 3)
        end=time.time()
        print(f"Time taken for frame {frame_id}: {end-start:.3f} seconds")
        cv2.imshow("Fall Detection - MediaPipe", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Fall detection finished")


if __name__ == "__main__":
    main()
    
