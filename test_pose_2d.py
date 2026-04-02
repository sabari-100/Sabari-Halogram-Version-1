"""
test_pose_2d.py
───────────────
STEP 2 — Test 2-D pose detection on your webcam.

Run:  python test_pose_2d.py

You should see a green skeleton overlaid on your body.
Press Q to quit.
"""

import cv2
import mediapipe as mp

mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

CAMERA_INDEX = 0   # Change if needed

cap  = cv2.VideoCapture(CAMERA_INDEX)
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

print("[INFO] 2-D Pose Detection started. Press Q to quit.")
print("[INFO] Stand ~1-2 m from your webcam for best results.")

while True:
    ok, frame = cap.read()
    if not ok:
        continue

    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=(0, 255, 0), thickness=2, circle_radius=4),
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(0, 200, 0), thickness=2),
        )
        status = "POSE DETECTED"
        color  = (0, 255, 0)

        # Print first 5 landmark positions as sanity check
        lm = results.pose_landmarks.landmark
        info = f"Nose: ({lm[0].x:.2f}, {lm[0].y:.2f}, {lm[0].z:.2f})"
        cv2.putText(frame, info, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
    else:
        status = "NO POSE — STEP CLOSER"
        color  = (0, 0, 255)

    cv2.putText(frame, status, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("2-D Pose Detection Test | Q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pose.close()
print("[OK] 2-D Pose test complete.")
