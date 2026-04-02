"""
test_webcam.py
──────────────
STEP 1 — Test your webcam before running the full system.

Run:  python test_webcam.py

You should see a window showing your webcam feed.
Press Q to quit.  If you see an error, change CAMERA_INDEX.
"""

import cv2

CAMERA_INDEX = 0   # Try 0, 1, 2 if one doesn't work

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print(f"[ERROR] Could not open camera {CAMERA_INDEX}")
    print("  → Try changing CAMERA_INDEX to 1 or 2")
    exit(1)

print(f"[OK] Camera {CAMERA_INDEX} opened successfully. Press Q to quit.")

while True:
    ok, frame = cap.read()
    if not ok:
        print("[WARN] Failed to read frame.")
        continue

    cv2.putText(frame, "Webcam OK — Press Q to quit",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Webcam Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[OK] Webcam test complete.")
