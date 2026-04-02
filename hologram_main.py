"""
========================================================
AI-Based Real-Time Holographic Communication System
========================================================
Author: Student Project
Description:
    Captures live webcam video, detects 2D body pose using
    MediaPipe, lifts it to 3D, and renders a hologram-style
    3D visualization in real time.

Run: python hologram_main.py
"""

import cv2
import numpy as np
import mediapipe as mp
import matplotlib
matplotlib.use('TkAgg')          # Use TkAgg backend for live updating
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import threading
import time


# ─────────────────────────────────────────────────────────
# 1.  CONFIGURATION
# ─────────────────────────────────────────────────────────
CAMERA_INDEX   = 0      # Change to 1 if your external webcam is index 1
FRAME_WIDTH    = 640
FRAME_HEIGHT   = 480
SMOOTH_FACTOR  = 0.4    # 0 = no smoothing, 1 = fully frozen (keep 0.3-0.5)
HOLO_COLOR     = '#00FFFF'  # Cyan hologram tint
BG_COLOR       = '#050A14'  # Deep-space background


# ─────────────────────────────────────────────────────────
# 2.  MEDIAPIPE SETUP
#     MediaPipe Pose gives us 33 landmarks in (x, y, z)
#     coordinates.  x & y are normalized [0,1]; z is a
#     relative depth hint (not true metric depth).
# ─────────────────────────────────────────────────────────
mp_pose     = mp.solutions.pose
mp_drawing  = mp.solutions.drawing_utils
mp_styles   = mp.solutions.drawing_styles

POSE_DETECTOR = mp_pose.Pose(
    static_image_mode=False,        # Video stream → False
    model_complexity=1,             # 0=lite, 1=full, 2=heavy
    smooth_landmarks=True,          # Built-in temporal smoothing
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ─────────────────────────────────────────────────────────
# 3.  SKELETON CONNECTIONS
#     Each tuple = (landmark_A_index, landmark_B_index)
#     We colour groups differently for visual clarity.
# ─────────────────────────────────────────────────────────
SKELETON_GROUPS = {
    'torso':      [(11,12),(11,23),(12,24),(23,24)],
    'left_arm':   [(11,13),(13,15),(15,17),(15,19),(15,21)],
    'right_arm':  [(12,14),(14,16),(16,18),(16,20),(16,22)],
    'left_leg':   [(23,25),(25,27),(27,29),(27,31)],
    'right_leg':  [(24,26),(26,28),(28,30),(28,32)],
    'face':       [(0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8)],
}

GROUP_COLORS = {
    'torso':     '#00FFFF',
    'left_arm':  '#00BFFF',
    'right_arm': '#00BFFF',
    'left_leg':  '#00E5FF',
    'right_leg': '#00E5FF',
    'face':      '#7FFFFF',
}


# ─────────────────────────────────────────────────────────
# 4.  POSE PROCESSOR CLASS
# ─────────────────────────────────────────────────────────
class PoseProcessor:
    """
    Wraps MediaPipe detection and converts raw landmarks
    into smoothed (x, y, z) NumPy arrays.
    """

    def __init__(self, smooth=SMOOTH_FACTOR):
        self.smooth   = smooth
        self._prev    = None   # Previous frame's 3D landmarks

    def process(self, bgr_frame):
        """
        Args:
            bgr_frame: OpenCV BGR image
        Returns:
            annotated_frame : BGR with 2D skeleton overlay
            landmarks_3d    : np.ndarray shape (33, 3) or None
            raw_results     : MediaPipe result object
        """
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = POSE_DETECTOR.process(rgb)
        rgb.flags.writeable = True

        annotated = bgr_frame.copy()

        if results.pose_landmarks:
            # Draw 2-D skeleton on the camera feed
            mp_drawing.draw_landmarks(
                annotated,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 255, 255), thickness=2, circle_radius=3),
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 200, 200), thickness=2),
            )

            # Extract 3D landmarks
            lm = results.pose_landmarks.landmark
            xyz = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)

            # Temporal smoothing (exponential moving average)
            if self._prev is not None:
                xyz = self.smooth * self._prev + (1 - self.smooth) * xyz
            self._prev = xyz.copy()

            return annotated, xyz, results

        # No person detected
        self._prev = None
        return annotated, None, results


# ─────────────────────────────────────────────────────────
# 5.  3-D HOLOGRAM RENDERER
# ─────────────────────────────────────────────────────────
class HologramRenderer:
    """
    Maintains a Matplotlib 3-D figure and updates it in
    real time with the latest pose data.
    """

    def __init__(self):
        plt.ion()   # Interactive mode for live updating
        self.fig = plt.figure(figsize=(6, 8), facecolor=BG_COLOR)
        self.ax  = self.fig.add_subplot(111, projection='3d',
                                         facecolor=BG_COLOR)
        self._style_axes()
        self._artists = []  # Track drawn artists for clearing

    # ── Internal helpers ──────────────────────────────────

    def _style_axes(self):
        ax = self.ax
        ax.set_xlim(0, 1);  ax.set_xlabel('X', color='#00FFFF', fontsize=8)
        ax.set_ylim(0, 1);  ax.set_ylabel('Y', color='#00FFFF', fontsize=8)
        ax.set_zlim(-0.5, 0.5); ax.set_zlabel('Z', color='#00FFFF', fontsize=8)
        ax.tick_params(colors='#00FFFF', labelsize=7)
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor('#003333')
        ax.grid(True, color='#003333', alpha=0.5)
        self.fig.suptitle('HOLOGRAPHIC COMM SYSTEM',
                           color='#00FFFF', fontsize=11,
                           fontfamily='monospace', y=0.97)

    def _clear(self):
        for artist in self._artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._artists.clear()

    # ── Public interface ──────────────────────────────────

    def update(self, landmarks_3d):
        """
        Redraw the 3D hologram.

        Args:
            landmarks_3d: np.ndarray (33, 3) with (x, y, z)
                          values in [0,1] from MediaPipe.
                          We flip y so the figure is upright.
        """
        self._clear()

        if landmarks_3d is None:
            txt = self.ax.text(0.5, 0.5, 0.0,
                               'NO SIGNAL\nSTEP INTO VIEW',
                               color='#00FFFF', ha='center',
                               fontsize=14, fontfamily='monospace',
                               alpha=0.7)
            self._artists.append(txt)
        else:
            x =  landmarks_3d[:, 0]
            y =  1 - landmarks_3d[:, 1]   # Flip vertical (MediaPipe y=0 is top)
            z =  landmarks_3d[:, 2]

            # Draw skeleton bones
            for group, connections in SKELETON_GROUPS.items():
                color = GROUP_COLORS[group]
                for (a, b) in connections:
                    line, = self.ax.plot(
                        [x[a], x[b]], [y[a], y[b]], [z[a], z[b]],
                        color=color, linewidth=1.5, alpha=0.85
                    )
                    self._artists.append(line)

            # Draw joint spheres
            scatter = self.ax.scatter(x, y, z,
                                      c=HOLO_COLOR, s=18, alpha=0.9,
                                      depthshade=True, edgecolors='none')
            self._artists.append(scatter)

            # Scanline effect: horizontal planes
            for level in np.linspace(0.1, 0.9, 5):
                plane = self.ax.plot(
                    [0.1, 0.9], [level, level], [0, 0],
                    color='#003344', linewidth=0.5, alpha=0.3
                )
                self._artists.extend(plane)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


# ─────────────────────────────────────────────────────────
# 6.  MAIN LOOP
# ─────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  AI HOLOGRAPHIC COMMUNICATION SYSTEM  ")
    print("=" * 55)
    print(f"  Camera index : {CAMERA_INDEX}")
    print("  Press  Q  in the camera window to quit")
    print("=" * 55)

    # Open webcam
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {CAMERA_INDEX}. "
              "Try changing CAMERA_INDEX in the script.")
        return

    processor = PoseProcessor()
    renderer  = HologramRenderer()

    frame_count = 0
    t_start     = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Dropped frame.")
            continue

        frame_count += 1

        # ── 2D detection + landmark extraction ──
        annotated, landmarks_3d, _ = processor.process(frame)

        # ── FPS overlay on camera feed ──
        elapsed = time.time() - t_start
        fps = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        label = "POSE DETECTED" if landmarks_3d is not None else "NO POSE"
        cv2.putText(annotated, label, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # ── Show camera window ──
        cv2.imshow("Camera Feed  |  Press Q to quit", annotated)

        # ── Update 3D hologram (every frame) ──
        renderer.update(landmarks_3d)

        # ── Quit on Q ──
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Quit signal received.")
            break

    cap.release()
    cv2.destroyAllWindows()
    plt.close('all')
    print("[INFO] Session ended.")


if __name__ == "__main__":
    main()
