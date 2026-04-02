"""
hologram_advanced.py
─────────────────────
ADVANCED VERSION — adds:
  • Joint angle readouts on the 3-D figure
  • Gesture detection (arms raised = 'HANDS UP')
  • Optional video recording of the camera feed
  • Rotation animation of the 3-D hologram

Run:  python hologram_advanced.py
      python hologram_advanced.py --record     (saves output.avi)
      python hologram_advanced.py --rotate      (spin the 3D view)
"""

import sys
import cv2
import numpy as np
import mediapipe as mp
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

from pose_3d import reconstruct, compute_joint_angles   # Our helper module


# ─────────────────────────────────────────────────────────
# FLAGS (can also be set via command-line args)
# ─────────────────────────────────────────────────────────
RECORD_VIDEO = '--record' in sys.argv
ROTATE_VIEW  = '--rotate' in sys.argv
CAMERA_INDEX = 0
HOLO_COLOR   = '#00FFFF'
BG_COLOR     = '#050A14'


# ─────────────────────────────────────────────────────────
# GESTURE DETECTION
# ─────────────────────────────────────────────────────────
def detect_gesture(xyz):
    """
    Simple rule-based gesture recognition using joint positions.

    Returns: string label or None
    """
    if xyz is None:
        return None

    L_WRIST   = 15;  R_WRIST  = 16
    L_SHOULDER= 11;  R_SHOULDER=12
    L_HIP     = 23;  R_HIP    = 24

    # Hands raised: both wrists above both shoulders in image coords
    # (y=0 is top, so smaller y = higher on screen)
    l_raised = xyz[L_WRIST, 1] < xyz[L_SHOULDER, 1] - 0.05
    r_raised = xyz[R_WRIST, 1] < xyz[R_SHOULDER, 1] - 0.05

    if l_raised and r_raised:
        return "HANDS UP"

    # T-pose: wrists roughly at shoulder height, arms wide
    l_side = abs(xyz[L_WRIST, 1] - xyz[L_SHOULDER, 1]) < 0.08
    r_side = abs(xyz[R_WRIST, 1] - xyz[R_SHOULDER, 1]) < 0.08
    if l_side and r_side:
        return "T-POSE"

    return None


# ─────────────────────────────────────────────────────────
# ADVANCED HOLOGRAM RENDERER
# ─────────────────────────────────────────────────────────
class AdvancedRenderer:

    SKELETON = {
        'torso':     [(11,12),(11,23),(12,24),(23,24)],
        'arms':      [(11,13),(13,15),(12,14),(14,16)],
        'legs':      [(23,25),(25,27),(24,26),(26,28)],
    }
    COLORS = {
        'torso': '#00FFFF',
        'arms':  '#00BFFF',
        'legs':  '#00E5FF',
    }

    def __init__(self):
        plt.ion()
        self.fig = plt.figure(figsize=(6, 8), facecolor=BG_COLOR)
        self.ax  = self.fig.add_subplot(111, projection='3d',
                                         facecolor=BG_COLOR)
        self._angle = 30   # Current azimuth for rotation
        self._setup_axes()
        self._artists = []

    def _setup_axes(self):
        ax = self.ax
        ax.set_xlim(-0.5, 0.5); ax.set_ylim(-0.7, 0.7); ax.set_zlim(-0.5, 0.5)
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.pane.fill = False
            axis.pane.set_edgecolor('#002233')
        ax.grid(True, color='#002233', alpha=0.4)
        ax.set_xlabel('X', color='#00FFFF', fontsize=7)
        ax.set_ylabel('Y', color='#00FFFF', fontsize=7)
        ax.set_zlabel('Z', color='#00FFFF', fontsize=7)
        ax.tick_params(colors='#005566', labelsize=6)
        self.fig.suptitle('◈  HOLOGRAPHIC LINK  ◈',
                           color='#00FFFF', fontsize=12,
                           fontfamily='monospace')

    def _clear(self):
        for a in self._artists:
            try: a.remove()
            except: pass
        self._artists.clear()

    def update(self, xyz, gesture=None, angles=None):
        self._clear()

        # Slowly rotate if flag is set
        if ROTATE_VIEW:
            self._angle = (self._angle + 0.8) % 360
            self.ax.view_init(elev=15, azim=self._angle)

        if xyz is None:
            t = self.ax.text(0, 0, 0, 'NO SIGNAL',
                             color='#00FFFF', fontsize=16,
                             ha='center', fontfamily='monospace')
            self._artists.append(t)
        else:
            x =  xyz[:, 0]
            y = -xyz[:, 2]   # Use z as depth on Y axis for better 3D
            z = -xyz[:, 1]   # Inverted y → vertical axis

            # Draw bones
            for group, conns in self.SKELETON.items():
                c = self.COLORS[group]
                for a, b in conns:
                    ln, = self.ax.plot([x[a],x[b]], [y[a],y[b]], [z[a],z[b]],
                                       color=c, lw=2, alpha=0.9)
                    self._artists.append(ln)

            # Joints
            sc = self.ax.scatter(x, y, z, c=HOLO_COLOR, s=22,
                                 alpha=0.95, depthshade=True, edgecolors='none')
            self._artists.append(sc)

            # Gesture label
            if gesture:
                gt = self.ax.text(x[0], y[0], z[0]+0.15,
                                  f"◉ {gesture}",
                                  color='#FFFF00', fontsize=10,
                                  fontfamily='monospace', ha='center')
                self._artists.append(gt)

            # Show a couple of key angles
            if angles:
                info = (f"L-Elbow:{angles['left_elbow']:.0f}°  "
                        f"R-Elbow:{angles['right_elbow']:.0f}°")
                self.fig.text(0.5, 0.03, info, ha='center',
                              color='#00AAAA', fontsize=8,
                              fontfamily='monospace')

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  AI HOLOGRAPHIC SYSTEM  — ADVANCED MODE")
    if RECORD_VIDEO: print("  Recording to output.avi")
    if ROTATE_VIEW:  print("  3-D view will auto-rotate")
    print("  Press Q to quit")
    print("=" * 55)

    mp_pose    = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    writer = None
    if RECORD_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter('output.avi', fourcc, 20, (640, 480))

    pose = mp_pose.Pose(
        static_image_mode=False, model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    renderer = AdvancedRenderer()
    prev_xyz = None
    fc = 0; t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok: continue
        fc += 1

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        annotated = frame.copy()

        xyz_3d  = None
        gesture = None
        angles  = None

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0,255,255), thickness=2, circle_radius=3),
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0,200,200), thickness=2))

            raw_xyz = np.array([[p.x, p.y, p.z]
                                 for p in results.pose_landmarks.landmark],
                                dtype=np.float32)

            xyz_3d, angles = reconstruct(raw_xyz, prev_xyz,
                                         smooth=0.4, depth_scale=2.5)
            prev_xyz = xyz_3d.copy()
            gesture  = detect_gesture(raw_xyz)   # Use raw (0-1) space

        else:
            prev_xyz = None

        # Overlays
        fps = fc / (time.time() - t0 + 1e-6)
        cv2.putText(annotated, f"FPS {fps:.1f}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        if gesture:
            cv2.putText(annotated, gesture, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 3)

        cv2.imshow("Holographic System | Q to quit", annotated)
        renderer.update(xyz_3d, gesture, angles)

        if writer:
            writer.write(annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()
    plt.close('all')
    pose.close()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
