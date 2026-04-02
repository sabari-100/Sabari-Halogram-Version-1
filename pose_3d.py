"""
pose_3d.py
──────────
3-D Pose Reconstruction Helpers

MediaPipe provides a depth hint (z) for each landmark, but it is
NOT true metric depth — it is relative to the hip midpoint and
scaled to the torso length.  This module adds a simple geometric
"depth boost" using body-part proportions so the figure looks more
three-dimensional on screen.

Import this from hologram_main.py (optional enhancement).
"""

import numpy as np


# ─────────────────────────────────────────────────────────
# LANDMARK INDICES (MediaPipe Pose — 33 total)
# ─────────────────────────────────────────────────────────
NOSE         = 0
L_SHOULDER   = 11
R_SHOULDER   = 12
L_ELBOW      = 13
R_ELBOW      = 14
L_WRIST      = 15
R_WRIST      = 16
L_HIP        = 23
R_HIP        = 24
L_KNEE       = 25
R_KNEE       = 26
L_ANKLE      = 27
R_ANKLE      = 28


def normalize_to_hip_center(xyz):
    """
    Translate the skeleton so the hip midpoint = origin.
    This makes rotation & scaling pose-invariant.

    Args:
        xyz : np.ndarray  (33, 3)
    Returns:
        centered : np.ndarray (33, 3)
    """
    hip_mid = (xyz[L_HIP] + xyz[R_HIP]) / 2.0
    return xyz - hip_mid


def estimate_torso_scale(xyz):
    """
    Returns the distance from shoulder midpoint to hip midpoint.
    Used to normalize depth (z) so people at different distances
    look similarly proportioned.
    """
    shoulder_mid = (xyz[L_SHOULDER] + xyz[R_SHOULDER]) / 2.0
    hip_mid      = (xyz[L_HIP]      + xyz[R_HIP])      / 2.0
    return float(np.linalg.norm(shoulder_mid - hip_mid)) + 1e-6


def boost_depth(xyz, scale_factor=3.0):
    """
    MediaPipe's z values are tiny (often < 0.1).
    Multiply them so the figure looks 3-D in the Matplotlib viewer.

    Args:
        xyz          : np.ndarray (33, 3)
        scale_factor : how much to amplify z
    Returns:
        boosted : np.ndarray (33, 3)
    """
    out    = xyz.copy()
    out[:, 2] *= scale_factor
    return out


def smooth_landmarks(prev, curr, alpha=0.5):
    """
    Exponential Moving Average (EMA) smoothing.

    alpha = 0   → no update   (fully frozen)
    alpha = 1   → no history  (raw signal)
    """
    if prev is None:
        return curr.copy()
    return alpha * curr + (1 - alpha) * prev


def compute_joint_angles(xyz):
    """
    Compute a handful of useful joint angles (degrees).
    Useful for gesture recognition or AR overlays.

    Returns:
        dict  { joint_name : angle_degrees }
    """

    def angle(a, b, c):
        """Angle at vertex b formed by segments b-a and b-c."""
        ba = a - b
        bc = c - b
        cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
        return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))

    return {
        'left_elbow':   angle(xyz[L_SHOULDER], xyz[L_ELBOW],  xyz[L_WRIST]),
        'right_elbow':  angle(xyz[R_SHOULDER], xyz[R_ELBOW],  xyz[R_WRIST]),
        'left_knee':    angle(xyz[L_HIP],      xyz[L_KNEE],   xyz[L_ANKLE]),
        'right_knee':   angle(xyz[R_HIP],      xyz[R_KNEE],   xyz[R_ANKLE]),
        'left_shoulder':  angle(xyz[L_ELBOW],  xyz[L_SHOULDER], xyz[L_HIP]),
        'right_shoulder': angle(xyz[R_ELBOW],  xyz[R_SHOULDER], xyz[R_HIP]),
    }


def reconstruct(raw_xyz, prev_xyz=None, smooth=0.5, depth_scale=3.0):
    """
    Full reconstruction pipeline:
      1. Normalize to hip center
      2. Smooth (EMA)
      3. Boost depth for visual clarity

    Args:
        raw_xyz     : np.ndarray (33, 3) — fresh from MediaPipe
        prev_xyz    : np.ndarray (33, 3) or None — previous frame
        smooth      : EMA alpha
        depth_scale : z amplification

    Returns:
        xyz_out  : np.ndarray (33, 3)  ready for rendering
        angles   : dict of joint angles
    """
    xyz = normalize_to_hip_center(raw_xyz)
    xyz = smooth_landmarks(prev_xyz, xyz, alpha=smooth)
    xyz = boost_depth(xyz, scale_factor=depth_scale)

    angles = compute_joint_angles(xyz)
    return xyz, angles
