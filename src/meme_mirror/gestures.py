import numpy as np


def _hand_center(hand_pts):
    return np.mean(np.array(hand_pts, dtype=np.float32), axis=0)


def _finger_extended(hand_pts, tip_idx, pip_idx):
    """True if fingertip is farther from wrist than PIP joint (rough check)."""
    wrist = np.array(hand_pts[0], dtype=np.float32)
    tip = np.array(hand_pts[tip_idx], dtype=np.float32)
    pip = np.array(hand_pts[pip_idx], dtype=np.float32)
    return np.linalg.norm(tip - wrist) > np.linalg.norm(pip - wrist)


def _thumb_up(hand_pts):
    """Thumb extended up, other fingers folded."""
    thumb_tip = hand_pts[4]
    thumb_ip = hand_pts[3]
    index_tip = hand_pts[8]
    # thumb above index_mcp
    if thumb_tip[1] > hand_pts[5][1]:
        return False
    # thumb tip above thumb ip (pointing up)
    if thumb_tip[1] > thumb_ip[1]:
        return False
    # index folded
    if _finger_extended(hand_pts, 8, 6):
        return False
    return True


def _point_up(hand_pts):
    """Index extended up, middle/ring/pinky folded."""
    if not _finger_extended(hand_pts, 8, 6):
        return False
    for tip, pip in [(12, 10), (16, 14), (20, 18)]:
        if _finger_extended(hand_pts, tip, pip):
            return False
    # index tip above index mcp (vertical)
    return hand_pts[8][1] < hand_pts[5][1]


def _middle_finger(hand_pts):
    """Middle extended up, index/ring/pinky folded."""
    if not _finger_extended(hand_pts, 12, 10):
        return False
    for tip, pip in [(8, 6), (16, 14), (20, 18)]:
        if _finger_extended(hand_pts, tip, pip):
            return False
    return hand_pts[12][1] < hand_pts[9][1]


def _peace(hand_pts):
    """Index + middle extended, others folded."""
    if not (_finger_extended(hand_pts, 8, 6) and _finger_extended(hand_pts, 12, 10)):
        return False
    for tip, pip in [(16, 14), (20, 18)]:
        if _finger_extended(hand_pts, tip, pip):
            return False
    return True


def detect_gesture(hands, pose_kp, frame_shape):
    """
    hands: list of hand landmark lists [(x,y), ...21]
    pose_kp: dict {idx: (x,y,vis)} or None
    frame_shape: (h, w)
    Returns: gesture string
    """
    h, w = frame_shape[:2]

    if not hands:
        return "none"

    # face region from pose (nose idx 0) or fallback to upper-center
    if pose_kp and 0 in pose_kp and pose_kp[0][2] > 0.5:
        face_y = pose_kp[0][1]
    else:
        face_y = int(h * 0.3)

    # 1 hand gestures
    if len(hands) == 1:
        hpts = hands[0]
        if _thumb_up(hpts):
            return "thumbs_up"
        if _point_up(hpts):
            return "point_up"
        if _middle_finger(hpts):
            return "middle_finger"
        if _peace(hpts):
            return "peace"
        # facepalm: hand center near face
        cx, cy = _hand_center(hpts)
        if abs(cy - face_y) < h * 0.15 and abs(cx - w / 2) < w * 0.25:
            return "facepalm"
        return "none"

    # 2 hands: pray / pointing_fingers
    c1 = _hand_center(hands[0])
    c2 = _hand_center(hands[1])
    dist = np.linalg.norm(c1 - c2) / w
    avg_y = (c1[1] + c2[1]) / 2

    if dist < 0.15 and avg_y > face_y:
        # pointing_fingers: both index extended, tips close
        h1, h2 = hands[0], hands[1]
        idx1_ext = _finger_extended(h1, 8, 6)
        idx2_ext = _finger_extended(h2, 8, 6)
        if idx1_ext and idx2_ext:
            tip1 = np.array(h1[8], dtype=np.float32)
            tip2 = np.array(h2[8], dtype=np.float32)
            if np.linalg.norm(tip1 - tip2) / w < 0.2:
                return "pointing_fingers"
        return "pray"

    return "none"
