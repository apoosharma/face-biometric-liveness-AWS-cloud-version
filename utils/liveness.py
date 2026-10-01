"""
================================================================================
LIVENESS DETECTION MODULE: utils/liveness.py
Face-based Biometric Authentication System with Liveness Detection
================================================================================

VIVA / TECHNICAL EXPLANATION:
-----------------------------
1. What is Presentation Attack Detection (PAD)?
   Biometric face recognition systems are vulnerable to spoofing attacks, primarily:
   - 2D Print Attacks: Holding a printed photograph of an authorized user.
   - 2D Screen/Replay Attacks: Displaying a video or picture on a smartphone/tablet.
   - 3D Mask Attacks: Wearing a silicon mask.

2. Soukupová & Čech (2016) Eye Aspect Ratio (EAR) Algorithm:
   - For each detected face, 68 facial landmark coordinates (x, y) are mapped.
   - Eye landmarks:
     * Left eye points:  p36 (outer corner), p37 (top-left), p38 (top-right),
                         p39 (inner corner), p40 (bottom-right), p41 (bottom-left)
     * Right eye points: p42 (inner corner), p43 (top-left), p44 (top-right),
                         p45 (outer corner), p46 (bottom-right), p47 (bottom-left)
   
   - Formula:
                    ||p2 - p6|| + ||p3 - p5||
           EAR = -------------------------------
                        2 * ||p1 - p4||

     Where ||p_i - p_j|| is the Euclidean 2D distance between landmark points:
     distance = sqrt((x_i - x_j)^2 + (y_i - y_j)^2)

   - Mathematical intuition:
     * When the eye is open, vertical distances ||p2 - p6|| and ||p3 - p5|| are large,
       yielding EAR values between 0.27 and 0.38.
     * When the eye blinks (closes), vertical distances approach zero,
       causing EAR to drop rapidly below 0.21 - 0.23.
     * A real human blink lasts between 100ms to 400ms (typically 2-4 consecutive frames at 30 FPS).
     * A printed photo maintains a static EAR indefinitely (zero blink variation),
       allowing our algorithm to intercept photo spoofing attacks within a configurable timeout window.
================================================================================
"""

import os
import time
import math
import numpy as np
import cv2
from scipy.spatial import distance as dist

# 68-Point Facial Landmark Index Mapping (0-indexed)
# Left eye indices: 36 to 41
LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
# Right eye indices: 42 to 47
RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]

# Standard Thresholds for Blink Detection
DEFAULT_EAR_THRESHOLD = 0.23        # EAR below this indicates eye closure
DEFAULT_CONSECUTIVE_FRAMES = 2      # Minimum consecutive frames below threshold to count as blink
DEFAULT_SPOOF_TIMEOUT_SEC = 3.5     # Seconds without blink before triggering static photo spoof alert


def calculate_ear(eye_landmarks: np.ndarray) -> float:
    """
    Calculates the Eye Aspect Ratio (EAR) given 6 facial landmark points of an eye.
    
    Parameters:
        eye_landmarks (np.ndarray): Array of shape (6, 2) containing (x, y) coordinates.
                                   Points ordered: p1 (corner), p2, p3, p4 (corner), p5, p6.
                                   
    Returns:
        float: Computed Eye Aspect Ratio.
    """
    if len(eye_landmarks) < 6:
        return 0.30

    # Compute Euclidean distances between the two sets of vertical landmarks
    # Vertical distance 1: ||p2 - p6||
    v1 = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
    # Vertical distance 2: ||p3 - p5||
    v2 = dist.euclidean(eye_landmarks[2], eye_landmarks[4])

    # Compute Euclidean distance between the horizontal eye landmark coordinates
    # Horizontal distance: ||p1 - p4||
    h = dist.euclidean(eye_landmarks[0], eye_landmarks[3])

    # Avoid division by zero if coordinates overlap
    if h == 0:
        return 0.0

    # Apply the Soukupová & Čech formula
    ear = (v1 + v2) / (2.0 * h)
    return float(ear)


class LivenessDetector:
    """
    Presentation Attack Detection (PAD) engine tracking ocular dynamics,
    blink frequency, and static-photo spoofing detection in real time.
    """

    def __init__(
        self,
        ear_threshold: float = DEFAULT_EAR_THRESHOLD,
        consec_frames: int = DEFAULT_CONSECUTIVE_FRAMES,
        spoof_timeout: float = DEFAULT_SPOOF_TIMEOUT_SEC
    ):
        """
        Initializes the Liveness Detector state machine.
        
        Parameters:
            ear_threshold (float): Threshold under which an eye is deemed closed.
            consec_frames (int): Number of consecutive frames needed for eye closure.
            spoof_timeout (float): Max seconds allowed without a blink before flagging as photo attack.
        """
        self.ear_threshold = ear_threshold
        self.consec_frames = consec_frames
        self.spoof_timeout = spoof_timeout

        # State tracking variables
        self.frame_counter = 0
        self.blink_count = 0
        self.ear_history = []
        self.first_face_seen_time = None
        self.is_live = False
        self.is_spoof = False
        self.liveness_status = "Scanning for Blink..."
        self.current_ear = 0.30

        # Attempt to load dlib predictor if available
        self.dlib_detector = None
        self.dlib_predictor = None
        self._init_dlib_if_available()

        # OpenCV Haar cascade fallbacks for robust execution without external files
        self.face_cascade = None
        self.eye_cascade = None
        try:
            if hasattr(cv2, 'CascadeClassifier') and hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        except Exception:
            pass

    def _init_dlib_if_available(self):
        """
        Initializes dlib frontal face detector and 68-point shape predictor if available.
        """
        try:
            import dlib
            self.dlib_detector = dlib.get_frontal_face_detector()
            
            # Check local data folder for the dat file
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
            dat_path = os.path.join(data_dir, "shape_predictor_68_face_landmarks.dat")
            
            if os.path.exists(dat_path):
                self.dlib_predictor = dlib.shape_predictor(dat_path)
            else:
                self.dlib_predictor = None
        except Exception:
            self.dlib_detector = None
            self.dlib_predictor = None

    def reset(self):
        """
        Resets the liveness detector state machine for a new authentication session.
        """
        self.frame_counter = 0
        self.blink_count = 0
        self.ear_history = []
        self.first_face_seen_time = None
        self.is_live = False
        self.is_spoof = False
        self.liveness_status = "Scanning for Blink..."
        self.current_ear = 0.30

    def process_frame(self, frame: np.ndarray):
        """
        Processes a single video frame to extract facial landmarks, compute EAR,
        update blink counters, and classify liveness (Real vs Spoof).
        
        Parameters:
            frame (np.ndarray): Input BGR or RGB video frame from webcam.
            
        Returns:
            dict: Diagnostic results containing:
                - 'is_live' (bool): True if verified genuine live face.
                - 'is_spoof' (bool): True if flagged as static photo spoof.
                - 'status' (str): Human-readable status string.
                - 'ear' (float): Current frame Eye Aspect Ratio.
                - 'blink_count' (int): Total verified blinks recorded in session.
                - 'landmarks' (list): 2D points [(x, y), ...] for overlay drawing.
                - 'face_box' (tuple): (x, y, w, h) of primary face.
        """
        if frame is None:
            return self._empty_result()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        now = time.time()

        landmarks_points = []
        face_box = None
        ear = 0.30

        # Method 1: Dlib 68-Point Landmark Extraction (if predictor is available)
        if self.dlib_detector is not None and self.dlib_predictor is not None:
            try:
                rects = self.dlib_detector(gray, 0)
                if len(rects) > 0:
                    rect = rects[0]
                    face_box = (rect.left(), rect.top(), rect.width(), rect.height())
                    shape = self.dlib_predictor(gray, rect)
                    landmarks = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])
                    landmarks_points = landmarks.tolist()

                    left_eye = landmarks[LEFT_EYE_INDICES]
                    right_eye = landmarks[RIGHT_EYE_INDICES]

                    left_ear = calculate_ear(left_eye)
                    right_ear = calculate_ear(right_eye)
                    ear = (left_ear + right_ear) / 2.0
            except Exception:
                pass

        # Method 2: OpenCV Ocular Geometry Fallback (Zero external dependency)
        if face_box is None:
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
            if len(faces) > 0:
                # Select largest detected face
                face_box = tuple(faces[0])
                fx, fy, fw, fh = face_box
                
                # Face upper half contains the eyes
                face_roi_gray = gray[fy:fy + int(fh * 0.55), fx:fx + fw]
                eyes = self.eye_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20))
                
                if len(eyes) >= 2:
                    # Sort eyes horizontally: left eye (viewer's left) and right eye
                    eyes = sorted(eyes, key=lambda e: e[0])
                    # Synthetic landmark representation around eye bounding boxes for EAR
                    ear_samples = []
                    for ex, ey, ew, eh in eyes[:2]:
                        # Aspect ratio of the eye contour: height / width
                        eye_aspect = eh / float(ew) if ew > 0 else 0.3
                        # Normalized to standard 0.20-0.35 range
                        ear_samples.append(min(max(eye_aspect * 0.55, 0.15), 0.40))
                    ear = float(np.mean(ear_samples))
                elif len(eyes) == 1:
                    # One eye detected - might be partially closed or in mid-blink
                    ex, ey, ew, eh = eyes[0]
                    eye_aspect = eh / float(ew) if ew > 0 else 0.25
                    ear = float(min(max(eye_aspect * 0.50, 0.18), 0.35))
                else:
                    # Eyes closed during blink
                    ear = 0.19

        # If no face detected, reset timer
        if face_box is None:
            self.first_face_seen_time = None
            return self._empty_result()

        # Face is present: Update temporal timer
        if self.first_face_seen_time is None:
            self.first_face_seen_time = now

        self.current_ear = ear
        self.ear_history.append(ear)
        if len(self.ear_history) > 60:
            self.ear_history.pop(0)

        # Liveness State Machine Logic:
        # Check if EAR is below the threshold (eyes closed)
        if ear < self.ear_threshold:
            self.frame_counter += 1
        else:
            # If eyes were closed for sufficient consecutive frames and then re-opened -> VALID BLINK
            if self.frame_counter >= self.consec_frames:
                self.blink_count += 1
                self.is_live = True
                self.is_spoof = False
                self.liveness_status = "LIVE: REAL FACE CONFIRMED (Blink Passed)"
            self.frame_counter = 0

        # Check for Static Photo Spoofing Attack:
        elapsed_face_time = now - self.first_face_seen_time
        if not self.is_live:
            if elapsed_face_time > self.spoof_timeout and self.blink_count == 0:
                self.is_spoof = True
                self.liveness_status = "SPOOF DETECTED: Photo Attack (No Blink)"
            else:
                remaining = max(0.0, self.spoof_timeout - elapsed_face_time)
                self.liveness_status = f"Scanning for Blink... ({remaining:.1f}s remaining)"

        return {
            "is_live": self.is_live,
            "is_spoof": self.is_spoof,
            "status": self.liveness_status,
            "ear": float(ear),
            "blink_count": int(self.blink_count),
            "landmarks": landmarks_points,
            "face_box": face_box
        }

    def _empty_result(self) -> dict:
        """Returns default state when no face is present."""
        return {
            "is_live": False,
            "is_spoof": False,
            "status": "No Face Detected",
            "ear": 0.0,
            "blink_count": self.blink_count,
            "landmarks": [],
            "face_box": None
        }


def draw_liveness_hud(frame: np.ndarray, result: dict, user_label: str = "Unknown", confidence: float = 0.0) -> np.ndarray:
    """
    Renders diagnostic HUD overlay on the video frame for live presentation:
    - Face bounding box color-coded (Green: Live & Matched, Red: Spoof, Yellow: Unverified/Scanning)
    - Facial landmark points / ocular markers
    - Real-time EAR readout & Blink counter
    - Live status badge
    
    Parameters:
        frame (np.ndarray): Original BGR/RGB image.
        result (dict): Diagnostic output from LivenessDetector.process_frame().
        user_label (str): Matched user name or 'Unknown'.
        confidence (float): Recognition confidence score (0.0 to 1.0).
        
    Returns:
        np.ndarray: Annotated video frame.
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    face_box = result.get("face_box")
    is_live = result.get("is_live", False)
    is_spoof = result.get("is_spoof", False)
    ear = result.get("ear", 0.0)
    blink_count = result.get("blink_count", 0)
    status_text = result.get("status", "")

    # 1. Determine HUD Color Theme
    if is_spoof:
        box_color = (0, 0, 220)       # Red: Spoof Photo Attack
        status_color = (0, 0, 255)
        badge_bg = (0, 0, 180)
    elif is_live:
        box_color = (0, 200, 0)       # Green: Genuine Live Face
        status_color = (0, 255, 0)
        badge_bg = (0, 140, 0)
    else:
        box_color = (0, 215, 255)     # Yellow/Cyan: Scanning / Awaiting Blink
        status_color = (0, 255, 255)
        badge_bg = (50, 50, 50)

    # 2. Draw Face Bounding Box & Corner Brackets
    if face_box:
        x, y, bw, bh = face_box
        # Main box
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), box_color, 2)
        
        # Tech aesthetic corner accents
        line_len = int(bw * 0.15)
        thick = 3
        # Top-left
        cv2.line(annotated, (x, y), (x + line_len, y), box_color, thick)
        cv2.line(annotated, (x, y), (x, y + line_len), box_color, thick)
        # Top-right
        cv2.line(annotated, (x + bw, y), (x + bw - line_len, y), box_color, thick)
        cv2.line(annotated, (x + bw, y), (x + bw, y + line_len), box_color, thick)
        # Bottom-left
        cv2.line(annotated, (x, y + bh), (x + line_len, y + bh), box_color, thick)
        cv2.line(annotated, (x, y + bh), (x, y + bh - line_len), box_color, thick)
        # Bottom-right
        cv2.line(annotated, (x + bw, y + bh), (x + bw - line_len, y + bh), box_color, thick)
        cv2.line(annotated, (x + bw, y + bh), (x + bw, y + bh - line_len), box_color, thick)

        # Draw User Identity Label above bounding box
        label_text = f"{user_label} ({confidence * 100:.1f}%)" if confidence > 0 else user_label
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x, max(0, y - th - 10)), (x + tw + 10, y), box_color, -1)
        cv2.putText(annotated, label_text, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 3. Draw 68 Facial Landmarks (if mapped)
    landmarks = result.get("landmarks", [])
    if len(landmarks) == 68:
        for (lx, ly) in landmarks:
            cv2.circle(annotated, (int(lx), int(ly)), 1, (0, 255, 255), -1)
        # Highlight eye contours in green
        for idx in LEFT_EYE_INDICES + RIGHT_EYE_INDICES:
            cv2.circle(annotated, (int(landmarks[idx][0]), int(landmarks[idx][1])), 2, (0, 255, 0), -1)

    # 4. Top Telemetry HUD Bar (Glassmorphism effect)
    overlay = annotated.copy()
    cv2.rectangle(overlay, (10, 10), (w - 10, 75), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)
    cv2.rectangle(annotated, (10, 10), (w - 10, 75), box_color, 1)

    # Status Text & Metrics
    cv2.putText(annotated, f"STATUS: {status_text}", (25, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2)
    metrics_str = f"EAR: {ear:.3f} (Thresh: {DEFAULT_EAR_THRESHOLD:.2f})  |  Blinks: {blink_count}"
    cv2.putText(annotated, metrics_str, (25, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)

    # Liveness Badge in Top-Right
    badge_text = "REAL FACE" if is_live else ("SPOOF ATTACK" if is_spoof else "CHECKING")
    (bw, bh), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    badge_x = w - bw - 35
    cv2.rectangle(annotated, (badge_x, 22), (w - 20, 58), badge_bg, -1)
    cv2.putText(annotated, badge_text, (badge_x + 8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    return annotated
