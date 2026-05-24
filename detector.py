"""
Detector module for face, eye, and head movement detection.
This module uses MediaPipe Face Mesh to detect facial features and track drowsiness indicators.
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from utils import (
    calculate_eye_aspect_ratio,
    get_eye_landmarks_from_mesh,
    calculate_distance,
    normalize_coordinates,
)


class DrowsinessDetector:
    """
    Main detector class for drowsiness detection.
    Tracks eyes and head movements to identify signs of drowsiness.
    """
    
    # MediaPipe Face Mesh landmark indices for eyes - FULL CONTOUR
    # These include all points around the eye margin
    # Right eye contour
    RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    # Left eye contour  
    LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    
    # Nose tip landmark index for head nod detection
    NOSE_TIP_INDEX = 1
    
    def __init__(
        self,
        ear_threshold=0.2,
        closed_eyes_duration_threshold=2.0,
        nod_count_threshold=3,
        nod_time_window=10.0,
    ):
        """
        Initialize the drowsiness detector.
        
        Args:
            ear_threshold: Eye Aspect Ratio threshold for detecting closed eyes (default 0.2)
            closed_eyes_duration_threshold: Time in seconds eyes must be closed to trigger alarm (default 2.0)
            nod_count_threshold: Number of head nods to trigger alarm (default 3)
            nod_time_window: Time window in seconds for counting nods (default 10.0)
        """
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        
        # Initialize MediaPipe drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Drowsiness detection thresholds
        self.ear_threshold = ear_threshold
        self.closed_eyes_duration_threshold = closed_eyes_duration_threshold
        self.nod_count_threshold = nod_count_threshold
        self.nod_time_window = nod_time_window
        
        # State tracking variables
        self.closed_eyes_count = 0  # Consecutive frames with closed eyes
        self.closed_eyes_start_time = None  # Time when eyes first closed
        self.previous_nose_y = None  # Previous nose Y position for nod detection
        self.nod_history = deque()  # Timestamp history of nods
        self.is_nodding = False  # Current nod state
        self.nod_threshold = 15  # Pixel threshold for detecting head nods
        
        # Face detection state
        self.face_detected = False
        self.current_frame_counter = 0
        
        print("[INFO] DrowsinessDetector initialized successfully.")
    
    def detect(self, frame):
        """
        Process a frame and detect drowsiness indicators.
        
        Args:
            frame: The video frame (numpy array) from OpenCV
            
        Returns:
            tuple: (is_drowsy, frame_with_landmarks, detection_data)
                - is_drowsy: Boolean indicating if drowsiness is detected
                - frame_with_landmarks: Frame with facial landmarks drawn
                - detection_data: Dict with EAR, state, and other metrics
        """
        self.current_frame_counter += 1
        
        # Convert frame to RGB (MediaPipe requires RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_height, frame_width, _ = frame.shape
        
        # Initialize detection data
        detection_data = {
            "left_ear": 0.0,
            "right_ear": 0.0,
            "average_ear": 0.0,
            "eyes_closed_duration": 0.0,
            "nod_count": len(self.nod_history),
            "face_detected": False,
            "is_drowsy": False,
        }
        
        # Run face mesh detection
        results = self.face_mesh.process(rgb_frame)
        
        # Initialize output frame
        output_frame = frame.copy()
        
        # Check if face was detected
        if not results.multi_face_landmarks or len(results.multi_face_landmarks) == 0:
            self.face_detected = False
            self.closed_eyes_count = 0
            return False, output_frame, detection_data
        
        self.face_detected = True
        detection_data["face_detected"] = True
        
        # Get face landmarks
        face_landmarks = results.multi_face_landmarks[0]
        
        # Extract eye landmarks using full eye contour
        left_eye = get_eye_landmarks_from_mesh(face_landmarks, self.LEFT_EYE_INDICES)
        right_eye = get_eye_landmarks_from_mesh(face_landmarks, self.RIGHT_EYE_INDICES)
        
        # DEBUG: Print eye landmark coordinates every 150 frames
        if self.current_frame_counter % 150 == 0:
            left_y_values = [pt[1] for pt in left_eye]
            right_y_values = [pt[1] for pt in right_eye]
            left_vertical = max(left_y_values) - min(left_y_values)
            right_vertical = max(right_y_values) - min(right_y_values)
            print(f"\n[DEBUG] Frame {self.current_frame_counter} - Eye Landmarks Analysis:")
            print(f"  LEFT_EYE Y-range: {min(left_y_values):.4f} to {max(left_y_values):.4f} (vertical spread: {left_vertical:.4f})")
            print(f"  RIGHT_EYE Y-range: {min(right_y_values):.4f} to {max(right_y_values):.4f} (vertical spread: {right_vertical:.4f})")
        
        # Convert to pixel coordinates for drawing
        left_eye_pixels = [
            normalize_coordinates(pt[0], pt[1], frame_width, frame_height)
            for pt in left_eye
        ]
        right_eye_pixels = [
            normalize_coordinates(pt[0], pt[1], frame_width, frame_height)
            for pt in right_eye
        ]
        
        # Draw eyes on frame
        self._draw_eyes(output_frame, left_eye_pixels, right_eye_pixels)
        
        # Calculate Eye Aspect Ratio (EAR)
        left_ear = calculate_eye_aspect_ratio(left_eye)
        right_ear = calculate_eye_aspect_ratio(right_eye)
        average_ear = (left_ear + right_ear) / 2.0
        
        detection_data["left_ear"] = round(left_ear, 3)
        detection_data["right_ear"] = round(right_ear, 3)
        detection_data["average_ear"] = round(average_ear, 3)
        
        # Detect closed eyes
        eyes_are_closed = average_ear < self.ear_threshold
        
        # DEBUG: Print EAR values every 30 frames to diagnose issues
        if self.current_frame_counter % 30 == 0:
            status = "CLOSED" if eyes_are_closed else "OPEN"
            print(f"[DEBUG] Frame {self.current_frame_counter}: Left EAR={left_ear:.4f}, Right EAR={right_ear:.4f}, Avg EAR={average_ear:.4f}, Threshold={self.ear_threshold}, Eyes {status}")
        
        if eyes_are_closed:
            self.closed_eyes_count += 1
            if self.closed_eyes_start_time is None:
                self.closed_eyes_start_time = self.current_frame_counter / 30.0  # Assuming 30 FPS
                print(f"[DEBUG] Eyes CLOSED detected at frame {self.current_frame_counter}!")
        else:
            self.closed_eyes_count = 0
            self.closed_eyes_start_time = None
        
        # Calculate how long eyes have been closed
        if self.closed_eyes_start_time is not None:
            eyes_closed_duration = (
                self.current_frame_counter / 30.0 - self.closed_eyes_start_time
            )
            detection_data["eyes_closed_duration"] = round(eyes_closed_duration, 2)
        
        # Detect head nods
        nose_landmark = face_landmarks.landmark[self.NOSE_TIP_INDEX]
        nose_y = normalize_coordinates(nose_landmark.x, nose_landmark.y, frame_width, frame_height)[1]
        
        if self.previous_nose_y is not None:
            nod_movement = abs(nose_y - self.previous_nose_y)
            
            # Detect downward nod
            if nod_movement > self.nod_threshold and not self.is_nodding:
                self.is_nodding = True
                self.nod_history.append(self.current_frame_counter / 30.0)
        
        # Reset nod state when nose moves back up
        if self.previous_nose_y is not None:
            nod_movement = abs(nose_y - self.previous_nose_y)
            if nod_movement <= self.nod_threshold / 2:
                self.is_nodding = False
        
        self.previous_nose_y = nose_y
        
        # Clean up old nods outside the time window
        current_time = self.current_frame_counter / 30.0
        while (
            len(self.nod_history) > 0
            and current_time - self.nod_history[0] > self.nod_time_window
        ):
            self.nod_history.popleft()
        
        detection_data["nod_count"] = len(self.nod_history)
        
        # Determine if drowsy based on criteria
        eyes_closed_too_long = (
            self.closed_eyes_start_time is not None
            and detection_data["eyes_closed_duration"]
            >= self.closed_eyes_duration_threshold
        )
        nodding_too_much = len(self.nod_history) >= self.nod_count_threshold
        
        is_drowsy = eyes_closed_too_long or nodding_too_much
        detection_data["is_drowsy"] = is_drowsy
        
        return is_drowsy, output_frame, detection_data
    
    def _draw_eyes(self, frame, left_eye_pixels, right_eye_pixels):
        """
        Draw eye landmarks on the frame.
        
        Args:
            frame: The video frame (numpy array)
            left_eye_pixels: List of pixel coordinates for left eye
            right_eye_pixels: List of pixel coordinates for right eye
        """
        # Draw circles on eye landmarks
        for point in left_eye_pixels:
            cv2.circle(frame, point, 2, (0, 255, 0), -1)
        
        for point in right_eye_pixels:
            cv2.circle(frame, point, 2, (0, 255, 0), -1)
        
        # Draw contours around eyes
        left_eye_array = np.array(left_eye_pixels, dtype=np.int32)
        right_eye_array = np.array(right_eye_pixels, dtype=np.int32)
        
        cv2.polylines(frame, [left_eye_array], True, (0, 255, 0), 2)
        cv2.polylines(frame, [right_eye_array], True, (0, 255, 0), 2)
    
    def draw_face_bbox(self, frame, face_landmarks):
        """
        Draw a bounding box around the detected face.
        
        Args:
            frame: The video frame (numpy array)
            face_landmarks: MediaPipe face landmarks
            
        Returns:
            frame: The frame with bounding box drawn
        """
        # Get bounding box coordinates
        frame_height, frame_width, _ = frame.shape
        
        x_min = int(min(lm.x for lm in face_landmarks.landmark) * frame_width)
        x_max = int(max(lm.x for lm in face_landmarks.landmark) * frame_width)
        y_min = int(min(lm.y for lm in face_landmarks.landmark) * frame_height)
        y_max = int(max(lm.y for lm in face_landmarks.landmark) * frame_height)
        
        # Add padding
        padding = 10
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(frame_width, x_max + padding)
        y_max = min(frame_height, y_max + padding)
        
        # Draw bounding box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        return frame
    
    def reset(self):
        """
        Reset the detector state.
        Useful for clearing detection history between sessions.
        """
        self.closed_eyes_count = 0
        self.closed_eyes_start_time = None
        self.previous_nose_y = None
        self.nod_history.clear()
        self.is_nodding = False
        self.face_detected = False
        self.current_frame_counter = 0
        print("[INFO] Detector state reset.")
