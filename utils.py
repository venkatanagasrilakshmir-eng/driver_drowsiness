"""
Utility functions for Eye Aspect Ratio (EAR) calculation and other helper functions.
This module provides mathematical calculations for drowsiness detection.
"""

import numpy as np
from scipy.spatial import distance


def calculate_eye_aspect_ratio(eye_landmarks):
    """
    Measure direct vertical distance between upper and lower eyelids.
    
    MediaPipe provides 6 points per eye:
    - Index 0: Right corner (outer corner)
    - Index 1: Upper outer area
    - Index 2: Upper center (top-middle of eye opening)
    - Index 3: Left corner (inner corner)  
    - Index 4: Lower outer area
    - Index 5: Lower center (bottom-middle of eye opening)
    
    We measure the DIRECT vertical distance between upper center (pt 2) and lower center (pt 5).
    This is the most reliable indicator of eye opening/closure.
    
    When eyes are CLOSED: distance ≈ 0.001-0.01 (eyelids touching)
    When eyes are OPEN: distance ≈ 0.03-0.15 (significant separation)
    
    Args:
        eye_landmarks: Array of 6 eye landmark points in normalized coordinates
        
    Returns:
        float: Vertical distance between upper and lower eyelids (< 0.02 for closed, > 0.03 for open)
    """
    # Convert to numpy array
    eye_landmarks = np.array(eye_landmarks, dtype=np.float32)
    
    # Get the center upper and center lower eyelid points
    # Point 2 (index 2): Upper center of eye
    # Point 5 (index 5): Lower center of eye
    upper_point = eye_landmarks[2]  # Upper center
    lower_point = eye_landmarks[5]  # Lower center
    
    # Direct vertical distance between upper and lower eyelid centers
    vertical_distance = abs(lower_point[1] - upper_point[1])
    
    return vertical_distance


def get_eye_landmarks_from_mesh(face_landmarks, eye_indices):
    """
    Extract eye landmarks from MediaPipe face mesh landmarks.
    
    Args:
        face_landmarks: MediaPipe face landmarks object
        eye_indices: List of indices for the eye landmarks
        
    Returns:
        list: List of (x, y) coordinates for the eye
    """
    eye = []
    for idx in eye_indices:
        landmark = face_landmarks.landmark[idx]
        eye.append([landmark.x, landmark.y])
    
    return eye


def smooth_value(current_value, previous_value, smoothing_factor=0.7):
    """
    Apply exponential smoothing to reduce noise in calculated values.
    
    Args:
        current_value: The new value to be smoothed
        previous_value: The previous smoothed value
        smoothing_factor: Weight for the current value (0 to 1)
        
    Returns:
        float: The smoothed value
    """
    if previous_value is None:
        return current_value
    
    smoothed = (smoothing_factor * current_value) + (
        (1 - smoothing_factor) * previous_value
    )
    return smoothed


def calculate_distance(point1, point2):
    """
    Calculate Euclidean distance between two points.
    
    Args:
        point1: Tuple or list of (x, y) coordinates
        point2: Tuple or list of (x, y) coordinates
        
    Returns:
        float: The distance between the two points
    """
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def normalize_coordinates(x, y, frame_width, frame_height):
    """
    Normalize coordinates to pixel values based on frame dimensions.
    
    Args:
        x: Normalized x coordinate (0 to 1)
        y: Normalized y coordinate (0 to 1)
        frame_width: Width of the video frame in pixels
        frame_height: Height of the video frame in pixels
        
    Returns:
        tuple: (pixel_x, pixel_y) coordinates
    """
    pixel_x = int(x * frame_width)
    pixel_y = int(y * frame_height)
    
    return pixel_x, pixel_y


def draw_text(frame, text, position, font_scale=1.0, color=(0, 255, 0), thickness=2):
    """
    Draw text on the video frame.
    
    Args:
        frame: The video frame (numpy array)
        text: The text to display
        position: Tuple of (x, y) position
        font_scale: Size of the text
        color: BGR color tuple
        thickness: Thickness of the text
        
    Returns:
        frame: The modified frame with text drawn
    """
    import cv2
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
    )
    return frame
