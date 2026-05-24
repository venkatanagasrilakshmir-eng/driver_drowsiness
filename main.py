"""
Driver Drowsiness Detection System
==================================

Main entry point for the drowsiness detection application.
Integrates face detection, eye tracking, and alarm triggering.

This project uses:
- OpenCV (cv2) for video capture and image processing
- MediaPipe for facial landmark detection
- NumPy for numerical operations
- Pygame for audio playback

Author: Student Project
Date: 2024
"""

import cv2
import sys
import time
from detector import DrowsinessDetector
from alarm import DrowsinessAlarm
from utils import draw_text


class DrowsinessDetectionApp:
    """
    Main application class for the drowsiness detection system.
    Manages the overall flow of video capture, detection, and alarm triggering.
    """
    
    def __init__(self):
        """
        Initialize the drowsiness detection application.
        Sets up camera, detector, and alarm components.
        """
        print("=" * 60)
        print("DRIVER DROWSINESS DETECTION SYSTEM")
        print("=" * 60)
        
        # Initialize camera
        self.cap = None
        self.frame_width = None
        self.frame_height = None
        self.fps = 30
        
        # Initialize detector and alarm
        self.detector = None
        self.alarm = None
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.fps_counter = 0
        
        # State tracking
        self.is_running = True
        self.alarm_active = False
        
        # Initialize components
        self._initialize_camera()
        self._initialize_detector()
        self._initialize_alarm()
        
        print("[INFO] Application initialized successfully.")
        print("[INFO] Press 'q' to quit, 'r' to reset detector state.")
        print("=" * 60)
    
    def _initialize_camera(self):
        """
        Initialize the webcam for video capture.
        Handles camera detection and configuration.
        """
        print("[INFO] Initializing camera...")
        
        # Try multiple camera indices
        camera_indices = [0, 1, 2, -1]  # -1 is a fallback that sometimes works
        self.cap = None
        
        for idx in camera_indices:
            try:
                print(f"[INFO] Trying camera index {idx}...")
                temp_cap = cv2.VideoCapture(idx)
                
                if temp_cap.isOpened():
                    # Try to read a frame to verify it works
                    ret, frame = temp_cap.read()
                    if ret and frame is not None:
                        print(f"[INFO] Successfully opened camera at index {idx}")
                        self.cap = temp_cap
                        break
                    else:
                        print(f"[WARNING] Camera index {idx} opened but cannot read frames")
                        temp_cap.release()
            except Exception as e:
                print(f"[WARNING] Camera index {idx} failed: {e}")
                continue
        
        # If no physical camera found, show error with troubleshooting
        if self.cap is None:
            print("[ERROR] Camera initialization failed - no camera device found!")
            print("\nTROUBLESHOOTING:")
            print("1. Ensure your camera/webcam is properly connected to your computer")
            print("2. Close any other applications using the camera (Zoom, Teams, OBS, etc.)")
            print("3. Check Device Manager to confirm your camera is recognized")
            print("4. Try unplugging and replugging your webcam")
            print("5. Check Windows Settings > Privacy > Camera permissions")
            print("6. Restart your computer if nothing works")
            sys.exit(1)
        
        try:
            # Set camera resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer
            
            # Get actual frame dimensions
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"[INFO] Camera initialized. Resolution: {self.frame_width}x{self.frame_height}")
        
        except Exception as e:
            print(f"[ERROR] Failed to configure camera: {e}")
            sys.exit(1)
    
    def _initialize_detector(self):
        """
        Initialize the drowsiness detector.
        """
        print("[INFO] Initializing detector...")
        
        try:
            # Create detector with thresholds
            self.detector = DrowsinessDetector(
                ear_threshold=0.02,  # Threshold for closed eyes (direct vertical distance)
                closed_eyes_duration_threshold=2.0,  # 2 seconds of closed eyes
                nod_count_threshold=3,  # 3 nods to trigger alarm
                nod_time_window=10.0,  # Within 10 seconds
            )
            print("[INFO] Detector initialized successfully.")
        
        except Exception as e:
            print(f"[ERROR] Failed to initialize detector: {e}")
            sys.exit(1)
    
    def _initialize_alarm(self):
        """
        Initialize the alarm system.
        """
        print("[INFO] Initializing alarm...")
        
        try:
            self.alarm = DrowsinessAlarm(alarm_file_path="alarm_sound.wav")
            print("[INFO] Alarm initialized successfully.")
        
        except Exception as e:
            print(f"[ERROR] Failed to initialize alarm: {e}")
            sys.exit(1)
    
    def _process_frame(self, frame):
        """
        Process a single frame for drowsiness detection.
        
        Args:
            frame: The video frame from the camera
            
        Returns:
            tuple: (is_drowsy, processed_frame, detection_data)
        """
        # Run detection
        is_drowsy, frame_with_landmarks, detection_data = self.detector.detect(frame)
        
        return is_drowsy, frame_with_landmarks, detection_data
    
    def _draw_ui(self, frame, detection_data, is_drowsy):
        """
        Draw the user interface elements on the frame.
        Includes EAR values, status, and warnings.
        
        Args:
            frame: The video frame
            detection_data: Dictionary with detection metrics
            is_drowsy: Boolean indicating drowsiness state
            
        Returns:
            frame: Frame with UI elements drawn
        """
        # Calculate eye state based on threshold
        ear_threshold = 0.02
        eyes_are_closed = detection_data['average_ear'] < ear_threshold
        
        # Display EAR values
        frame = draw_text(
            frame,
            f"Left EAR: {detection_data['left_ear']:.3f}",
            (10, 30),
            font_scale=0.6,
            color=(200, 200, 0),
            thickness=1,
        )
        
        frame = draw_text(
            frame,
            f"Right EAR: {detection_data['right_ear']:.3f}",
            (10, 55),
            font_scale=0.6,
            color=(200, 200, 0),
            thickness=1,
        )
        
        frame = draw_text(
            frame,
            f"Avg EAR: {detection_data['average_ear']:.3f}",
            (10, 80),
            font_scale=0.6,
            color=(200, 200, 0),
            thickness=1,
        )
        
        # Display eye state
        eye_state_color = (0, 0, 255) if eyes_are_closed else (0, 255, 0)
        eye_state_text = "Eyes: CLOSED" if eyes_are_closed else "Eyes: OPEN"
        frame = draw_text(
            frame,
            eye_state_text,
            (10, 155),
            font_scale=0.6,
            color=eye_state_color,
            thickness=2,
        )
        
        # Display closed eyes duration
        frame = draw_text(
            frame,
            f"Eyes Closed: {detection_data['eyes_closed_duration']:.1f}s",
            (10, 105),
            font_scale=0.6,
            color=(200, 200, 0),
            thickness=1,
        )
        
        # Display nod count
        frame = draw_text(
            frame,
            f"Nods (10s): {detection_data['nod_count']}",
            (10, 130),
            font_scale=0.6,
            color=(200, 200, 0),
            thickness=1,
        )
        
        # Display FPS
        frame = draw_text(
            frame,
            f"FPS: {self.fps_counter:.1f}",
            (self.frame_width - 120, 30),
            font_scale=0.6,
            color=(0, 255, 0),
            thickness=1,
        )
        
        # Display status
        if not detection_data["face_detected"]:
            frame = draw_text(
                frame,
                "NO FACE DETECTED",
                (self.frame_width // 2 - 100, 60),
                font_scale=1.0,
                color=(0, 0, 255),
                thickness=2,
            )
        else:
            status_color = (0, 0, 255) if is_drowsy else (0, 255, 0)
            status_text = "DROWSY! WAKE UP!" if is_drowsy else "ALERT"
            
            frame = draw_text(
                frame,
                status_text,
                (self.frame_width // 2 - 80, 60),
                font_scale=1.2,
                color=status_color,
                thickness=2,
            )
        
        return frame
    
    def _update_fps_counter(self):
        """
        Update the FPS counter for real-time display.
        """
        self.frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        if elapsed_time >= 1.0:
            self.fps_counter = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = current_time
    
    def _handle_key_input(self, key):
        """
        Handle keyboard input.
        
        Args:
            key: The key pressed (from cv2.waitKey)
        """
        if key == ord('q'):
            print("[INFO] Quit command received.")
            self.is_running = False
        
        elif key == ord('r'):
            print("[INFO] Reset command received.")
            self.detector.reset()
            self.alarm.stop_alarm()
            self.alarm_active = False
    
    def run(self):
        """
        Main application loop.
        Continuously captures frames, processes them, and displays results.
        """
        print("[INFO] Starting main loop...")
        
        consecutive_failed_frames = 0
        max_consecutive_failures = 30  # Allow up to 30 consecutive frame failures before giving up
        
        try:
            while self.is_running:
                # Capture frame from camera
                ret, frame = self.cap.read()
                
                if not ret:
                    consecutive_failed_frames += 1
                    print(f"[WARNING] Failed to read frame ({consecutive_failed_frames}/{max_consecutive_failures})")
                    
                    # Give camera a moment to recover
                    if consecutive_failed_frames < max_consecutive_failures:
                        time.sleep(0.1)
                        continue
                    else:
                        print("[ERROR] Camera failed to recover. Exiting.")
                        break
                else:
                    consecutive_failed_frames = 0  # Reset counter on successful frame
                
                # Flip frame horizontally for better user experience
                frame = cv2.flip(frame, 1)
                
                # Process frame for drowsiness detection
                is_drowsy, frame_with_landmarks, detection_data = self._process_frame(frame)
                
                # Handle alarm triggering/stopping
                if is_drowsy:
                    if not self.alarm_active:
                        self.alarm.trigger_alarm(loops=-1)
                        self.alarm_active = True
                else:
                    if self.alarm_active:
                        self.alarm.stop_alarm()
                        self.alarm_active = False
                
                # Draw UI elements
                frame_with_ui = self._draw_ui(frame_with_landmarks, detection_data, is_drowsy)
                
                # Update FPS counter
                self._update_fps_counter()
                
                # Display the frame
                cv2.imshow("Driver Drowsiness Detection", frame_with_ui)
                
                # Wait for key press (1ms timeout)
                key = cv2.waitKey(1) & 0xFF
                if key != 255:  # 255 means no key pressed
                    self._handle_key_input(key)
        
        except KeyboardInterrupt:
            print("[INFO] Keyboard interrupt received.")
        
        except Exception as e:
            print(f"[ERROR] An error occurred in the main loop: {e}")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """
        Clean up resources before exiting.
        Releases camera and closes windows.
        """
        print("\n[INFO] Cleaning up resources...")
        
        # Stop alarm if playing
        if self.alarm and self.alarm_active:
            self.alarm.stop_alarm()
        
        # Release camera
        if self.cap:
            self.cap.release()
            print("[INFO] Camera released.")
        
        # Close all OpenCV windows
        cv2.destroyAllWindows()
        print("[INFO] All windows closed.")
        
        print("=" * 60)
        print("SYSTEM SHUTDOWN COMPLETE")
        print("=" * 60)


def main():
    """
    Entry point for the application.
    """
    try:
        app = DrowsinessDetectionApp()
        app.run()
    
    except Exception as e:
        print(f"[FATAL ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
