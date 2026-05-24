# driver_drowsiness detection 
# Project Description
The Driver Drowsiness Detection System is a real-time computer vision application that monitors driver alertness using facial analysis. The system detects signs of drowsiness based on eye closure duration and head nodding patterns, triggering an audio alarm when the driver appears to be drowsy.

This project is designed as a college assignment and demonstrates practical applications of computer vision, facial landmark detection, and real-time signal processing.

# Features
Real-Time Face Detection: Uses MediaPipe Face Mesh to detect faces and extract 468 facial landmarks
Eye Aspect Ratio (EAR) Calculation: Monitors eye closure by calculating the distance ratio between eye landmarks
Head Nod Detection: Tracks vertical head movements to detect repeated nodding patterns
Dual Drowsiness Criteria:
Eyes closed for more than 2 seconds continuously
More than 3 head nods within a 10-second window
Audio Alarm: Generates or plays an alarm sound when drowsiness is detected
Live Feedback: Displays real-time metrics including EAR values, eye closure duration, and nod count
Visual Alerts: Shows "DROWSY! WAKE UP!" warning in red when drowsiness is detected
# Technology Stack
Python 3.8+: Programming language
OpenCV (cv2): Video capture and image processing
MediaPipe: Facial landmark detection and face mesh
NumPy: Numerical operations and array handling
Pygame: Audio playback for alarm sounds
SciPy: Statistical calculations and distance metrics
# Installation
Prerequisites
Python 3.8 or higher
Webcam/Camera on your device
Windows, macOS, or Linux operating system
Step-by-Step Setup
Clone or Extract the Project

cd driver_drowsiness
Create a Virtual Environment (Recommended)

# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
Install Dependencies

pip install -r requirements.txt
Verify Installation

python -c "import cv2, mediapipe, numpy, pygame; print('All dependencies installed successfully!')"
# How to Run
Basic Usage
python main.py
The application will:

Open a window showing your webcam feed
Display facial landmarks and eye contours in green
Show real-time Eye Aspect Ratio (EAR) values
Track head nods and eye closure duration
Trigger an alarm when drowsiness is detected
Keyboard Controls
q: Quit the application
r: Reset the detector state and stop the alarm
# How It Works
1. Eye Aspect Ratio (EAR)
The system calculates the Eye Aspect Ratio using the formula:

EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
Where p1-p6 are the eye landmark coordinates. Lower EAR values indicate closed eyes.

Threshold: EAR < 0.2 indicates closed eyes
Duration Check: Eyes must stay closed for >2 seconds to trigger alarm
2. Head Nod Detection
The system tracks the vertical position of the nose tip (landmark index 1) to detect head nodding.

Movement Threshold: >15 pixels vertical movement triggers a nod detection
Nod Count: More than 3 nods within 10 seconds triggers alarm
3. Real-Time Processing
The system operates at ~30 FPS and:

Captures frames from the webcam
Detects facial landmarks using MediaPipe
Calculates EAR for both eyes
Tracks nod history within a time window
Determines drowsiness based on dual criteria
Triggers or stops the alarm accordingly
4. Alarm System
Uses Pygame mixer for audio playback
Can play a pre-recorded alarm sound (alarm_sound.wav)
Automatically generates a 1kHz beep tone if no audio file is present
Alarm loops continuously until drowsiness is no longer detected
# File Descriptions
main.py: Contains the main application class that orchestrates video capture, detection, and alarm triggering
detector.py: Implements the DrowsinessDetector class with eye and head movement detection
alarm.py: Implements the DrowsinessAlarm class for sound generation and playback
utils.py: Contains utility functions for EAR calculation, coordinate normalization, and text rendering
# Performance Tuning
Adjusting Sensitivity
You can modify the thresholds in main.py when initializing the detector:

self.detector = DrowsinessDetector(
    ear_threshold=0.2,                    # Lower = more sensitive to closed eyes
    closed_eyes_duration_threshold=2.0,   # Seconds of eye closure before alarm
    nod_count_threshold=3,                # Number of nods to trigger alarm
    nod_time_window=10.0,                 # Time window for counting nods
)
# Common Adjustments
Too many false alarms: Increase ear_threshold to 0.25 or increase closed_eyes_duration_threshold to 2.5
Not detecting drowsiness: Decrease ear_threshold to 0.15 or decrease closed_eyes_duration_threshold to 1.5
Nod detection too sensitive: Increase nod_count_threshold to 4 or decrease nod_threshold in detector.py
# Troubleshooting
## Camera Not Detected
Ensure your webcam is connected and functioning
Try changing the camera index from 0 to 1 in main.py
Check for permission issues (allow camera access in system settings)
## No Landmarks Detected
Ensure adequate lighting (face detection requires clear visibility)
Position your face clearly in the center of the frame
Reduce distance from the camera (optimal: 30-60 cm)
Alarm Not Playing
Check system volume levels
Ensure Pygame mixer initialized successfully (check console output)
Verify pygame installation: pip install --upgrade pygame
# Low FPS Performance
Reduce frame resolution in _initialize_camera()
Close background applications consuming CPU
Try setting static_image_mode=False in detector.py for faster processing
# Limitations and Considerations
Lighting Dependency: Requires adequate lighting for accurate face detection
Single Face: Only detects one face at a time (designed for single driver)
Camera Angle: Works best with front-facing camera
Real-Time Constraints: Performance depends on CPU capabilities
Personal Variation: Optimal thresholds may vary between individuals
# Future Improvements
 Multi-face detection for passenger monitoring
 Machine learning classification for drowsiness patterns
 Mobile/embedded device optimization using TensorFlow Lite
 Integration with vehicle CAN bus for telemetry data
 Cloud-based analytics and reporting
 Notification system (SMS/Email) for remote monitoring
 Driver behavior analytics and fatigue scoring
 Customizable alarm sounds and visual alerts
 Data logging and statistical analysis
 Support for wearable devices (smartwatch integration)
 Deep learning-based eye state classification
 Facial expression analysis for emotion detection
# Methodology & References
## Eye Aspect Ratio
This project is based on the research paper: "Real-Time Eye Blink Detection using Facial Landmarks" by Tereza Soukupová and Jan Čech

##  MediaPipe Face Mesh
Uses Google's MediaPipe framework for efficient facial landmark detection
Tracks 468 3D facial landmarks in real-time

# License
This project is provided for educational purposes.

# Contact & Support
For issues or questions, please ensure:

All dependencies are correctly installed
Camera permissions are granted
Python version is 3.8 or higher
All files are in the correct directory structure
# Disclaimer
This system is designed for educational purposes and driver awareness. It should not be used as the sole safety mechanism in vehicles. Always ensure proper sleep before driving and follow all traffic safety regulations.
