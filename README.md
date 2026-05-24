# Driver Drowsiness Detection System

## Overview

The Driver Drowsiness Detection System is a real-time computer vision application designed to monitor and detect signs of driver fatigue through advanced facial analysis. By leveraging facial landmark detection and eye movement tracking, the system identifies drowsiness indicators and provides immediate alerts to prevent potential accidents.

This project demonstrates the practical application of computer vision techniques, including facial landmark detection, real-time signal processing, and behavioral analysis.

## Key Features

- **Real-Time Face Detection**: Utilizes MediaPipe Face Mesh to detect faces and extract 468 facial landmarks with high precision
- **Eye Aspect Ratio (EAR) Analysis**: Monitors eye closure duration through mathematical analysis of facial landmarks
- **Head Movement Tracking**: Detects repetitive head nodding patterns associated with drowsiness
- **Dual Detection Criteria**:
  - Continuous eye closure exceeding 2 seconds
  - More than 3 head nods within a 10-second interval
- **Audio Alert System**: Provides immediate audio notification when drowsiness is detected
- **Real-Time Metrics Display**: Shows live Eye Aspect Ratio, eye closure duration, and head nod count
- **Visual Warning System**: Displays prominent "DROWSY! WAKE UP!" alert in high-visibility red

## Technology Stack

| Component | Purpose |
|-----------|---------|
| **Python 3.8+** | Core programming language |
| **OpenCV (cv2)** | Video capture and image processing |
| **MediaPipe** | Facial landmark detection |
| **NumPy** | Numerical computing and array operations |
| **Pygame** | Audio playback and alarm management |
| **SciPy** | Statistical calculations and distance metrics |

## System Requirements

### Prerequisites

- Python 3.8 or higher
- Functional webcam or camera device
- Compatible operating system: Windows, macOS, or Linux
- Minimum 4GB RAM recommended
- Stable CPU with multi-threading capability

## Installation Guide

### 1. Clone or Extract the Project

```bash
cd driver_drowsiness
```

### 2. Create a Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import cv2, mediapipe, numpy, pygame; print('All dependencies installed successfully!')"
```

## Usage

### Running the Application

```bash
python main.py
```

**Expected Behavior:**
- Webcam feed window displays facial landmarks in green
- Real-time Eye Aspect Ratio (EAR) values are shown
- Head nod count and eye closure duration are tracked
- Audio alarm triggers upon drowsiness detection

### Keyboard Controls

| Key | Function |
|-----|----------|
| `q` | Quit application |
| `r` | Reset detector state and stop alarm |

## Technical Implementation

### 1. Eye Aspect Ratio (EAR) Calculation

The Eye Aspect Ratio is computed using the following formula:

```
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
```

Where p1-p6 represent the eye landmark coordinates from MediaPipe Face Mesh.

**Detection Thresholds:**
- EAR < 0.2 indicates closed eyes
- Eyes must remain closed for >2 seconds to trigger alarm

### 2. Head Nod Detection

The system monitors the vertical position of the nose tip (landmark index 1) to identify head nodding patterns.

**Detection Parameters:**
- Vertical movement threshold: >15 pixels
- Nod detection trigger: More than 3 nods within 10 seconds

### 3. Real-Time Processing Pipeline

The system operates at approximately 30 FPS with the following workflow:

1. Capture video frame from webcam
2. Detect facial landmarks using MediaPipe
3. Calculate EAR for both eyes
4. Track nod history within time window
5. Evaluate drowsiness criteria
6. Trigger or deactivate alarm as needed

### 4. Alarm System

- **Audio Engine**: Pygame mixer for reliable audio playback
- **Default Behavior**: Plays pre-recorded alarm sound (alarm_sound.wav) if available
- **Fallback**: Generates 1kHz beep tone if audio file is unavailable
- **Operation**: Continuous alarm loop until drowsiness is no longer detected

## Project Structure

```
driver_drowsiness/
├── main.py              # Application orchestration and video capture
├── detector.py          # DrowsinessDetector class implementation
├── alarm.py             # DrowsinessAlarm class and audio management
├── utils.py             # Utility functions for EAR calculation and rendering
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Configuration & Performance Tuning

### Adjusting Detection Sensitivity

Modify the following parameters in `main.py` when initializing the detector:

```python
self.detector = DrowsinessDetector(
    ear_threshold=0.2,                    # Lower = more sensitive to eye closure
    closed_eyes_duration_threshold=2.0,   # Seconds before alarm trigger
    nod_count_threshold=3,                # Number of nods required for alarm
    nod_time_window=10.0,                 # Detection time window in seconds
)
```

### Common Tuning Scenarios

| Scenario | Solution |
|----------|----------|
| Excessive false alarms | Increase `ear_threshold` to 0.25; Increase `closed_eyes_duration_threshold` to 2.5 |
| Missed drowsiness detection | Decrease `ear_threshold` to 0.15; Decrease `closed_eyes_duration_threshold` to 1.5 |
| Oversensitive nod detection | Increase `nod_count_threshold` to 4; Decrease nod threshold in detector.py |

## Troubleshooting Guide

### Camera Not Detected

- Verify webcam is properly connected
- Try alternative camera indices (change 0 to 1 in main.py)
- Check system settings for camera permission access
- Restart the application after granting permissions

### No Facial Landmarks Detected

- Ensure adequate lighting in the environment
- Position face centrally in camera frame
- Maintain optimal distance from camera (30-60 cm recommended)
- Verify camera lens is clean and unobstructed

### Audio Alarm Not Playing

- Check system volume settings
- Verify Pygame mixer initialization in console output
- Upgrade Pygame: `pip install --upgrade pygame`
- Test audio with system sound settings

### Low Frame Rate Performance

- Reduce video capture resolution in `_initialize_camera()`
- Close background applications consuming CPU resources
- Set `static_image_mode=False` in detector.py for improved speed
- Consider hardware acceleration if available

## Limitations

- **Lighting Dependency**: Requires adequate ambient lighting for reliable face detection
- **Single Subject**: Designed for single-driver monitoring only
- **Camera Orientation**: Optimized for front-facing camera positioning
- **Real-Time Constraints**: Performance varies based on CPU capabilities
- **Individual Variation**: Optimal thresholds may require user-specific calibration

## Methodology & References

### Eye Aspect Ratio Algorithm

Based on the research paper: *"Real-Time Eye Blink Detection using Facial Landmarks"* by Tereza Soukupová and Jan Čech

### Facial Landmark Detection

Utilizes Google's MediaPipe framework for efficient real-time detection of 468 3D facial landmarks with multi-platform support.

## Planned Enhancements

- Multi-face detection for passenger monitoring
- Machine learning classification for advanced drowsiness pattern recognition
- Mobile and embedded device optimization using TensorFlow Lite
- Vehicle CAN bus integration for telemetry correlation
- Cloud-based analytics platform
- Remote notification system (SMS/Email alerts)
- Advanced driver behavior analytics
- Customizable alarm configurations
- Historical data logging and statistical reporting
- Wearable device integration
- Deep learning-based eye state classification
- Facial expression analysis for emotion detection

## License

This project is provided for educational and research purposes.

## Support & Contact

For technical support or inquiries:

1. Verify all dependencies are correctly installed
2. Confirm camera permissions are granted at system level
3. Ensure Python version is 3.8 or higher
4. Validate correct file directory structure

## Disclaimer

This system is developed for educational purposes and driver awareness enhancement only. **It should not be relied upon as the sole safety mechanism in vehicles.** Drivers must ensure adequate rest and follow all applicable traffic regulations. The developers are not liable for accidents, injuries, or damages resulting from system misuse or failure.

---

*Last Updated: 2026*
