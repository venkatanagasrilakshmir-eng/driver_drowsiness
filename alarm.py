"""
Alarm module for triggering drowsiness alerts.
This module handles audio playback of alarm sounds when drowsiness is detected.
"""

import os
import numpy as np
import pygame
from pathlib import Path


class DrowsinessAlarm:
    """
    Handles alarm sound generation and playback.
    Can either play a pre-recorded alarm sound or generate a beep dynamically.
    """
    
    def __init__(self, alarm_file_path="alarm_sound.wav"):
        """
        Initialize the alarm system.
        
        Args:
            alarm_file_path: Path to the alarm sound file (optional)
        """
        self.alarm_file_path = alarm_file_path
        self.is_playing = False
        self.sound = None
        
        # Initialize pygame mixer for audio
        try:
            pygame.mixer.init()
            print("[INFO] Pygame mixer initialized successfully.")
        except Exception as e:
            print(f"[WARNING] Failed to initialize pygame mixer: {e}")
            print("[WARNING] Alarm sounds will not work without initialization.")
    
    def _generate_beep_sound(self, frequency=1000, duration=1.0, sample_rate=44100):
        """
        Generate a simple beep sound using sine wave.
        
        Args:
            frequency: Frequency of the beep in Hz (default 1000 Hz)
            duration: Duration of the beep in seconds (default 1 second)
            sample_rate: Sample rate in Hz (default 44100)
            
        Returns:
            pygame.mixer.Sound: The generated sound object
        """
        # Generate time array
        num_samples = int(sample_rate * duration)
        t = np.linspace(0, duration, num_samples)
        
        # Generate sine wave
        frequency = 2 * np.pi * frequency
        wave = np.sin(frequency * t)
        
        # Convert to 16-bit audio
        wave = (wave * 32767).astype(np.int16)
        
        # Create stereo sound (left and right channels)
        stereo_wave = np.zeros((num_samples, 2), dtype=np.int16)
        stereo_wave[:, 0] = wave
        stereo_wave[:, 1] = wave
        
        # Create pygame Sound object
        sound = pygame.sndarray.make_sound(stereo_wave)
        
        return sound
    
    def _load_alarm_sound(self):
        """
        Load the alarm sound from file.
        If file doesn't exist, generate a beep sound dynamically.
        
        Returns:
            pygame.mixer.Sound: The alarm sound object
        """
        # Check if alarm file exists
        if os.path.exists(self.alarm_file_path):
            try:
                sound = pygame.mixer.Sound(self.alarm_file_path)
                print(f"[INFO] Loaded alarm sound from: {self.alarm_file_path}")
                return sound
            except Exception as e:
                print(f"[WARNING] Failed to load alarm sound: {e}")
        
        # Generate beep sound if file doesn't exist
        print("[INFO] Generating dynamic beep sound...")
        sound = self._generate_beep_sound()
        
        return sound
    
    def trigger_alarm(self, loops=-1):
        """
        Trigger the alarm sound (start playing).
        
        Args:
            loops: Number of times to loop (-1 for infinite loop, 0 for play once)
        """
        try:
            if not self.is_playing:
                # Load the alarm sound
                if self.sound is None:
                    self.sound = self._load_alarm_sound()
                
                # Play the sound
                self.sound.play(loops=loops)
                self.is_playing = True
                print("[ALARM] Drowsiness detected! Alarm triggered!")
        
        except Exception as e:
            print(f"[ERROR] Failed to trigger alarm: {e}")
    
    def stop_alarm(self):
        """
        Stop the alarm sound.
        """
        try:
            if self.is_playing:
                pygame.mixer.stop()
                self.is_playing = False
                print("[INFO] Alarm stopped.")
        
        except Exception as e:
            print(f"[ERROR] Failed to stop alarm: {e}")
    
    def is_alarm_playing(self):
        """
        Check if the alarm is currently playing.
        
        Returns:
            bool: True if alarm is playing, False otherwise
        """
        return self.is_playing and pygame.mixer.get_busy()
