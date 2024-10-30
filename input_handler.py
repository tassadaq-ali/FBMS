# src/input_handler.py

import cv2
import librosa
import numpy as np
from pydub import AudioSegment
import os

def load_image(image_path):
    """
    Load an image from the specified path and convert it to RGB.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb

def load_audio(audio_path, sr=22050):
    """
    Load an audio file and return the audio signal and sample rate.
    """
    audio, sample_rate = librosa.load(audio_path, sr=sr)
    return audio, sample_rate

def preprocess_image(image, target_size=(256, 256)):
    """
    Resize and normalize the image.
    """
    image_resized = cv2.resize(image, target_size)
    image_normalized = image_resized / 255.0  # Normalize to [0,1]
    return image_normalized

def preprocess_audio(audio, sample_rate, target_length=10):
    """
    Trim or pad the audio to the target length in seconds.
    """
    target_length_samples = target_length * sample_rate
    if len(audio) > target_length_samples:
        audio = audio[:target_length_samples]
    else:
        padding = target_length_samples - len(audio)
        audio = np.pad(audio, (0, padding), 'constant')
    return audio

def load_video_frames(video_path):
    """
    Load video frames from the specified path.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    return frames
