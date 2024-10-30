# src/animation_synthesis.py

import numpy as np
import cv2

def merge_animations(body_frames, facial_frames, alpha=0.5):
    """
    Merge body and facial animation frames.
    Assumes both body_frames and facial_frames are lists of numpy arrays/images.
    """
    combined_frames = []
    for body, face in zip(body_frames, facial_frames):
        # Resize face frame to match body frame if necessary
        if face.shape != body.shape:
            face = cv2.resize(face, (body.shape[1], body.shape[0]))
        
        # Blend images
        combined = cv2.addWeighted(body, 1 - alpha, face, alpha, 0)
        combined_frames.append(combined)
    return combined_frames
