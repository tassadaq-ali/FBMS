# src/renderer.py

import cv2
import os

def display_frames(frames, window_name='Animation'):
    """
    Display a sequence of frames in a window.
    """
    for frame in frames:
        cv2.imshow(window_name, frame)
        if cv2.waitKey(30) & 0xFF == 27:  # Exit on ESC
            break
    cv2.destroyAllWindows()

def save_video(frames, output_path, fps=30):
    """
    Save a sequence of frames as a video file.
    """
    if not frames:
        raise ValueError("No frames to save.")
    
    height, width, layers = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can choose other codecs
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        # Ensure frame is in BGR format
        if frame.shape[2] == 3:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame
        out.write(frame_bgr)
    
    out.release()
