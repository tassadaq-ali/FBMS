# src/motion_generation.py

import torch
import numpy as np
from models.audio2pose.audio2pose_model import Audio2PoseModel  # Adjust the import path based on Audio2Pose repository
import os

class MotionGenerator:
    def __init__(self, audio2pose_checkpoint_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Initialize Audio2Pose Model
        self.audio2pose = Audio2PoseModel()
        self.audio2pose.load_state_dict(torch.load(audio2pose_checkpoint_path, map_location=self.device))
        self.audio2pose.to(self.device)
        self.audio2pose.eval()
    
    def generate_motion(self, audio_features):
        """
        Generate motion (pose sequences) from audio features.
        """
        with torch.no_grad():
            audio_tensor = torch.tensor(audio_features).float().unsqueeze(0).to(self.device)  # Shape: [1, feature_dim]
            pose_sequence = self.audio2pose(audio_tensor)  # Assuming output shape [1, seq_len, joints*3]
            pose_sequence = pose_sequence.squeeze(0).cpu().numpy()  # Shape: [seq_len, joints*3]
        return pose_sequence
    
    def close(self):
        self.audio2pose.cpu()
        torch.cuda.empty_cache()
