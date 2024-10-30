# src/main.py

import os
import logging
from input_handler import load_image, load_audio, preprocess_image, preprocess_audio
from audio_processing import extract_audio_features, analyze_prosody
from motion_generation import MotionGenerator
from character_rigging import setup_scene
from animation_synthesis import prepare_pose_sequence
from renderer import display_frames, save_video
import subprocess

def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Paths
    source_image_path = 'data/input_images/source.jpg'
    audio_path = 'data/input_audio/input.wav'
    audio2pose_checkpoint_path = 'models/audio2pose/audio2pose.pth'  # Replace with actual path
    character_model_path = 'models/character_model/character.fbx'  # Replace with actual path
    character_object_name = 'Character'  # Replace with your character's object name in Blender
    blender_apply_motion_script = 'blender_scripts/apply_motion.py'
    blender_render_script = 'blender_scripts/render_animation.py'
    output_video_path = 'data/output/animation.mp4'
    pose_sequence_path = 'data/output/pose_sequence.npy'
    
    # Ensure output directory exists
    os.makedirs('data/output', exist_ok=True)
    
    try:
        # Step 1: Input Handling
        logging.info("Loading and preprocessing source image...")
        source_image = load_image(source_image_path)
        source_image = preprocess_image(source_image)
        
        logging.info("Loading and preprocessing audio...")
        audio, sr = load_audio(audio_path)
        audio = preprocess_audio(audio, sr)
        
        # Step 2: Audio Processing
        logging.info("Extracting audio features...")
        audio_features = extract_audio_features(audio, sr)
        logging.info("Analyzing prosody...")
        avg_pitch = analyze_prosody(audio, sr)
        logging.info(f"Average Pitch: {avg_pitch}")
        
        # Step 3: Motion Generation
        logging.info("Generating motion from audio features...")
        motion_generator = MotionGenerator(audio2pose_checkpoint_path)
        pose_sequence = motion_generator.generate_motion(audio_features)
        motion_generator.close()
        
        # Step 4: Prepare Pose Sequence
        logging.info("Preparing pose sequence for animation...")
        pose_sequence_prepared = prepare_pose_sequence(pose_sequence)
        np.save(pose_sequence_path, pose_sequence_prepared)
        
        # Step 5: Character Rigging and Animation in Blender
        logging.info("Setting up Blender scene with character model and texture...")
        # Note: Character rigging and texture application are handled within Blender scripts
        # You need to run Blender scripts separately or automate their execution
        
        # Apply Motion using Blender
        logging.info("Applying motion to character in Blender...")
        blender_apply_motion_cmd = [
            'blender',  # Ensure Blender is in your PATH
            '--background',  # Run in background mode
            '--python', blender_apply_motion_script,
            '--',
            pose_sequence_path,
            character_object_name
        ]
        subprocess.run(blender_apply_motion_cmd, check=True)
        
        # Render Animation using Blender
        logging.info("Rendering animation in Blender...")
        blender_render_cmd = [
            'blender',
            '--background',
            '--python', blender_render_script,
            '--',
            output_video_path
        ]
        subprocess.run(blender_render_cmd, check=True)
        
        logging.info(f"Animation successfully saved to {output_video_path}")
    
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
