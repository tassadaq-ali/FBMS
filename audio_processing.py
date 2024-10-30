# src/audio_processing.py

import librosa
import numpy as np

def extract_audio_features(audio, sample_rate):
    """
    Extract relevant audio features for synchronization.
    """
    # Extract Mel-frequency cepstral coefficients (MFCCs)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

def analyze_prosody(audio, sample_rate):
    """
    Analyze prosody features like pitch, rhythm, and stress.
    """
    # Extract pitch (fundamental frequency)
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sample_rate)
    pitch = []
    for i in range(pitches.shape[1]):
        index = magnitudes[:, i].argmax()
        pitch_value = pitches[index, i]
        if pitch_value > 0:
            pitch.append(pitch_value)
    pitch = np.array(pitch)
    avg_pitch = np.mean(pitch) if len(pitch) > 0 else 0
    return avg_pitch
