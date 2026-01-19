import joblib
import numpy as np
import os
from pathlib import Path
from preprocess import load_audio_file, preprocess_audio_signal, split_audio_into_segments, extract_features

# --------------------------
# Load model, threshold, feature mask
# --------------------------
model_data = joblib.load("model/child_detector_calibrated.joblib")
model = model_data["model"]          # Calibrated XGBClassifier
threshold = model_data["threshold"]  # Optimized probability threshold
sampling_rate = 16000                # Must match training

# Load boolean mask of features selected by RFECV
features_mask = joblib.load("model/features_mask.joblib")  # boolean mask or integer indices
num_selected_features = np.sum(features_mask)

# --------------------------
# Predict segment-level probability
# --------------------------
def predict_child_or_adult(feature_vector: np.ndarray) -> float:
    """
    Predict probability of being child (1) given a 1D feature vector.
    Returns probability (float between 0 and 1).
    """
    if not isinstance(feature_vector, np.ndarray):
        raise TypeError("feature_vector must be a NumPy ndarray")
    if feature_vector.ndim != 1:
        raise ValueError("feature_vector must be 1D")
    if len(feature_vector) != num_selected_features:
        raise ValueError(f"feature_vector length must be {num_selected_features}")

    X_input = feature_vector.reshape(1, -1)
    proba = model.predict_proba(X_input)[:, 1][0]  # probability of child
    return proba

# --------------------------
# Convert audio to model-ready vectors
# --------------------------
def audio_to_model_vector(audio_path: str) -> list:
    """
    Convert an audio file into a list of segment vectors ready for the model.
    Each segment is a 1D array of selected features.
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    audio_signal, sr = load_audio_file(audio_path, sampling_rate)
    audio_signal = preprocess_audio_signal(audio_signal, sr)
    segments = split_audio_into_segments(audio_signal, sr)

    # Extract features per segment and select only model features
    segments_vecs = []
    for segment in segments:
        seg_feat = extract_features(segment, sr)
        if isinstance(seg_feat, np.ndarray):
            selected_vec = seg_feat[features_mask]  # boolean mask indexing
        else:
            # If extract_features returns dict or Series
            selected_vec = np.array([seg_feat[i] for i, m in enumerate(features_mask) if m])
        segments_vecs.append(selected_vec)
    
    return segments_vecs

# --------------------------
# Predict on a full audio file
# --------------------------
def predict_audio(audio_path: str, aggregation: str = "mean") -> tuple:
    """
    Predict child/adult from a full audio file.

    Parameters:
    - audio_path: path to audio file
    - aggregation: "mean" or "median" of segment probabilities

    Returns:
    - label: str, "child" or "adult"
    - probability: float (0-1) for child
    """
    segments_vecs = audio_to_model_vector(audio_path)
    segs_probabilities = [predict_child_or_adult(seg) for seg in segments_vecs]

    if aggregation == "mean":
        audio_proba = np.mean(segs_probabilities)
    elif aggregation == "median":
        audio_proba = np.median(segs_probabilities)
    else:
        raise ValueError("aggregation must be 'mean' or 'median'")

    label = "child" if audio_proba >= threshold else "adult"
    return label, audio_proba

# --------------------------
# CLI usage
# --------------------------
if __name__ == "__main__":
    path = Path(input("Enter audio path: "))
    label, proba = predict_audio(path)
    print(f"Probability of being a child: {proba*100:.2f}%")
    print(f"Predicted label: {label}")
