import os
import glob
import numpy as np
import librosa
import joblib
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd



# Audio loading & preprocessing


def load_audio_file(file_path: str, sampling_rate:int) -> tuple[np.ndarray, int]:
    audio_signal, sampling_rate = librosa.load(file_path, mono=True, sr=sampling_rate)
    return audio_signal.astype(np.float32), sampling_rate




def normalize_audio_signal(audio_signal: np.ndarray) -> np.ndarray:
    if audio_signal.size == 0:
        return audio_signal
    return librosa.util.normalize(audio_signal)




def remove_silence_using_energy(
    audio_signal: np.ndarray,
    sampling_rate: int,
    frame_duration_ms: float = 25.0,
    hop_duration_ms: float = 12.5,
    energy_threshold_ratio: float = 0.10
) -> np.ndarray:
    

    frame_length_samples = int(sampling_rate * frame_duration_ms / 1000)
    hop_length_samples = int(sampling_rate * hop_duration_ms / 1000)

    rms_energy = librosa.feature.rms(y=audio_signal, frame_length=frame_length_samples, hop_length=hop_length_samples)[0]
    max_energy = np.max(rms_energy)

    energy_threshold = energy_threshold_ratio * max_energy
    keep_frames = rms_energy > energy_threshold

    sample_mask = np.zeros(len(audio_signal), dtype=bool)

    for frame_idx, keep in enumerate(keep_frames):
        if keep:
            start = frame_idx * hop_length_samples
            end = min(start + frame_length_samples, len(audio_signal))
            sample_mask[start:end] = True

    trimmed_signal = audio_signal[sample_mask]
    return trimmed_signal




def preprocess_audio_signal(
    audio_signal: np.ndarray,
    sampling_rate: int,
    frame_duration_ms: float = 25.0,
    hop_duration_ms: float = 12.5,
    energy_threshold_ratio: float = 0.10
) -> np.ndarray:
    """### preprocess audio file, raise ValueError if audio doesn't have enough energy"""

    if audio_signal.size == 0:
        raise ValueError("audio signal has a size of 0")
    
    frame_length_samples = int(sampling_rate * frame_duration_ms / 1000)
    hop_length_samples = int(sampling_rate * hop_duration_ms / 1000)
    rms_energy = librosa.feature.rms(y=audio_signal, frame_length=frame_length_samples, hop_length=hop_length_samples)[0]
    max_energy = np.max(rms_energy)
    if max_energy == 0:
        raise ValueError("audio with no energy (empty)")
    
    
    energy_threshold = energy_threshold_ratio * max_energy
    keep_frames = rms_energy > energy_threshold
    if not np.any(keep_frames):
        raise ValueError("all audio frames energies are below threshold") 
    
    audio_signal = remove_silence_using_energy(audio_signal, sampling_rate, frame_duration_ms, energy_threshold_ratio=energy_threshold_ratio)
    audio_signal = normalize_audio_signal(audio_signal)
    if audio_signal.size == 0: 
        raise ValueError("energy normalization removed all audio frames")
    else: 
        return audio_signal



# Feature extraction


def extract_features(
    audio_signal: np.ndarray,
    sampling_rate: int,
    frame_ms: float = 25.0,
    hop_ms: float = 12.5,
    n_mfcc: int = 8 #more than this might make it harder to distinguish between women and children
) -> np.ndarray:
    """
    ### Extract MFCC, delta, delta-delta, spectral features, and pitch-based features.
    Raise ValueError if the audio signal is not suitable for feature extraction.
    """


    if audio_signal.size == 0:
        raise ValueError("audio signal has size 0")

    frame_length = int(sampling_rate * frame_ms / 1000.0)
    hop_length = int(sampling_rate * hop_ms / 1000.0)

    if audio_signal.shape[0] < frame_length:
        raise ValueError(
            f"audio too short ({audio_signal.shape[0]} samples) "
            f"for frame length ({frame_length})"
        )

    
    # MFCC + deltas
    
    mfccs = librosa.feature.mfcc(
        y=audio_signal,
        sr=sampling_rate,
        n_mfcc=n_mfcc,
        n_fft=frame_length,
        hop_length=hop_length
    )

    if mfccs.shape[1] == 0:
        raise ValueError("MFCC extraction produced no frames")

    delta = librosa.feature.delta(mfccs, order=1)
    delta2 = librosa.feature.delta(mfccs, order=2)

    
    # Spectral features
    
    centroid = librosa.feature.spectral_centroid(
        y=audio_signal,
        sr=sampling_rate,
        n_fft=frame_length,
        hop_length=hop_length
    )

    bandwidth = librosa.feature.spectral_bandwidth(
        y=audio_signal,
        sr=sampling_rate,
        n_fft=frame_length,
        hop_length=hop_length
    )

    
    # Fundamental frequency (YIN)
    
    f0 = librosa.yin(
        y=audio_signal,
        fmin=50.0,
        fmax=500.0,
        sr=sampling_rate,
        frame_length=max(frame_length, 800),
        hop_length=hop_length
    )

    if f0.size == 0:
        raise ValueError("F0 extraction failed (empty output)")

    voiced_f0 = f0[f0 > 0]

    if voiced_f0.size == 0:
        raise ValueError("no voiced frames detected (cannot compute pitch features)")

    # Pitch statistics
    f0_mean = np.mean(voiced_f0)
    f0_std = np.std(voiced_f0)
    f0_range = np.max(voiced_f0) - np.min(voiced_f0)

    if voiced_f0.size < 2:
        raise ValueError("not enough voiced frames to compute jitter")

    jitter = np.mean(np.abs(np.diff(voiced_f0))) / (f0_mean + 1e-8)

    
    # Shimmer (RMS-based)
    
    rms = librosa.feature.rms(
        y=audio_signal,
        frame_length=frame_length,
        hop_length=hop_length
    ).flatten()

    if rms.size < f0.size:
        raise ValueError("RMS and F0 frame mismatch")

    voiced_rms = rms[:len(f0)][f0 > 0]

    if voiced_rms.size < 2:
        raise ValueError("not enough voiced RMS frames to compute shimmer")

    shimmer = np.mean(np.abs(np.diff(voiced_rms))) / (np.mean(voiced_rms) + 1e-8)

    
    # Formant estimation (LPC)
    
    lpc_order = int(2 + sampling_rate / 1000)
    lpc_coeffs = librosa.lpc(y=audio_signal, order=lpc_order)

    if np.any(np.isnan(lpc_coeffs)) or np.any(np.isinf(lpc_coeffs)):
        raise ValueError("LPC coefficients are unstable")

    roots = np.roots(lpc_coeffs)
    roots = roots[np.imag(roots) >= 0]

    if roots.size < 2:
        raise ValueError("not enough LPC roots to estimate formants")

    angles = np.angle(roots)
    formants = np.sort(angles * (sampling_rate / (2 * np.pi)))

    if len(formants) < 2:
        raise ValueError("could not estimate F1 and F2")

    f1_f2_ratio = formants[0] / (formants[1] + 1e-8)

    
    # Aggregate frame-wise features
    
    frame_features = np.vstack([
        mfccs,
        delta,
        delta2,
        centroid,
        bandwidth,
        f0[np.newaxis, :]
    ]).astype(np.float32)

    mean_features = frame_features.mean(axis=1)
    std_features = frame_features.std(axis=1)

    
    # Final feature vector
    
    return np.concatenate([
        mean_features,
        std_features,
        np.array([
            f0_mean,
            f0_std,
            f0_range,
            jitter,
            shimmer,
            f1_f2_ratio
        ], dtype=np.float32)
    ])






def split_audio_into_segments(
    audio_signal: np.ndarray,
    sampling_rate: int,
    segment_duration_s = 0.625,   
    hop_duration_s = 0.625   
) -> list[np.ndarray]:
    """
    ### Split audio into overlapping fixed-length segments.
    Raise ValueError if the audio is not suitable for segmentation.
    """

    
    # Basic validity checks
    
    if audio_signal.size == 0:
        raise ValueError("audio signal has size 0")

    if segment_duration_s <= 0:
        raise ValueError("segment_duration_s must be > 0")

    if hop_duration_s <= 0:
        raise ValueError("hop_duration_s must be > 0")

    if hop_duration_s > segment_duration_s:
        raise ValueError("hop_duration_s cannot be larger than segment_duration_s")

    seg_len = int(segment_duration_s * sampling_rate)
    hop_len = int(hop_duration_s * sampling_rate)

    if seg_len <= 0 or hop_len <= 0:
        raise ValueError("invalid segment or hop length after conversion to samples")

    
    # Length requirement
    
    if audio_signal.shape[0] < seg_len:
        raise ValueError(
            f"audio too short ({audio_signal.shape[0]} samples) "
            f"for one segment ({seg_len} samples)"
        )

    
    # Segment extraction
    
    segments = [
        audio_signal[i:i + seg_len]
        for i in range(0, audio_signal.shape[0] - seg_len + 1, hop_len)
    ]

    if len(segments) == 0:
        raise ValueError("segmentation produced zero segments")

    return segments




def list_audio_files(folder: str) -> list[str]:
    exts = ("*.wav", "*.mp3", "*.flac", "*.m4a", "*.ogg")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
    return sorted(files)





def process_single_file(csv_row, audio_dir, sampling_rate):

    """### build a vector from an audio that is described in the csv containg filenames and their label
    raise ValueError catched from the preprocessing functions in case of invalid audio file
    raise FileNotfoundError if a file mentioned in the CSV is not found"""
    filename, label = csv_row["filename"], csv_row["label"]
    path = os.path.join(audio_dir, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"file specified in row not found: {filename}")

    try:
        audio_signal, sr = load_audio_file(path, sampling_rate)
        audio_signal = preprocess_audio_signal(audio_signal, sr)
        segments = split_audio_into_segments(audio_signal, sr)
        features = [extract_features(segment, sr) for segment in segments]
        # code adults as 0 and children as 1 
        labels = [0] * len(features) if label == "adult" else [1] * len(features)
        return features, labels
    
    except ValueError: #catch the defined Value error in the preprocessing functions to elimine bad audios
        raise 
    except Exception: 
        raise





def build_classification_dataset(data_folder: str, sampling_rate: int, max_workers: int = None):
    """ ### use audios and the labeling CSV file to build vectors dataset.
      **Note:** you must name the audios folder 'audios/' and the csv file 'labeled-files' """
    
    audio_dir = os.path.join(data_folder, "audios")
    if not os.path.exists(audio_dir): 
        raise FileNotFoundError(f"make sure audios folder is named 'audios/' and is inside '{data_folder}'")
    
    csv_path = os.path.join(data_folder, "labeled-files.csv")
    try: 
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"make sure labeling CSV is named 'labeled-files' and is inside {data_folder}")
    
    if not {"filename", "label"}.issubset(df.columns):
        raise ValueError("labeling CSV must contain 'filename' column for audios filenames, and 'label' column with 'adult' and 'child' labels")

    features, labels = [], []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        #in case of a ValueError when processing a file, it's re-raised when calling result()
        # so there we can ignore non valid audio files
        futures = [executor.submit(process_single_file, row, audio_dir, sampling_rate)
                    for _, row in tqdm(df.iterrows(), desc="submitting audio files to preprocessing")]
        
        print("\nbuilding dataset...")
        skipped_audios = 0
        for future in as_completed(futures):
            try: 
                audio_features, audio_label = future.result()
                features.extend(audio_features)
                labels.extend(audio_label)
            except ValueError as ve: 
                print(f"audio file skipped for defined preprocessing error: {ve}")
                skipped_audios += 1
            except FileNotFoundError as fe: 
                print(f"audio file skipped for defined preprocessing error: {fe}")
                skipped_audios += 1
            
            except Exception as e: 
                print(f"audio file skipped for unexpected error: {e}")
                skipped_audios += 1


    if len(features) == 0:
        raise RuntimeError("resulted dataset is empty for some unknown reason.")
    
    print("skipped audios: ", skipped_audios)
    print("vectorized audios: ", len(features))
    
    X = np.vstack(features)
    y = np.array(labels)

    saving_path = os.path.join(data_folder, "dataset.joblib")
    joblib.dump((X, y), saving_path, compress=3)
    print(f"dataset saved to {saving_path}")




if __name__ == "__main__": 
    try: 
        build_classification_dataset("data", 16000)
    except KeyboardInterrupt: 
        print("process terminated")
        exit(0)
        