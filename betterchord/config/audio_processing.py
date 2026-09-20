import librosa
import numpy as np
import warnings
import os

# spec_data lives in data along with training_data and test_data
SPEC_CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'spec_data')

# Dataset-level normalization statistics
# Computed by compute_dataset_stats.py across all 50k+ training files
# Used instead of per-sample z-score to ensure consistent feature space
# across all recordings regardless of sparsity or recording environment
DATASET_MEAN = -38.3493
DATASET_STD  = 22.8682

# Noise bank lives in data/noise_bank/processed
NOISE_BANK_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'noise_bank', 'processed')

# Noise bank cache, loaded into memory once if not existing
_noise_bank = None

def find_chord_position_time(y, sr):
    # This is a function to find in what section of the audio clip the chord is being played
    # y: Audio Signal
    # sr: Sample Rate

    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)

    if len(onset_frames) == 0:
        warnings.warn("No onsets detected. Using beginning of file. (This may lead to inaccurate chord detection, you actually strum?)")
        onset_time = 0.10
    else:
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        onset_strength = librosa.onset.onset_strength(y=y, sr=sr)
        onset_strength_at_onsets = onset_strength[onset_frames]
        max_strength_index = np.argmax(onset_strength_at_onsets)
        onset_time = onset_times[max_strength_index]

    # Pre-window tightened to 100ms before onset
    start_time = max(0, onset_time - 0.10)
    end_time   = min(len(y) / sr, onset_time + 2.5)

    return start_time, end_time


def get_clip_ratio(y):
    # Fraction of samples near digital ceiling - the actual definition of clipping
    # Clean recording: ~0.001  |  Mild clipping: ~0.005  |  Heavy clipping: 0.01+
    return float(np.mean(np.abs(y) > 0.98))


def load_audio(file_path, sr=22050, target_duration=2.75):
    # Load audio file
    # file_path: Path to the audio file
    # sr: Sample rate to load the audio at (default is 22050)

    y_full, sr = librosa.load(file_path, sr=sr, duration=60)
    start_time, end_time = find_chord_position_time(y_full, sr)
    y_full = y_full[int(start_time * sr):int(end_time * sr)]

    if len(y_full) < int(target_duration * sr):
        y_full = librosa.util.fix_length(y_full, size=int(target_duration * sr))

    # Step 1: Adaptive soft clipping based on clip ratio
    # Measures how clipped the signal is, applies stronger compression for worse recordings
    # Continuous mapping: clean recordings get light touch, clipped get aggressive compression
    clip_ratio = get_clip_ratio(y_full)
    percentile  = max(90.0, 99.5 - clip_ratio * 950)
    peak        = np.percentile(np.abs(y_full), percentile)
    if peak > 0:
        y_full = np.tanh(y_full / peak)

    # Step 2: RMS normalize to consistent energy level across all recording environments
    rms = np.sqrt(np.mean(y_full ** 2))
    if rms > 0:
        y_full = y_full * (0.1 / rms)

    return y_full, sr


def create_spectrogram_unnormalized(y, sr, n_bins=84, bins_per_octave=12, hop_length=512):
    # Same as create_spectrogram but returns raw dB values without z-score
    # Used by compute_dataset_stats.py to calculate dataset-level mean and std, otherwise not used in training or on run
    My_CQT_Spec = librosa.cqt(y=y, sr=sr, n_bins=n_bins, bins_per_octave=bins_per_octave,
                               hop_length=hop_length, fmin=librosa.note_to_hz('E2'))
    cqt_mag = np.abs(My_CQT_Spec)
    cqt_mag_clipped = cqt_mag.copy()
    for b in range(min(10, cqt_mag_clipped.shape[0])):
        sustained = cqt_mag_clipped[b, 6:]
        threshold = np.percentile(sustained, 99)
        x = cqt_mag_clipped[b, :6]
        cqt_mag_clipped[b, :6] = threshold * np.tanh(x / threshold)
    return librosa.amplitude_to_db(cqt_mag_clipped, ref=1.0)


def create_spectrogram(y, sr, n_bins=84, bins_per_octave=12, hop_length=512):
    # Create CQT spectrogram
    # y = audio signal
    # sr = sample rate
    # n_bins = number of CQT bins (84 = 7 octaves x 12 bins/octave)
    # bins_per_octave = 12 means exactly 1 bin per semitone
    # hop_length = hop length for CQT

    # fmin=E2 (82Hz) = lowest guitar note, removes sub-bass mic rumble and room noise
    My_CQT_Spec = librosa.cqt(y=y, sr=sr, n_bins=n_bins, bins_per_octave=bins_per_octave,
                               hop_length=hop_length, fmin=librosa.note_to_hz('E2'))

    cqt_mag = np.abs(My_CQT_Spec)

    # Targeted onset transient suppression in low-frequency bins
    # Diagnostic showed: spikes occur only in bins 0-10 (E2-B2) at frames 3-4
    # These are pick attack transients, not sustained chord information
    # Clip is localized to BOTH dimensions - only where spikes actually occur
    # Sustained note energy (frames 6+) and mid/high bins left completely untouched
    # Targeted onset transient suppression in low-frequency bins
    # Spikes occur in bins 0-10 at frames 3-4 (pick attack transients)
    # Threshold derived from SUSTAINED region (frames 6+) of each bin
    # so it adapts per recording and per frequency - no fixed arbitrary values
    cqt_mag_clipped = cqt_mag.copy()
    low_bins     = 10
    onset_frames = 6

    for b in range(low_bins):
        sustained = cqt_mag_clipped[b, onset_frames:]  # clean reference region

        # Use p99 of sustained region as threshold
        # No floor guard, sustained threshold is already adaptive
        # If sustained energy is low, onset spike should also be low
        threshold = np.percentile(sustained, 99)

        # Soft tanh compression, preserves ordering and rid of outliers without hard clips
        x = cqt_mag_clipped[b, :onset_frames]
        cqt_mag_clipped[b, :onset_frames] = threshold * np.tanh(x / threshold)

    # ref=1.0 gives consistent absolute dB values
    # RMS normalization + targeted transient suppression ensure stable input
    cqt_db = librosa.amplitude_to_db(cqt_mag_clipped, ref=1.0)

    # Dataset-level normalization using precomputed statistics
    # More stable than per-sample z-score for sparse spectrograms
    # (high position voicings with fewer strings get inflated z-scores per-sample)
    cqt_normalized = (cqt_db - DATASET_MEAN) / DATASET_STD

    # Post-normalization winsorization
    cqt_normalized = np.clip(cqt_normalized, -3.0, 3.0)

    return cqt_normalized


def load_or_cache_spectrogram(file_path):
    # Load spectrograms from cache if it exists, otherwise computes it and saves it
    # Cache lives in data/spec_data

    data_dir  = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
    file_norm = os.path.normpath(file_path)
    rel_path  = os.path.relpath(file_norm, data_dir)
    cache_path = os.path.join(SPEC_CACHE_DIR, os.path.splitext(rel_path)[0] + '.npy')

    if os.path.exists(cache_path):
        return np.load(cache_path)

    y, sr       = load_audio(file_path)
    spectrogram = create_spectrogram(y, sr)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, spectrogram)

    return spectrogram


def load_noise_bank():
    # Load all noise npy into memory once

    global _noise_bank

    if _noise_bank is not None:
        return _noise_bank

    if not os.path.exists(NOISE_BANK_DIR):
        _noise_bank = []
        return _noise_bank

    noise_files = sorted([f for f in os.listdir(NOISE_BANK_DIR) if f.endswith('.npy')])
    _noise_bank = [np.load(os.path.join(NOISE_BANK_DIR, f)) for f in noise_files]
    return _noise_bank


def apply_noise_augmentation(spectrogram):
    # Mix a random noise slice from the noise bank into the spectrogram.
    # Returns augmented spectrogram same shape as input.

    noise_bank = load_noise_bank()
    if not noise_bank:
        return spectrogram

    noise_array = noise_bank[np.random.randint(len(noise_bank))]

    spec_width  = spectrogram.shape[1]
    noise_width = noise_array.shape[1]

    if noise_width <= spec_width:
        repeats     = (spec_width // noise_width) + 1
        noise_slice = np.tile(noise_array, (1, repeats))[:, :spec_width]
    else:
        max_start   = noise_width - spec_width
        start       = np.random.randint(0, max_start)
        noise_slice = noise_array[:, start:start + spec_width]

    # SNR targeting using linear power ratio
    roll = np.random.rand()
    if roll < 0.20:
        snr_db = np.random.uniform(18, 25)   # Heavy noise
    elif roll < 0.80:
        snr_db = np.random.uniform(25, 35)   # Moderate noise
    else:
        snr_db = np.random.uniform(40, 50)   # Light noise

    signal_rms = np.sqrt(np.mean(spectrogram ** 2))
    noise_rms  = np.sqrt(np.mean(noise_slice ** 2))

    if noise_rms > 0 and signal_rms > 0:
        target_noise_rms = signal_rms / (10 ** (snr_db / 20.0))
        noise_slice      = noise_slice * (target_noise_rms / noise_rms)

    return spectrogram + noise_slice