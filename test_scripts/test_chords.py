"""
Chord Test Script
Automatically discovers and tests all wav files in the betterchord directory.

Naming convention for test files:
    <ChordName>_<tab_with_underscores>.wav
    e.g. D9_X_5_4_5_5_X.wav
         Cmaj_X_3_2_0_1_0.wav
         F#maj9_2_1_3_1_2_1.wav

Usage:
    python test_chords.py
"""

import os
import sys
import torch
import numpy as np
import librosa
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'betterchord'))

from cnn_model import ChordCNN
from audio_processing import load_or_cache_spectrogram, load_audio, create_spectrogram
from music_theory import identify_chord_smart

# Set USE_ENSEMBLE = True to average outputs from all three seed models
# Set USE_ENSEMBLE = False to use single model (MODEL_PATH)
USE_ENSEMBLE = False

_BETTERCHORD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'betterchord')

MODEL_PATH  = os.path.join(_BETTERCHORD_DIR, 'chord_cnn.pth')

ENSEMBLE_PATHS = [
    os.path.join(_BETTERCHORD_DIR, 'chord_cnn_seed1.pth'),
    os.path.join(_BETTERCHORD_DIR, 'chord_cnn_seed2.pth'),
    os.path.join(_BETTERCHORD_DIR, 'chord_cnn_seed3.pth'),
]

TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'betterchord_mytesting', 'my_data_raw')
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'betterchord_mytesting', 'my_test_processed')
THRESHOLD = 0.48 # Found through inference testing
CHROMATIC  = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

# Known test file prefixes to auto-detect
TEST_PREFIXES = ["D9_", "Dm", "F#", "Cmaj", "Am", "Em", "G_", "C_", "Bm",
                 "Bb", "Ab", "Eb", "E_", "A_", "F_", "B_", "Gm", "Fm",
                 "Dminor", "Dmaj", "Gmaj", "Amaj", "Aminor", "Asus4",
                 "A69", "D_", "G7_", "G9_", "Gmaj7", "G_", "Cminor", "Bminor"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Maps any alternate spelling a filename might use -> canonical chord suffix.
# Keys are lowercase, no spaces. Add new ones here as you encounter them.
# Structure: "filename_spelling" -> "music_theory_spelling"
# Both sides are compared after lowercasing and stripping spaces/slashes.
# Major spellings (Xmaj -> X)
_MAJOR_ROOTS = ["c", "d", "e", "f", "g", "a", "b",
                "c#", "d#", "f#", "g#", "a#",
                "db", "eb", "gb", "ab", "bb"]
_MAJOR_ALIASES = {f"{r}maj": r for r in _MAJOR_ROOTS}

# Minor spellings (Xminor -> Xm)
_MINOR_ALIASES = {
    f"{r}minor": f"{r}m" for r in _MAJOR_ROOTS
}

# Minor chord type spellings
_QUALITY_ALIASES = {
    # minor extended
    "minor7":   "m7",
    "minor9":   "m9",
    "minor11":  "m11",
    "minor13":  "m13",
    "minormaj7":"mmaj7",
    "minmaj7":  "mmaj7",
    # dominant
    "dom7":     "7",
    "dom9":     "9",
    # diminished
    "diminished":  "dim",
    "diminished7": "dim7",
    # augmented
    "augmented":   "aug",
    # suspended
    "suspended2":  "sus2",
    "suspended4":  "sus4",
    # half-diminished
    "halfdim":     "m7b5",
    "halfdim7":    "m7b5",
    "hdim7":       "m7b5",
}

# Full chord aliases (specific whole-chord name overrides)
_FULL_ALIASES = {
    **_MAJOR_ALIASES,
    **_MINOR_ALIASES,
    # per-root minor+extension aliases for common filename patterns
    **{f"{r}minor9":  f"{r}m9"  for r in _MAJOR_ROOTS},
    **{f"{r}minor7":  f"{r}m7"  for r in _MAJOR_ROOTS},
    **{f"{r}minor11": f"{r}m11" for r in _MAJOR_ROOTS},
    **{f"{r}minor13": f"{r}m13" for r in _MAJOR_ROOTS},
    "a69": "a6",   # A6/9 normalizes to a6 after slash strip
    "a6/9": "a6",
}


def _normalize_chord_str(s):
    """Lowercase, strip spaces and slashes (ignores inversion for matching)."""
    return s.lower().replace(" ", "").split("/")[0]


def _apply_quality_aliases(s):
    """Replace any known quality alias substring in s."""
    for wrong, right in _QUALITY_ALIASES.items():
        if wrong in s:
            s = s.replace(wrong, right)
    return s


def is_correct(predicted_chord, expected_chord):
    """
    Check if prediction matches expected chord name.

    Matching strategy (in order):
      1. Direct substring / prefix match after normalization
      2. Full-chord alias lookup  (e.g. Dminor9 -> Dm9)
      3. Quality-alias substitution  (e.g. 'minor7' -> 'm7' in the string)
      4. All of the above applied to both predicted and expected
    """
    pred = _normalize_chord_str(predicted_chord)
    exp  = _normalize_chord_str(expected_chord)

    # 1. Direct match
    if exp == pred or pred.startswith(exp) or exp in pred:
        return True

    # 2. Full-chord alias on expected
    exp_alias = _FULL_ALIASES.get(exp, exp)
    if exp_alias == pred or pred.startswith(exp_alias) or exp_alias in pred:
        return True

    # 3. Quality-alias substitution on expected
    exp_qual = _apply_quality_aliases(exp)
    if exp_qual == pred or pred.startswith(exp_qual) or exp_qual in pred:
        return True

    # 4. Quality-alias substitution on predicted (e.g. model returns 'am7',
    #    expected after alias is 'am7' — catches cross-direction mismatches)
    pred_qual = _apply_quality_aliases(pred)
    if exp_alias == pred_qual or pred_qual.startswith(exp_alias):
        return True

    return False


def debug_audio(file_path, sr=22050):
    """
    Print diagnostic info about the audio file and onset detection.
    Helps diagnose cases where the model sees an empty or wrong spectrogram.
    """
    print(f"      [audio debug]")

    # Load raw file
    try:
        y_full, sr_actual = librosa.load(file_path, sr=sr, duration=60)
    except Exception as e:
        print(f"        ERROR loading file: {e}")
        return

    duration   = len(y_full) / sr
    peak       = float(np.max(np.abs(y_full)))
    rms        = float(np.sqrt(np.mean(y_full ** 2)))
    print(f"        File duration : {duration:.3f}s  |  Peak: {peak:.4f}  |  RMS: {rms:.4f}")

    if peak < 0.001:
        print(f"        ⚠ File appears silent — check recording")
        return

    # Onset detection
    onset_frames = librosa.onset.onset_detect(y=y_full, sr=sr)
    if len(onset_frames) == 0:
        print(f"        ⚠ No onsets detected — onset_time defaulting to 0.10s")
        onset_time = 0.10
    else:
        onset_times    = librosa.frames_to_time(onset_frames, sr=sr)
        onset_strength = librosa.onset.onset_strength(y=y_full, sr=sr)
        strength_at    = onset_strength[onset_frames]
        best_idx       = int(np.argmax(strength_at))
        onset_time     = float(onset_times[best_idx])
        print(f"        Onsets detected : {len(onset_frames)}  |  "
              f"Strongest at {onset_time:.3f}s  (strength={strength_at[best_idx]:.2f})")

    start_time = max(0, onset_time - 0.10)
    end_time   = min(duration, onset_time + 2.5)
    clip_len   = end_time - start_time

    print(f"        Clip window     : {start_time:.3f}s → {end_time:.3f}s  ({clip_len:.3f}s)")

    if clip_len < 0.5:
        print(f"        ⚠ Clip is very short ({clip_len:.3f}s) — onset may have misfired")

    # Check clipped section RMS
    y_clip    = y_full[int(start_time * sr):int(end_time * sr)]
    clip_rms  = float(np.sqrt(np.mean(y_clip ** 2))) if len(y_clip) > 0 else 0.0
    print(f"        Clip RMS        : {clip_rms:.4f}"
          + ("  ⚠ very quiet clip" if clip_rms < 0.02 else ""))


def parse_expected_chord(fname):
    """
    Extract expected chord name from filename.
    e.g. D9_X_5_4_5_5_X.wav    -> D9
         Cmaj_X_3_2_0_1_0.wav  -> Cmaj
         F#maj9_2_1_3_1_2_1.wav -> F#maj9
         Dminor9_X_5_3_5_5_X.wav -> Dminor9
    Splits on first _ that is followed by a digit or X (start of tab).
    """
    name  = os.path.splitext(fname)[0]
    parts = name.split("_")

    for i, part in enumerate(parts):
        if part == "X" or (part.lstrip("-").isdigit()):
            return "_".join(parts[:i])

    return parts[0]


def load_test_spectrogram(file_path):
    # Load spectrogram for test files, caching into my_test_processed
    # instead of the main spec_data directory
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    fname      = os.path.splitext(os.path.basename(file_path))[0]
    cache_path = os.path.join(PROCESSED_DIR, fname + '.npy')

    if os.path.exists(cache_path):
        return np.load(cache_path)

    y, sr       = load_audio(file_path)
    spectrogram = create_spectrogram(y, sr)
    np.save(cache_path, spectrogram)
    return spectrogram


def load_model(path=None):
    # Load a single model from path (defaults to MODEL_PATH)
    model = ChordCNN(chromatic_notes=12)
    model.load_state_dict(torch.load(path or MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model


def load_ensemble():
    # Load all ensemble models - skip any that don't exist yet
    models = []
    for path in ENSEMBLE_PATHS:
        if os.path.exists(path):
            models.append(load_model(path))
        else:
            print(f"  [!] Ensemble model not found, skipping: {os.path.basename(path)}")
    return models


def run_inference(models, file_path):
    # models can be a single model or a list for ensemble
    if not isinstance(models, list):
        models = [models]

    spectrogram = load_test_spectrogram(file_path)
    spec_tensor = (torch.tensor(spectrogram, dtype=torch.float32)
                   .unsqueeze(0).unsqueeze(0).to(device))

    all_note = []
    all_root = []
    all_bass = []

    with torch.no_grad():
        for model in models:
            note_out, root_out, bass_out = model(spec_tensor)
            all_note.append(note_out.squeeze().cpu().numpy())
            all_root.append(root_out.squeeze().cpu().numpy())
            all_bass.append(bass_out.squeeze().cpu().numpy())

    # Average logits across all models
    import numpy as np
    note_logits = np.mean(all_note, axis=0)
    root_logits = np.mean(all_root, axis=0)
    bass_logits = np.mean(all_bass, axis=0)

    result = identify_chord_smart(note_logits, root_logits, bass_logits, THRESHOLD)

    # Note probs from averaged logits
    note_probs           = 1 / (1 + np.exp(-note_logits))  # sigmoid
    result["note_probs"] = {CHROMATIC[i]: round(float(note_probs[i]), 3)
                            for i in range(12)}

    # Attach raw spectrogram stats for diagnostic use
    result["spec_min"]  = round(float(spectrogram.min()), 3)
    result["spec_max"]  = round(float(spectrogram.max()), 3)
    result["spec_mean"] = round(float(spectrogram.mean()), 3)

    return result


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*65)
    print("  CHORD IDENTIFICATION TEST")
    print("="*65)
    print(f"\n  Model  : {MODEL_PATH}")
    print(f"  Device : {device}")
    print(f"  Dir    : {TEST_DIR}\n")

    if USE_ENSEMBLE:
        models = load_ensemble()
        if not models:
            print(f"  [!] No ensemble models found")
            return
        print(f"  Mode   : Ensemble ({len(models)} models)")
    else:
        if not os.path.exists(MODEL_PATH):
            print(f"  [!] Model not found: {MODEL_PATH}")
            return
        models = load_model()
        print(f"  Mode   : Single model")

    test_files = sorted([
        f for f in os.listdir(TEST_DIR)
        if f.endswith('.wav')
        and any(f.startswith(p) for p in TEST_PREFIXES)
    ])

    if not test_files:
        print(f"  [!] No test wav files found in {TEST_DIR}")
        print(f"  Naming convention: ChordName_tab_with_underscores.wav")
        print(f"  Example: D9_X_5_4_5_5_X.wav")
        return

    print(f"  Found {len(test_files)} test files\n")

    correct = 0
    total   = 0

    for fname in test_files:
        file_path    = os.path.join(TEST_DIR, fname)
        expected     = parse_expected_chord(fname)
        result       = run_inference(models, file_path)

        chord     = result["chord"]
        root      = result["root"]
        bass      = result["bass"]
        notes     = result["notes"]
        method    = result["method"]
        conf      = result["confidence"]
        probs     = result["note_probs"]
        reasoning = result.get("reasoning", "")

        correct_flag = is_correct(chord, expected)
        status       = "✓" if correct_flag else "✗"
        total       += 1
        if correct_flag:
            correct += 1

        # ── Soft notes: above soft threshold (0.30) but below hard (THRESHOLD)
        soft_notes  = {k: v for k, v in probs.items() if 0.30 <= v < THRESHOLD}
        # ── Theoretical notes: what the identified chord should contain
        from chord_info import get_chord_info
        info         = get_chord_info(chord.split("/")[0]) if chord else None
        theory_notes = [iv["note"] for iv in info["intervals"]] if info else []

        print(f"  {status} {fname}")
        print(f"      Expected : {expected}")
        print(f"      Got      : {chord}  root={root}  bass={bass}")
        print(f"      Method   : {method}  Confidence: {conf:.3f}")
        print(f"      [1] Active notes  (>={THRESHOLD})  : {notes}")
        if soft_notes:
            print(f"      [2] Soft notes   (0.30-{THRESHOLD}) : "
                  + "  ".join(f"{k}={v:.3f}" for k, v in sorted(soft_notes.items(), key=lambda x: -x[1])))
        else:
            print(f"      [2] Soft notes   (0.30-{THRESHOLD}) : none")
        print(f"      [3] Theory notes (chord {chord.split('/')[0]}) : {theory_notes}")

        # Spectrogram sanity check
        spec_mean = result["spec_mean"]
        spec_max  = result["spec_max"]
        if len(notes) == 0 or spec_max < 0.5:
            print(f"      [spec] min={result['spec_min']}  "
                  f"max={spec_max}  mean={spec_mean}  ← may be empty/stale")
            debug_audio(file_path)
        else:
            print(f"      [spec] min={result['spec_min']}  "
                  f"max={spec_max}  mean={spec_mean}")

        if reasoning:
            print(f"      Reasoning: {reasoning}")

        print()

    print(f"  {'─'*60}")
    print(f"  Result: {correct}/{total} correct ({100*correct/total:.1f}%)")
    print("="*65)


if __name__ == "__main__":
    main()