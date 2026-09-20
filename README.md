# BetterChord 🎸

ML-powered guitar chord recognition with 92.96% test set accuracy, built on a CNN + a music theory engine.
Record or upload a single strum, and BetterChord identifies the chord,
shows you real fretboard voicings, breaks down the theory behind it, and
finds real songs that use it.

**Live: [better-chord.vercel.app](https://better-chord.vercel.app)**

## Status

This is an active passion project, not a finished product, but it's now
live on the web. This repository is the core ML + music-theory pipeline
(audio → chord → voicings/theory/songs) that powers the live app: a
runnable command-line tool, plus the full model training pipeline. It's
verified against real audio end to end. The live app's web frontend and
its API server (which wrap this same pipeline for the browser) live in a
separate, private working repository and aren't included here.

## Features

- **Chord identification from a single audio recording** — works from a
  real guitar strum. Recommend anyone to strum their guitar clearly, 
  with a pick, and loud. Standard tuning (make sure its tuned well!)
- **Multiple real fretboard voicings** for the identified chord
- **Music theory breakdown** — the actual notes/intervals that make up
  the chord, in plain language
- **Song recommendations** — real songs that use the identified chord,
  pulled from a database of tens of thousands of tracks
- **Guide-tone explanations** — when a closely related chord's songs get
  pulled in alongside your search (e.g. `C13` also surfaces `C7add13`),
  there's an actual explanation of *why*, written for someone who
  doesn't already know music theory

## How it works

1. **Librosa** turns the recorded audio into a CQT spectrogram.
2. A **PyTorch CNN** takes that spectrogram and predicts three things:
   which of the 12 chromatic notes are present, the root, and the bass.
   It predicts raw notes, not a fixed set of chord classes. This means
   it can theoretically identify any chord whose interval pattern is
   known to the system, not just chords it was explicitly trained on.
3. A **rule-based music theory engine** takes those note predictions and
   determines the actual chord name by matching against a canonical
   interval registry.
4. That registry is the core of the project: the project's three data
   sources (the chord naming engine, a database of scraped fretboard
   voicings, and a database of real songs and their chords) used to
   independently disagree about how to name and group chords. A shared
   registry now reconciles all three by comparing actual interval
   content rather than trusting names/strings. This closed dozens of real
   coverage gaps and fixing several silent-wrong-answer bugs along the
   way.

## Tech stack

- **PyTorch** — CNN for note/root/bass prediction (also exported to
  **ONNX Runtime** via `export_onnx.py`, for inference without PyTorch)
- **Librosa** — audio analysis and spectrogram generation
- **NumPy / pandas** — numerical processing

The live web app additionally wraps this pipeline with a FastAPI backend
and a React frontend — that glue code isn't part of this repo (see
Status above).

## Project structure

See [STRUCTURE.md](STRUCTURE.md) for the full annotated layout. In short:

```
BetterChord/
├── main.py                               # inference: audio -> chord -> theory/voicings/songs
├── requirements-gpu.txt                  # local dev, CUDA GPU
├── requirements-cpu.txt                  # CPU-only
├── LICENSE                               # MIT -- code only, see Licensing below
├── STRUCTURE.md
├── betterchord/
│   ├── config/                           # live-serving modules
│   │   ├── chord_parser.py
│   │   ├── interval_calculator.py
│   │   ├── music_theory.py
│   │   ├── chord_info.py
│   │   ├── voicings.py
│   │   ├── songs.py
│   │   └── audio_processing.py
│   ├── training_scripts/                 # CNN training pipeline (not used at runtime)
│   │   ├── cnn_model.py
│   │   ├── chord_to_notes.py
│   │   ├── database.py
│   │   ├── train.py
│   │   └── export_onnx.py                # trained-weights .pth -> .onnx export step
│   │                                      # (the weight files themselves aren't in this
│   │                                      # repo -- see Licensing below)
│   └── data_scripts/                     # registry/maintenance pipeline, rerun with data changes/updates
│       ├── registry_builder.py
│       ├── guide_tone_grouping.py
│       └── build_normalized_columns.py
└── test_scripts/
    ├── test_audio_file.py
    └── test_chords.py
```

The training/voicing/song data (`data/` — raw audio, the voicing
database, the song database) isn't included in this repo. See
**Licensing** below for where the published data and model weights live.

## Prerequisites

- Python 3.8+
- Git
- ffmpeg (⚠️ needed for M4A/MP3 and other non-WAV/FLAC formats)

**Install ffmpeg:**

Mac:
```bash
brew install ffmpeg
```

Windows: download from https://ffmpeg.org/download.html

Linux:
```bash
sudo apt-get install ffmpeg
```

## Installation

```bash
git clone https://github.com/[user]/BetterChord_Public.git
cd BetterChord_Public
```

Then install dependencies depending on your machine:

**If you have a CUDA GPU** (recommended for training, optional for inference):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-gpu.txt
```

**CPU-only:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-cpu.txt
```

**Note:** the trained model's training/test data isn't included directly
in this repo. See [Licensing](#licensing) below for where it's published.

## Usage

Use the live app at [better-chord.vercel.app](https://better-chord.vercel.app),
or run the full pipeline directly from the command line:

```bash
python main.py <filename>
```

This identifies the chord from the given audio file and prints the
identified chord, its theory breakdown, a few real fretboard voicings,
and a few real songs that use it.

**Before running `main.py` yourself:** download `chord_cnn.pth` (or
`chord_cnn.onnx`) from the Hugging Face model repo linked under
Licensing below and place it in `betterchord/training_scripts/` — the
weight file isn't included in this repo.

## Roadmap

This is an ongoing project. The two biggest open items: improving the
model itself (more/better training data is the single biggest lever),
and finishing the publication of the underlying training/voicing/song
data in a reproducible, properly-attributed form (see Licensing below
for what's currently available).

## Licensing

This repository's **code** is MIT licensed — see [LICENSE](LICENSE).

**The trained model weights (`chord_cnn.pth` / `chord_cnn.onnx`) are
licensed separately, under CC BY-NC 4.0 (Attribution-NonCommercial), NOT
MIT.** The code license does not cover the weights — if you use the
weights, the CC BY-NC 4.0 terms (attribution required, non-commercial
use only) apply to them independently of what you do with the code.

Where things live:

- **Model weights** (`chord_cnn.pth` / `chord_cnn.onnx`, CC BY-NC 4.0):
  published on Hugging Face —
  `[PLACEHOLDER: Hugging Face model repo URL, not yet finalized]`.
  See that repo's model card for training details, intended use, and
  known limitations.
- **Training/voicing/song data**: published separately from this code
  repo (raw audio, the voicing database, the song database) —
  `[PLACEHOLDER: link to the published dataset — Hugging Face Datasets
  or Kaggle, not yet finalized]`. Care was taken around the licensing of
  each underlying data source (some of it is more restrictively licensed
  than the code or the model weights); see that dataset's own
  documentation for the full per-source attribution and license terms
  before reusing it.

# Guitar Emoji!!! 🎸🎸🎸
