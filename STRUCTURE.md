> This is the file layout of this repository -- the public copy of the
> BetterChord core ML/theory pipeline. It does not include the web
> frontend, the deployment tooling, or the internal project-management
> notes used while building the live app -- see `README.md` for what's
> published where (including the live app itself and where the trained
> model weights and the training/voicing/song data live).

Folders marked with `*` are not shipped in this repo -- see the sections
below for what goes in them.

```
BetterChord/
  main.py
  requirements-gpu.txt
  requirements-cpu.txt
  LICENSE
  README.md
  STRUCTURE.md
  .gitignore
  betterchord/
    __init__.py
    config/
      chord_parser.py
      interval_calculator.py
      music_theory.py
      chord_info.py
      voicings.py
      songs.py
      audio_processing.py
    training_scripts/
      cnn_model.py
      chord_to_notes.py
      database.py
      train.py
      export_onnx.py
    data_scripts/
      registry_builder.py
      guide_tone_grouping.py
      build_normalized_columns.py
  test_scripts/
    test_audio_file.py
    test_chords.py
  test_assets/
    Gminor_3_5_5_3_3_3.wav
  data/
    registry/
      quality_registry.json
      guide_tone_groups.json
    *training_data/
    *test_data/
    *spec_data/
    *noise_bank/
      *processed/
    *betterchord_mytesting/
      *my_data_raw/
      *my_test_processed/
    *voicing_data/
    *song_data/
```

## What's in each part

**`main.py`** is the inference orchestrator: audio -> spectrogram -> CNN
-> chord -> theory info/voicings/songs, as one callable pipeline. Run it
directly from the command line; see `README.md` for usage.

**`requirements-gpu.txt`** is for local dev with a CUDA GPU;
**`requirements-cpu.txt`** is CPU-only (covers both training and the
ONNX export step).

**`LICENSE`** is MIT, and covers the code in this repo only -- the
trained model weights are licensed separately; see `README.md`'s
Licensing section.

**`betterchord/config/`** holds the live-serving modules `main.py`
actually calls at inference time: `chord_parser.py`,
`interval_calculator.py`, `music_theory.py`, `chord_info.py`,
`voicings.py`, `songs.py`, `audio_processing.py`.

**`betterchord/training_scripts/`** is the CNN training pipeline -- not
used at inference time. `cnn_model.py`, `chord_to_notes.py`,
`database.py`, and `train.py` are the training pipeline itself.
`export_onnx.py` is a one-time-per-training-run step that exports the
trained PyTorch weights (`train.py`'s output) to ONNX via
`torch.onnx.export`, with a built-in numerical-equivalence check; re-run
it after every retrain. Neither the resulting `.pth` nor the `.onnx`
weight file is included in this repo -- both are published on Hugging
Face; see `README.md`'s Licensing section for where.

**`betterchord/data_scripts/`** is the registry/maintenance pipeline
(`registry_builder.py`, `guide_tone_grouping.py`,
`build_normalized_columns.py`), rerun when the voicing/song data
changes.

**`test_scripts/`** holds personal test scripts for the audio/theory
pipeline: `test_audio_file.py`, `test_chords.py`.

**`test_assets/`** holds one shipped audio fixture --
`Gminor_3_5_5_3_3_3.wav` -- kept deliberately separate from both `data/`
(the large-scale training/voicing/song data, documented above and below)
and `test_scripts/` (code, not audio). It exists because
`betterchord/training_scripts/export_onnx.py` hard-requires a real test
recording for its own built-in numerical-equivalence check (does the
freshly-exported `.onnx` produce the same result as the source `.pth`
on real audio?) and exits immediately if that file is missing. The
production repo points this at `frontend/public/assets/`, which is
entirely excluded here, so this repo ships its own copy of the exact
same file instead -- the same fixture already used elsewhere in the
project's own testing history, not a new or different recording.

**`data/registry/`** is small (~43 KB total) but genuinely load-bearing:
`quality_registry.json` (the canonical chord-quality -> interval-set
registry) and `guide_tone_groups.json` (ambiguous-quality groupings
behind the guide-tone "songs for a closely related chord" explanation)
are both read directly by `voicings.py` and `songs.py` at request time
(via `json.load`), so the pipeline can't identify or look up a chord
without them. Unlike the rest of `data/`, this subfolder is included in
this repo -- confirmed directly (not assumed) that it's the only part of
`data/` that's actually git-tracked in the private working repo this
copy was built from.

## Setting up `data/` for training

`data/registry/` (above) is the only part of `data/` this repo actually
ships. If you want to run `betterchord/training_scripts/train.py`
against your own data, the training code expects a real local `data/`
directory at the repo root with the subfolders below -- **none of these
are included in this repo.** They aren't shipped as empty/placeholder
folders in git either; this section exists purely so you know what to
create locally before training. Verified directly against the real
training code (`database.py`, `train.py`, `audio_processing.py`), not
assumed.

- **`data/training_data/`** and **`data/test_data/`** -- the actual
  training and held-out test sets. `train.py` sets `data_root` to
  `data/` at the repo root and calls `database.py`'s `get_dataloader()`,
  which reads `data_root/training_data` and `data_root/test_data`
  directly. Each is expected to contain one subfolder per chord class
  (e.g. `training_data/Cmaj7/`), and each of those subfolders holds
  `.wav` audio files for that chord -- `database.py`'s `ChordDataset`
  labels each file based on its filename prefix (`SELF_`/`SELFAC_` for
  real recordings with an embedded tab, `SYNTH_` for synthesized
  renders, `IDMT_` or a plain chord-name folder for externally-sourced
  audio labeled from the folder name alone). **Must already exist** --
  `ChordDataset.__init__` calls `os.listdir()` on the folder directly,
  with no existence check or auto-creation, so a missing directory
  raises a plain `FileNotFoundError` rather than being created for you.

- **`data/spec_data/`** -- a spectrogram cache, mirroring the same
  relative folder structure as `training_data`/`test_data` but with each
  `.wav` replaced by a cached `.npy` (the computed CQT spectrogram for
  that file). Populated by `audio_processing.py`'s
  `load_or_cache_spectrogram()`. **Auto-created** -- if a cached file
  doesn't exist yet, the function computes it and calls
  `os.makedirs(..., exist_ok=True)` before saving, so this directory
  never needs to exist ahead of time; it's a pure performance cache, safe
  to delete entirely (it will just be regenerated, more slowly, on the
  next run).

- **`data/noise_bank/processed/`** -- pre-processed background-noise
  samples (`.npy` files) mixed into `SYNTH`/`IDMT` training audio at
  random SNR for noise-robustness augmentation (real recordings are left
  clean). Read by `audio_processing.py`'s `load_noise_bank()`. **Neither
  auto-created nor required** -- if the directory doesn't exist,
  `load_noise_bank()` returns an empty list and training simply proceeds
  with noise augmentation disabled, rather than erroring.

**A separate, related case: `test_scripts/test_chords.py`'s own
`data/betterchord_mytesting/` folders.** This is for running that
one shipped script manually yourself, not for training a model (`train.py`
never touches this) and not for `main.py`'s normal inference pipeline --
so it's documented here rather than folded into the list above. Like the
training folders above, **neither subfolder is shipped in this repo, not
even empty:**

- **`data/betterchord_mytesting/my_data_raw/`** -- a flat folder of
  `.wav` test recordings, named `<ChordName>_<tab_with_underscores>.wav`
  (e.g. `D9_X_5_4_5_5_X.wav`) -- a different convention from
  `training_data`/`test_data` above (no per-chord subfolders; the chord
  name and fretting are both encoded directly in the filename, and only
  filenames starting with one of the script's own recognized prefixes
  are picked up). **Must already exist** -- `test_chords.py`'s `main()`
  reads it directly via `os.listdir(TEST_DIR)`, with no existence check.
- **`data/betterchord_mytesting/my_test_processed/`** -- a `.npy`
  spectrogram cache for those same test files, kept deliberately separate
  from the main `data/spec_data/` cache above (per the script's own
  comment) so running this script doesn't mix into that cache.
  **Auto-created** -- `load_test_spectrogram()` calls
  `os.makedirs(PROCESSED_DIR, exist_ok=True)` before writing to it.

**`data/voicing_data/` and `data/song_data/` (the two production
databases) are deliberately NOT part of this list** -- confirmed via a
direct search of every training-path file that neither is referenced by
any of `database.py`, `train.py`, or `audio_processing.py`. Both are
referenced only by the live-serving `betterchord/config/` modules
(`voicings.py`, `songs.py`) and by the `betterchord/data_scripts/`
registry-maintenance tooling, which is a separate concern from training
a model -- see the "Not included in this repo" note below for where
those live instead.

**Not included in this repo:** the trained model weights
(`chord_cnn.pth` / `chord_cnn.onnx`) and the rest of the `data/`
directory (raw training/test audio, the voicing database, and the song
database) -- both are published separately; see README.md's Licensing
section for where. To run `main.py` you need `chord_cnn.pth` (or
`chord_cnn.onnx`) in `betterchord/training_scripts/` yourself; to run
`train.py` against your own data, point it at your own local copy of
that data (see the `data_root` and `data/...` path handling inside
`betterchord/training_scripts/train.py` and `database.py`).
