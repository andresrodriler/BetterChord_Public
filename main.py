"""
main.py -- the inference orchestrator. Ties the whole pipeline together:

    audio file -> spectrogram -> CNN -> identify_chord_smart ->
    chord_info + voicings + songs -> one combined response dict

This is deliberately just plumbing -- no new logic. Every piece it calls
(audio_processing, music_theory, chord_info, voicings, songs) has
already been built and verified independently; this file's only job is
wiring them together correctly.

CNN inference runs via ONNX Runtime (chord_cnn.onnx) when available --
the deployed backend ships onnxruntime and no PyTorch -- and falls back
to the PyTorch model (chord_cnn.pth) wherever torch is installed. See
_init_backend / _run_cnn.

Usage:
    python main.py path/to/audio.wav
"""

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

# main.py lives at the BetterChord/ project root; everything it needs
# lives two levels down inside betterchord/config/ (music_theory,
# chord_info, voicings, songs, chord_parser, interval_calculator)
# and betterchord/training_scripts/ (audio_processing, cnn_model,
# chord_cnn.pth) -- neither is found by Python automatically since
# they're sibling subdirectories, not the same folder as this file.
sys.path.insert(0, os.path.join(_HERE, "betterchord", "config"))
sys.path.insert(0, os.path.join(_HERE, "betterchord", "training_scripts"))

from audio_processing import load_audio, create_spectrogram
from music_theory import identify_chord_smart
from chord_info import get_chord_info
from voicings import get_voicings
from songs import get_songs
import chord_parser as cp

# Two inference artifacts, both derived from the same training run:
#   chord_cnn.pth   -- the trained PyTorch model, the source of truth
#                      train.py produces.
#   chord_cnn.onnx  -- a converted copy of those exact weights, run via
#                      onnxruntime with no PyTorch dependency. The
#                      deployed container ships this one and NOT torch
#                      (see betterchord/training_scripts/export_onnx.py
#                      and the Dockerfile). Verified numerically
#                      equivalent on real audio at export time.
# torch and cnn_model are imported lazily, only on the fallback path --
# importing torch at module load costs ~300 MiB of RSS the deployed
# image does not pay.
MODEL_PATH = os.path.join(_HERE, "betterchord", "training_scripts", "chord_cnn.pth")
ONNX_PATH = os.path.join(_HERE, "betterchord", "training_scripts", "chord_cnn.onnx")

_MODEL_INPUT_SHAPE = (1, 1, 84, 119)  # ChordCNN's FC layer is fixed to 84x119 frames
_OUTPUT_NAMES = ["note_out", "root_out", "bass_out"]

_backend = None          # "onnx" or "torch", resolved on first inference
_onnx_session = None
_torch_model = None
_torch_device = None


def _init_backend():
    """Choose the inference backend once, lazily.

    ONNX first (onnxruntime installed and chord_cnn.onnx present) -- the
    deployed path, which keeps torch out of the image. Falls back to the
    PyTorch .pth wherever torch is installed and the .onnx has not been
    exported (local dev / testing against the weights directly).
    """
    global _backend, _onnx_session, _torch_model, _torch_device
    if _backend is not None:
        return

    if os.path.exists(ONNX_PATH):
        try:
            import onnxruntime as ort

            # Memory-lean settings for a small (~512 MB) instance: no CPU
            # memory arena (the arena holds large pre-allocated blocks),
            # single-threaded ops. intra_op_num_threads caps parallelism
            # WITHIN one inference, not concurrency across requests --
            # InferenceSession.run is thread-safe and releases the GIL,
            # so concurrent /identify calls still run in parallel (each
            # on one thread). Actual CNN compute is sub-millisecond; the
            # request cost is dominated by librosa's CQT.
            so = ort.SessionOptions()
            so.enable_cpu_mem_arena = False
            so.intra_op_num_threads = 1
            so.inter_op_num_threads = 1
            _onnx_session = ort.InferenceSession(
                ONNX_PATH, sess_options=so, providers=["CPUExecutionProvider"]
            )
            _backend = "onnx"
            return
        except ImportError:
            pass  # onnxruntime missing -- try torch

    import torch
    from cnn_model import ChordCNN

    _torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = ChordCNN()
    m.load_state_dict(torch.load(MODEL_PATH, map_location=_torch_device))
    m.to(_torch_device)
    m.eval()  # disables dropout/batchnorm training behavior -- required for inference
    _torch_model = m
    _backend = "torch"


def _run_cnn(spec):
    """spec: (84, 119) float array from create_spectrogram.
    Returns (note_logits, root_logits, bass_logits) as length-12 numpy
    arrays -- the raw head outputs, same as ChordCNN.forward.
    """
    _init_backend()
    x = np.ascontiguousarray(spec, dtype=np.float32)[None, None, :, :]
    if x.shape != _MODEL_INPUT_SHAPE:
        raise ValueError(
            f"spectrogram shape {x.shape} != model input {_MODEL_INPUT_SHAPE} "
            "-- the CNN's flattened FC layer is fixed to 84x119 frames."
        )

    if _backend == "onnx":
        note_out, root_out, bass_out = _onnx_session.run(_OUTPUT_NAMES, {"input": x})
    else:
        import torch

        with torch.no_grad():
            note_t, root_t, bass_t = _torch_model(torch.from_numpy(x).to(_torch_device))
        note_out, root_out, bass_out = (
            note_t.cpu().numpy(),
            root_t.cpu().numpy(),
            bass_t.cpu().numpy(),
        )

    return note_out[0], root_out[0], bass_out[0]


def _load_model():
    """Back-compat shim: load and return the PyTorch ChordCNN directly.
    Kept for local dev / test scripts that poke the .pth; the deployed
    path uses ONNX via _run_cnn and never calls this.
    """
    global _torch_model, _torch_device
    if _torch_model is None:
        import torch
        from cnn_model import ChordCNN

        _torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        m = ChordCNN()
        m.load_state_dict(torch.load(MODEL_PATH, map_location=_torch_device))
        m.to(_torch_device)
        m.eval()
        _torch_model = m
    return _torch_model


def identify_from_audio(file_path):
    """
    Full pipeline: path to an audio file -> one combined response dict
    with the identified chord, theory info, fretboard voicings, and
    matching songs (including the guide-tone related_notes explanation).
    """
    # 1. audio file -> waveform -> CQT spectrogram
    #    load_audio applies clip compression + RMS normalization;
    #    create_spectrogram applies dataset-mean/std normalization +
    #    winsorization -- both already tuned during training, untouched here.
    y, sr = load_audio(file_path)
    spec = create_spectrogram(y, sr)  # shape (84, 119) -- (n_bins, time_frames)

    # 2-3. spectrogram -> CNN forward pass -> raw logits for all three
    #      heads (NOT probabilities; identify_chord_smart applies
    #      sigmoid/softmax itself). Runs via ONNX or PyTorch depending on
    #      what's installed -- see _init_backend.
    note_logits, root_logits, bass_logits = _run_cnn(spec)

    # 4. raw logits -> identified chord (name, root, quality, confidence,
    #    bass, alternatives -- see identify_chord_smart's own docstring
    #    for the full return shape)
    identified = identify_chord_smart(note_logits, root_logits, bass_logits)
    # identified["chord"] is a DISPLAY string (e.g. "A6/9 / A6add9" --
    # primary name + alias, joined with " / ") -- never re-parse it.
    # Build a clean, single, parseable chord name from the structured
    # fields via format_chord instead. Reusing identified["chord"]
    # directly makes get_chord_info mis-parse the alias text as a bogus
    # bass note, and get_voicings/search find nothing.
    display_name = identified["chord"]
    bass = identified["bass"] if identified["bass"] != identified["root"] else None
    chord_name = cp.format_chord(identified["root"], identified["quality"], bass)

    # Print immediately -- if anything below fails, we still know what
    # chord was actually identified instead of losing it to a crash.
    print(f"[identified chord: {chord_name!r}, confidence={identified['confidence']:.3f}]")

    # 5. chord name -> theory info, fretboard voicings, matching songs
    info = get_chord_info(chord_name)

    # Voicings and songs are wrapped individually: a failure in either
    # one (e.g. a public downloader's release DB missing a column this
    # code expects -- see voicings.py's/songs.py's own public-repo notes)
    # must never lose the chord identification + theory breakdown already
    # computed above. get_songs() already degrades gracefully for its one
    # known cause (a public repo note explains it, see songs.py) by
    # returning {"error": ...} rather than raising; these try/excepts are
    # the backstop for anything else that goes wrong in either lookup, so
    # the pipeline always returns a usable result either way.
    try:
        voicing_result = get_voicings(chord_name)
    except Exception as e:
        voicing_result = {
            "voicings": [],
            "fallback": None,
            "displayed": chord_name,
            "translated": False,
            "translated_from": None,
            "bass": None,
            "error": f"Voicing lookup failed: {e}",
        }

    try:
        song_result = get_songs(chord_name)
    except Exception as e:
        song_result = {"error": f"Song lookup failed: {e}"}

    # Cap output for readability during manual testing -- 3 voicings, and
    # up to 3 songs per spelling within results_by_spelling.
    voicing_result = dict(voicing_result)
    voicing_result["voicings"] = voicing_result["voicings"][:3]

    song_result = dict(song_result)
    # get_songs() returns {"error": "..."} (no results_by_spelling key at all)
    # when the chord name can't be parsed, or parses but matches no known
    # registry quality -- both real, distinct failure modes worth seeing
    # clearly rather than crashing on a missing key.
    if "results_by_spelling" in song_result:
        song_result["results_by_spelling"] = {
            spelling: songs[:3] for spelling, songs in song_result["results_by_spelling"].items()
        }

    return {
        "identified": identified,
        "chord_name": chord_name,
        "info": info,
        "voicings": voicing_result,
        "songs": song_result,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <path-to-audio-file>")
        print("  (a relative path resolves against the current working")
        print("  directory, same as any other command-line tool)")
        sys.exit(1)

    # NOTE (public repo only): this used to join the given filename onto
    # a hardcoded personal directory (data/betterchord_mytesting/my_data_raw)
    # -- that directory isn't part of this repo (it never shipped here at
    # all) and doesn't exist for anyone who downloads this copy, so
    # keeping it as a fallback base wouldn't be "genuinely useful" here,
    # just a silent dead end. A relative path passed on the command line
    # now resolves the normal way -- against the current working
    # directory -- rather than being silently ignored in favor of a
    # personal path that can't exist for a public downloader.
    audio_path = sys.argv[1]

    result = identify_from_audio(audio_path)
    print(json.dumps(result, indent=2, default=str))