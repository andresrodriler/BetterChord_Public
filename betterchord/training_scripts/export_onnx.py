"""export_onnx.py -- one-time (per training run) export of the trained
ChordCNN from PyTorch to ONNX, for lightweight inference on the deployed
backend.

WHAT THIS DOES
    Loads chord_cnn.pth -- the trained PyTorch model, the source of truth
    that train.py produces -- and writes chord_cnn.onnx next to it: the
    SAME weights, in a format onnxruntime runs WITHOUT PyTorch installed.
    Training is not touched. This is a purely additive post-training
    step; nothing about how chord_cnn.pth is made changes.

WHY
    The deployed container (see the repo-root Dockerfile) ships
    onnxruntime instead of torch/torchvision/torchaudio. Importing torch
    alone costs ~300 MiB of anonymous RSS; onnxruntime is a fraction of
    that. The prior Phase 7 session measured the torch-based image needing
    a 2 GB instance; this export is what lets it fit a 512 MB tier.

WHEN TO RE-RUN
    Every time the model is retrained. The workflow:

        train.py  ->  chord_cnn.pth  ->  python export_onnx.py  ->  chord_cnn.onnx
        ->  upload chord_cnn.onnx to the private Hugging Face repo (replacing the old one)

    See this repo's README.md (Licensing section) for where the
    resulting chord_cnn.onnx gets published.

WHAT IT PRODUCES
    betterchord/training_scripts/chord_cnn.onnx
      input:   "input", shape (batch, 1, 84, 119)  -- the fixed CQT
               spectrogram size main.py's pipeline feeds the model
               (dropout is inactive in eval mode, so the graph is
               deterministic).
      outputs: "note_out", "root_out", "bass_out"  -- raw logits, exactly
               as ChordCNN.forward returns them; identify_chord_smart
               applies sigmoid/softmax downstream, unchanged.

    Before writing anything as "done", the script runs the REAL inference
    pipeline (load_audio + create_spectrogram) on a committed test WAV
    through both PyTorch and ONNX and fails loudly if the logits, or the
    final identified chord + confidence, diverge beyond tolerance.

Run from anywhere:  python betterchord/training_scripts/export_onnx.py
"""

import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "betterchord", "config"))
sys.path.insert(0, _HERE)

from cnn_model import ChordCNN
from audio_processing import load_audio, create_spectrogram
from music_theory import identify_chord_smart

PTH_PATH = os.path.join(_HERE, "chord_cnn.pth")
ONNX_PATH = os.path.join(_HERE, "chord_cnn.onnx")
# NOTE (public repo only): the production repo's export_onnx.py points
# this at frontend/public/assets/Gminor_3_5_5_3_3_3.wav -- frontend/ is
# entirely excluded from this repo, so that path doesn't exist here. A
# copy of the exact same file (the same fixture referenced elsewhere in
# the project's own testing history) is shipped in this repo instead, at
# test_assets/ -- see STRUCTURE.md for why it lives there.
TEST_WAV = os.path.join(_ROOT, "test_assets", "Gminor_3_5_5_3_3_3.wav")

OPSET = 17
# ORT-CPU and PyTorch-CPU both compute in fp32; they differ only in op
# library and accumulation order. For a small conv/bn/relu/maxpool/linear
# net the per-element gap is normally ~1e-6..1e-5. atol=1e-4 / rtol=1e-3
# is a comfortable margin that still catches a real defect (wrong
# weights, wrong shape, dropout leaking from training mode). The
# end-to-end check below is stricter: it requires the SAME chord name and
# confidence within 1e-4.
ATOL, RTOL = 1e-4, 1e-3
OUTPUT_NAMES = ["note_out", "root_out", "bass_out"]


def main():
    if not os.path.exists(PTH_PATH):
        sys.exit(f"export_onnx: {PTH_PATH} not found -- train the model first (train.py).")
    if not os.path.exists(TEST_WAV):
        sys.exit(f"export_onnx: test WAV not found at {TEST_WAV}.")

    model = ChordCNN()
    model.load_state_dict(torch.load(PTH_PATH, map_location="cpu"))
    model.eval()

    # Use the REAL preprocessing output shape as the export dummy -- do
    # not assume it from the model class. The FC layer is hard-wired to
    # 84x119 frames, so this shape is fixed, but confirm it here.
    y, sr = load_audio(TEST_WAV)
    spec = create_spectrogram(y, sr)
    print(f"export_onnx: real spectrogram shape from "
          f"{os.path.basename(TEST_WAV)} = {spec.shape}")
    x = np.ascontiguousarray(spec, dtype=np.float32)[None, None, :, :]  # (1,1,84,119)
    if x.shape != (1, 1, 84, 119):
        sys.exit(f"export_onnx: unexpected spectrogram shape {x.shape}; "
                 "the CNN input is fixed at (1, 1, 84, 119).")
    dummy = torch.from_numpy(x)

    torch.onnx.export(
        model,
        dummy,
        ONNX_PATH,
        input_names=["input"],
        output_names=OUTPUT_NAMES,
        dynamic_axes={
            "input": {0: "batch"},
            "note_out": {0: "batch"},
            "root_out": {0: "batch"},
            "bass_out": {0: "batch"},
        },
        opset_version=OPSET,
        training=torch.onnx.TrainingMode.EVAL,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"export_onnx: wrote {ONNX_PATH} "
          f"({os.path.getsize(ONNX_PATH):,} bytes, opset {OPSET})")

    # ---- numerical equivalence on the REAL test audio ----
    with torch.no_grad():
        t_note, t_root, t_bass = (o.numpy() for o in model(dummy))

    import onnxruntime as ort
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    o_note, o_root, o_bass = sess.run(OUTPUT_NAMES, {"input": x})

    logits_ok = True
    for name, t, o in [("note_out", t_note, o_note),
                       ("root_out", t_root, o_root),
                       ("bass_out", t_bass, o_bass)]:
        max_abs = float(np.max(np.abs(t - o)))
        close = bool(np.allclose(t, o, atol=ATOL, rtol=RTOL))
        print(f"export_onnx: {name}: max|delta| = {max_abs:.3e}  "
              f"allclose(atol={ATOL}, rtol={RTOL}) = {close}")
        logits_ok = logits_ok and close

    # end-to-end: identify_chord_smart must land on the same chord
    ti = identify_chord_smart(t_note[0], t_root[0], t_bass[0])
    oi = identify_chord_smart(o_note[0], o_root[0], o_bass[0])
    print(f"export_onnx: torch identify -> chord={ti['chord']!r} "
          f"root={ti['root']!r} quality={ti['quality']!r} bass={ti['bass']!r} "
          f"conf={ti['confidence']:.6f}")
    print(f"export_onnx: onnx  identify -> chord={oi['chord']!r} "
          f"root={oi['root']!r} quality={oi['quality']!r} bass={oi['bass']!r} "
          f"conf={oi['confidence']:.6f}")
    end_to_end_ok = (
        ti["chord"] == oi["chord"]
        and ti["root"] == oi["root"]
        and ti["quality"] == oi["quality"]
        and ti["bass"] == oi["bass"]
        and abs(float(ti["confidence"]) - float(oi["confidence"])) < 1e-4
    )
    print(f"export_onnx: end-to-end identical = {end_to_end_ok}")

    if not (logits_ok and end_to_end_ok):
        try:
            os.remove(ONNX_PATH)
        except OSError:
            pass
        sys.exit("export_onnx: FAILED numerical-equivalence check -- "
                 "chord_cnn.onnx removed. Do not deploy.")

    print("export_onnx: OK -- chord_cnn.onnx verified numerically "
          "equivalent to chord_cnn.pth on real audio.")


if __name__ == "__main__":
    main()
