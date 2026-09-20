# Converts a chord name to a 12-element binary vector where each position

# The chromatic scale
CHROMATIC = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

# Converts non-standard names to my chromatic convention:
# A#/D#/G# → Bb/Eb/Ab (sharp to flat)
# Gb/Db    → F#/C#    (flat to sharp, since F# and C# are more common in guitar)

# Normalizes non-standard root names to our chromatic convention
ROOT_ENHARMONIC_MAP = {
    "A#": "Bb",
    "D#": "Eb",
    "G#": "Ab",
    "Gb": "F#",
    "Db": "C#",
}

# dictionary of chord types, keys values are type of chord, list associated is notes used in it
# Generated directly from quality_registry.json for consistency with the
# rest of the pipeline (voicings.db, songs.py, music_theory.py all
# already derive from the same registry). Previously only had 8 entries,
# meaning any HF/IDMT/GADA-sourced training folder (no per-file tab data,
# so labeled purely from folder name) for any OTHER quality got silently
# skipped during training -- see database.py's ChordDataset, which drops
# a file entirely if chord_to_binary() returns None for its folder.
#
# NOTE: two registry qualities ('b9', '#9') don't survive this file's own
# naive parse_chord() when written as bare accidentals (e.g. "Cb9" fails
# to parse at all; "C#9" gets misread as root="C#", quality="9"). Any
# future training folder for these two MUST be named with the explicit
# "add" form instead -- Caddb9 / Cadd#9 -- which parses correctly and
# matches the same convention already used for external voicing sources.
QUALITY_INTERVALS = {
    '': [0, 4, 7],
    'add#9': [0, 3, 4, 7],
    '-5': [0, 4, 6],
    '11': [0, 2, 4, 5, 7, 10],
    '11b9': [0, 1, 4, 5, 7, 10],
    '13': [0, 2, 4, 5, 7, 9, 10],
    '13#11': [0, 2, 4, 6, 7, 9, 10],
    '13#9': [0, 3, 4, 5, 7, 9, 10],
    '13b9': [0, 1, 4, 5, 7, 9, 10],
    '13b9#11': [0, 1, 4, 6, 7, 9, 10],
    '13sus4': [0, 2, 5, 7, 9, 10],
    '5': [0, 7],
    '6': [0, 4, 7, 9],
    '6/9': [0, 2, 4, 7, 9],
    '6add11': [0, 4, 5, 7, 9],
    '6sus2': [0, 2, 7, 9],
    '6sus4': [0, 5, 7, 9],
    '7': [0, 4, 7, 10],
    '7#11': [0, 4, 6, 7, 10],
    '7#5#9': [0, 3, 4, 8, 10],
    '7#5b9': [0, 1, 4, 8, 10],
    '7#9': [0, 3, 4, 7, 10],
    '7add11': [0, 4, 5, 7, 10],
    '7add13': [0, 4, 7, 9, 10],
    '7b13': [0, 4, 7, 8, 10],
    '7b5': [0, 4, 6, 10],
    '7b5#9': [0, 3, 4, 6, 10],
    '7b5b9': [0, 1, 4, 6, 10],
    '7b9': [0, 1, 4, 7, 10],
    '7b9b13': [0, 1, 4, 7, 8, 10],
    '7no3': [0, 7, 10],
    '7sus2': [0, 2, 7, 10],
    '7sus4': [0, 5, 7, 10],
    '9': [0, 2, 4, 7, 10],
    '9#11': [0, 2, 4, 6, 7, 10],
    '9#5': [0, 2, 4, 8, 10],
    '9add13': [0, 2, 4, 7, 9, 10],
    '9b13': [0, 2, 4, 7, 8, 10],
    '9b5': [0, 2, 4, 6, 10],
    '9sus4': [0, 2, 5, 7, 10],
    'add#11': [0, 4, 6, 7],
    'add4': [0, 4, 5, 7],
    'add4add9': [0, 2, 4, 5, 7],
    'add9': [0, 2, 4, 7],
    'add9#11': [0, 2, 4, 6, 7],
    'addb13': [0, 4, 7, 8],
    'aug': [0, 4, 8],
    'aug13': [0, 2, 4, 5, 8, 9, 10],
    'aug7': [0, 4, 8, 10],
    'augmaj7': [0, 4, 8, 11],
    'addb9': [0, 1, 4, 7],
    'dim': [0, 3, 6],
    'dim7': [0, 3, 6, 9],
    'dim7b13': [0, 3, 6, 8],
    'm': [0, 3, 7],
    'm11': [0, 2, 3, 5, 7, 10],
    'm11b5': [0, 2, 3, 5, 6, 10],
    'm13': [0, 2, 3, 5, 7, 9, 10],
    'm6': [0, 3, 7, 9],
    'm6/9': [0, 2, 3, 7, 9],
    'm7': [0, 3, 7, 10],
    'm7#5': [0, 3, 8, 10],
    'm7add11': [0, 3, 5, 7, 10],
    'm7add13': [0, 3, 7, 9, 10],
    'm7b13': [0, 3, 7, 8, 10],
    'm7b5': [0, 3, 6, 10],
    'm7no5': [0, 3, 10],
    'm9': [0, 2, 3, 7, 10],
    'm9#5': [0, 2, 3, 8, 10],
    'madd4': [0, 3, 5, 7],
    'madd9': [0, 2, 3, 7],
    'maj11': [0, 2, 4, 5, 7, 11],
    'maj13': [0, 2, 4, 5, 7, 9, 11],
    'maj13#11': [0, 2, 4, 6, 7, 9, 11],
    'maj7': [0, 4, 7, 11],
    'maj7#11': [0, 4, 6, 7, 11],
    'maj7#9': [0, 3, 4, 7, 11],
    'maj7add11': [0, 4, 5, 7, 11],
    'maj7add13': [0, 4, 7, 9, 11],
    'maj7b5': [0, 4, 6, 11],
    'maj7sus2': [0, 2, 7, 11],
    'maj7sus4': [0, 5, 7, 11],
    'maj9': [0, 2, 4, 7, 11],
    'maj9#11': [0, 2, 4, 6, 7, 11],
    'maj9add13': [0, 2, 4, 7, 9, 11],
    'mb6': [0, 3, 8],
    'mb9': [0, 1, 3, 7],
    'mmaj13': [0, 2, 3, 5, 7, 9, 11],
    'mmaj7': [0, 3, 7, 11],
    'mmaj9': [0, 2, 3, 7, 11],
    'sus2': [0, 2, 7],
    'sus2add#11': [0, 2, 6, 7],
    'sus2sus4': [0, 2, 5, 7],
    'sus4': [0, 5, 7],
    'susb9': [0, 1, 5, 7],
}


def normalize(chord_name):
    # Apply enharmonic normalization to a chord name.
    # Works by parsing the root, normalizing it, then recombining with quality.

    if not chord_name:
        return chord_name
 
    # Extract root - check 2-char roots first to avoid Eb being parsed as E
    if len(chord_name) >= 2 and chord_name[1] in ("#", "b"):
        root    = chord_name[:2]
        quality = chord_name[2:]
    else:
        root    = chord_name[0]
        quality = chord_name[1:]
 
    # Normalize root if needed
    normalized_root = ROOT_ENHARMONIC_MAP.get(root, root)
 
    return normalized_root + quality


def parse_chord(chord):
    # Split chord name into (root, quality) Handles 1-char roots (C, D, E, F, G, A, B) and 2-char roots (C#, Bb, Eb, Ab, F#). Returns (None, None) if unrecognized. Also something that might need updating if new chord classes within db

    if not chord:
        return None, None

    if len(chord) >= 2 and chord[1] in ("#", "b"):
        root, quality = chord[:2], chord[2:]
    else:
        root, quality = chord[0], chord[1:]

    if root not in CHROMATIC:
        return None, None

    return root, quality


def chord_to_binary(chord_name):

    # Convert a chord name to a 12-element binary vector

    # Returns:
        # list: 12-element binary vector e.g. [1,0,0,1,0,0,0,1,0,0,0,0]
        # None: if chord name is unrecognized

    chord_name = normalize(chord_name)
    root, quality = parse_chord(chord_name)

    if root is None:
        print(f"  [!] Could not parse chord: '{chord_name}'")
        return None

    intervals = QUALITY_INTERVALS.get(quality)
    if intervals is None:
        return None

    root_idx = CHROMATIC.index(root)
    vector   = [0] * 12
    for interval in intervals:
        vector[(root_idx + interval) % 12] = 1

    return vector


def chord_to_notes(chord_name):

    # Returns the actual note names for a chord instead of binary vector just in case

    vector = chord_to_binary(chord_name)
    if vector is None:
        return None
    return [CHROMATIC[i] for i, v in enumerate(vector) if v == 1]