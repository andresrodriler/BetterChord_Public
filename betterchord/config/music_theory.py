# Identifies a chord name from a 12-element note probability vector, root index, and bass index, this part sucks so more comments below one annoying problem i found

import numpy as np
import chord_parser as cp

CHROMATIC = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
THRESHOLD = 0.48 # Found through inference testing

# Chord naming bias 
# When two chord names are genuinely ambiguous (same or overlapping notes), this score breaks the tie toward the name a guitarist would actually say. # Figured out that voicings to chord naming is a one to many problem so this is needed
# An example of this:
#   C major (C E G) shares its notes with Emb6 (E G C), difference is what acts as root, which usually on guitar root is the bass note, not always tho
#   Without a bias, the algorithm might pick the exotic name because it scores equally or slightly higher due to note coverage math.
# How it works:
#   Final score = raw_match_score * CONVENTIONALITY[quality]
#   So a perfect-scoring "6" match (0.95 * 0.75 = 0.71) loses to a
#   slightly-lower-scoring "" match (0.88 * 1.00 = 0.88).
# Ordering rationale (This is kinda a subjective part, logic is using the more 'simple' name, since this is more opionated, this is subject to change, rn its ok tho):
#   1.00  - major/minor triads: most common chords in all of western music
#   0.97  - power chord: extremely common, unambiguous
#   0.95  - dom7, maj7, m7: bread-and-butter jazz/pop chords
#   0.90  - sus, add, dim, aug: common but more specific
#   0.85  - 6th chords, 9th chords: common in jazz, less so elsewhere
#   0.80  - extended 7ths (7b9, 7#9 etc): jazz-specific, less common
#   0.72  - 11th chords: rare in guitar, usually voiced as something simpler
#   0.65  - 13th chords: very rare on guitar, almost always a jazz piano thing
#   0.50  - highly altered / exotic: a guitarist would virtually never name it this way

CONVENTIONALITY = {
    "":          1.00,   # major triad — most conventional possible
    "m":         1.00,   # minor triad
    "5":         0.97,   # power chord
    "maj7":      0.95,
    "m7":        0.95,
    "7":         0.95,
    "dim":       0.92,
    "aug":       0.90,
    "sus2":      0.90,
    "sus4":      0.90,
    "add9":      0.90,
    "madd9":     0.90,
    "add4":      0.88,
    "madd4":     0.88,
    "6":         0.85,
    "m6":        0.85,
    "9":         0.85,
    "maj9":      0.85,
    "m9":        0.85,
    "mmaj7":     0.82,
    "dim7":      0.82,
    "m7b5":      0.82,
    "6/9":       0.80,
    "m6/9":      0.80,
    "6sus4":     0.75,
    "6sus2":     0.73,
    "7sus2":     0.80,
    "7sus4":     0.80,
    "maj7sus2":  0.75,
    "maj7sus4":  0.75,
    "7b9":       0.80,
    "7#9":       0.80,
    "7b5":       0.78,
    "maj7b5":    0.78,
    "aug7":      0.78,
    "augmaj7":   0.78,
    "7b13":      0.75,
    "m7#5":      0.75,
    "mb6":       0.75,
    "-5":        0.75,
    "sus2sus4":  0.75,
    "mmaj9":     0.75,
    "9sus4":     0.75,
    "9#5":       0.72,
    "9b5":       0.72,
    "7(b5,b9)":  0.70,
    "7(b5,#9)":  0.70,
    "7(#5,b9)":  0.70,
    "7(#5,#9)":  0.70,
    "maj11":     0.72,
    "m11":       0.72,
    "7#11":      0.70,
    "maj7#11":   0.70,
    "11b9":      0.68,
    "m11b5":     0.68,
    "maj9#11":   0.68,
    "13":        0.65,
    "maj13":     0.65,
    "m13":       0.65,
    "13#11":     0.62,
    "maj13#11":  0.62,
    "13sus4":    0.62,
    "13b9":      0.60,
    "13#9":      0.60,
    "13b9#11":   0.58,
    "aug13":     0.55,
}
 
QUALITY_INTERVALS = {
    '': {"notes": [0, 4, 7], "required": [0, 4]},
    'm': {"notes": [0, 3, 7], "required": [0, 3]},
    'aug': {"notes": [0, 4, 8], "required": [0, 4, 8]},
    'dim': {"notes": [0, 3, 6], "required": [0, 3, 6], "aliases": ['mb5']},
    'sus2': {"notes": [0, 2, 7], "required": [0, 2]},
    'sus4': {"notes": [0, 5, 7], "required": [0, 5]},
    '5': {"notes": [0, 7], "required": [0, 7]},
    '-5': {"notes": [0, 4, 6], "required": [0, 4, 6]},
    'mb6': {"notes": [0, 3, 8], "required": [0, 3, 8], "aliases": ['maug']},
    '6': {"notes": [0, 4, 7, 9], "required": [0, 4, 9]},
    'm6': {"notes": [0, 3, 7, 9], "required": [0, 3, 9]},
    '6/9': {"notes": [0, 2, 4, 7, 9], "required": [0, 2, 4, 9], "aliases": ['6add9']},
    'm6/9': {"notes": [0, 2, 3, 7, 9], "required": [0, 2, 3, 9], "aliases": ['m6add9']},
    '6sus2': {"notes": [0, 2, 7, 9], "required": [0, 2, 9]},
    '6sus4': {"notes": [0, 5, 7, 9], "required": [0, 5, 9]},
    'maj7': {"notes": [0, 4, 7, 11], "required": [0, 4, 11]},
    'm7': {"notes": [0, 3, 7, 10], "required": [0, 3, 10]},
    '7': {"notes": [0, 4, 7, 10], "required": [0, 4, 10]},
    'mmaj7': {"notes": [0, 3, 7, 11], "required": [0, 3, 11]},
    'aug7': {"notes": [0, 4, 8, 10], "required": [0, 4, 8, 10], "aliases": ['7#5']},
    'augmaj7': {"notes": [0, 4, 8, 11], "required": [0, 4, 8, 11], "aliases": ['maj7#5']},
    'dim7': {"notes": [0, 3, 6, 9], "required": [0, 3, 6, 9]},
    'm7b5': {"notes": [0, 3, 6, 10], "required": [0, 3, 6, 10]},
    '7b5': {"notes": [0, 4, 6, 10], "required": [0, 4, 6, 10]},
    'maj7b5': {"notes": [0, 4, 6, 11], "required": [0, 4, 6, 11]},
    'm7#5': {"notes": [0, 3, 8, 10], "required": [0, 3, 8, 10], "aliases": ['m7b13']},
    '7b13': {"notes": [0, 4, 7, 8, 10], "required": [0, 4, 8, 10]},
    '7sus2': {"notes": [0, 2, 7, 10], "required": [0, 2, 10]},
    '7sus4': {"notes": [0, 5, 7, 10], "required": [0, 5, 10]},
    'maj7sus2': {"notes": [0, 2, 7, 11], "required": [0, 2, 11]},
    'maj7sus4': {"notes": [0, 5, 7, 11], "required": [0, 5, 11]},
    'sus2sus4': {"notes": [0, 2, 5, 7], "required": [0, 2, 5]},
    'add9': {"notes": [0, 2, 4, 7], "required": [0, 2, 4], "aliases": ['add2']},
    'add4': {"notes": [0, 4, 5, 7], "required": [0, 4, 5], "aliases": ['add11']},
    'madd9': {"notes": [0, 2, 3, 7], "required": [0, 2, 3], "aliases": ['madd2']},
    'madd4': {"notes": [0, 3, 5, 7], "required": [0, 3, 5], "aliases": ['madd11']},
    '9': {"notes": [0, 2, 4, 7, 10], "required": [0, 2, 4, 10]},
    'maj9': {"notes": [0, 2, 4, 7, 11], "required": [0, 2, 4, 11]},
    'm9': {"notes": [0, 2, 3, 7, 10], "required": [0, 2, 3, 10]},
    'mmaj9': {"notes": [0, 2, 3, 7, 11], "required": [0, 2, 3, 11]},
    '9#5': {"notes": [0, 2, 4, 8, 10], "required": [0, 2, 4, 8, 10]},
    '9b5': {"notes": [0, 2, 4, 6, 10], "required": [0, 2, 4, 6, 10]},
    '9sus4': {"notes": [0, 2, 5, 7, 10], "required": [0, 2, 5, 10], "aliases": ['11']},
    '7b9': {"notes": [0, 1, 4, 7, 10], "required": [0, 1, 4, 10]},
    '7#9': {"notes": [0, 3, 4, 7, 10], "required": [0, 3, 4, 10]},
    '7b5b9': {"notes": [0, 1, 4, 6, 10], "required": [0, 1, 4, 6, 10], "aliases": ['7(b5,b9)']},
    '7b5#9': {"notes": [0, 3, 4, 6, 10], "required": [0, 3, 4, 6, 10], "aliases": ['7(b5,#9)']},
    '7#5b9': {"notes": [0, 1, 4, 8, 10], "required": [0, 1, 4, 8, 10], "aliases": ['7alt', '7(#5,b9)']},
    '7#5#9': {"notes": [0, 3, 4, 8, 10], "required": [0, 3, 4, 8, 10], "aliases": ['7(#5,#9)']},
    'maj11': {"notes": [0, 2, 4, 5, 7, 11], "required": [0, 4, 5, 11]},
    'm11': {"notes": [0, 2, 3, 5, 7, 10], "required": [0, 3, 5, 10]},
    'maj9#11': {"notes": [0, 2, 4, 6, 7, 11], "required": [0, 2, 4, 6, 11]},
    '11b9': {"notes": [0, 1, 4, 5, 7, 10], "required": [0, 1, 4, 5, 10]},
    '7#11': {"notes": [0, 4, 6, 7, 10], "required": [0, 4, 6, 10]},
    'maj7#11': {"notes": [0, 4, 6, 7, 11], "required": [0, 4, 6, 11]},
    'm11b5': {"notes": [0, 2, 3, 5, 6, 10], "required": [0, 3, 5, 6, 10]},
    '13': {"notes": [0, 2, 4, 5, 7, 9, 10], "required": [0, 4, 9, 10]},
    'maj13': {"notes": [0, 2, 4, 5, 7, 9, 11], "required": [0, 4, 9, 11]},
    'm13': {"notes": [0, 2, 3, 5, 7, 9, 10], "required": [0, 3, 9, 10]},
    'aug13': {"notes": [0, 2, 4, 5, 8, 9, 10], "required": [0, 4, 8, 9, 10]},
    'maj13#11': {"notes": [0, 2, 4, 6, 7, 9, 11], "required": [0, 4, 6, 9, 11]},
    '13#11': {"notes": [0, 2, 4, 6, 7, 9, 10], "required": [0, 4, 6, 9, 10]},
    '13b9': {"notes": [0, 1, 4, 5, 7, 9, 10], "required": [0, 1, 4, 9, 10]},
    '13sus4': {"notes": [0, 2, 5, 7, 9, 10], "required": [0, 5, 9, 10]},
    '13#9': {"notes": [0, 3, 4, 5, 7, 9, 10], "required": [0, 3, 4, 9, 10]},
    '13b9#11': {"notes": [0, 1, 4, 6, 7, 9, 10], "required": [0, 1, 4, 6, 9, 10]},

    # -- Generated directly from quality_registry.json for consistency
    # with voicings.db / songs.py. Excludes 4 extremely rare qualities
    # (7no3, add11no5, m7no5, sus2add13no5 -- 1 real song occurrence each)
    # that the voicing scrape also excluded.
    '#9': {"notes": [0, 3, 4, 7], "required": [0, 3, 4], "aliases": ['(#9)']},
    '11': {"notes": [0, 2, 4, 5, 7, 10], "required": [0, 4, 5, 10]},
    '6add11': {"notes": [0, 4, 5, 7, 9], "required": [0, 4, 5, 9], "aliases": ['6add4']},
    '7add11': {"notes": [0, 4, 5, 7, 10], "required": [0, 4, 5, 10], "aliases": ['7add4']},
    '7add13': {"notes": [0, 4, 7, 9, 10], "required": [0, 4, 9, 10], "aliases": ['7add6']},
    '9#11': {"notes": [0, 2, 4, 6, 7, 10], "required": [0, 2, 4, 6, 10]},
    '9add13': {"notes": [0, 2, 4, 7, 9, 10], "required": [0, 2, 4, 9, 10]},
    '9b13': {"notes": [0, 2, 4, 7, 8, 10], "required": [0, 2, 4, 8, 10]},
    'add#11': {"notes": [0, 4, 6, 7], "required": [0, 4, 6], "aliases": ['(add#11)']},
    'add4add9': {"notes": [0, 2, 4, 5, 7], "required": [0, 2, 4, 5]},
    'add9#11': {"notes": [0, 2, 4, 6, 7], "required": [0, 2, 4, 6]},
    'addb13': {"notes": [0, 4, 7, 8], "required": [0, 4, 8]},
    'b9': {"notes": [0, 1, 4, 7], "required": [0, 1, 4], "aliases": ['(b9)']},
    'm7add11': {"notes": [0, 3, 5, 7, 10], "required": [0, 3, 5, 10], "aliases": ['m7add4']},
    'm7add13': {"notes": [0, 3, 7, 9, 10], "required": [0, 3, 9, 10]},
    'm9#5': {"notes": [0, 2, 3, 8, 10], "required": [0, 2, 3, 8, 10]},
    'maj7#9': {"notes": [0, 3, 4, 7, 11], "required": [0, 3, 4, 11], "aliases": ['maj7(+9)']},
    'maj7add11': {"notes": [0, 4, 5, 7, 11], "required": [0, 4, 5, 11]},
    'maj7add13': {"notes": [0, 4, 7, 9, 11], "required": [0, 4, 9, 11]},
    'maj9add13': {"notes": [0, 2, 4, 7, 9, 11], "required": [0, 2, 4, 9, 11]},
    'mb9': {"notes": [0, 1, 3, 7], "required": [0, 1, 3], "aliases": ['m-9']},
    'mmaj13': {"notes": [0, 2, 3, 5, 7, 9, 11], "required": [0, 3, 9, 11]},
    'sus2add#11': {"notes": [0, 2, 6, 7], "required": [0, 2, 6]},
}
 
ALIASES = {
    quality: data["aliases"]
    for quality, data in QUALITY_INTERVALS.items()
    if "aliases" in data
}

NOTE_NORMALIZE = {
    "A#": "Bb",
    "D#": "Eb",
    "G#": "Ab",
    "Gb": "F#",
    "Db": "C#",
}

def _build_enharmonic_map():
    mapping = {}
    for q in QUALITY_INTERVALS:
        for wrong, right in NOTE_NORMALIZE.items():
            mapping[wrong + q] = right + q
    return mapping
 
ENHARMONIC_MAP = _build_enharmonic_map()
 
 
def normalize_note(note):
    # Normalize a  note name to my spelling (e.g. G# -> Ab)
    return NOTE_NORMALIZE.get(note, note)
 
 
def normalize(chord_str):
    # Normalize a full chord string to canonical note spelling.
    # Handles plain chords ('A#m7') and slash chords ('D#m/Bb')

    if "/" in chord_str:
        parts = chord_str.split("/", 1)
        return normalize(parts[0]) + "/" + normalize_note(parts[1])
    return ENHARMONIC_MAP.get(chord_str, chord_str)
 


# Interval weights, this part sucks
# Defines how much each interval contributes to match scoring, this is needed since in guitar voicings, some intervals are more needed than other to make a chord naming possible, for example the 5th is a lot of times not included, like for a D9 X-5-4-5-5-X, there is no 5th interval which is an A, but its still a D9
# Required-note intervals get 1.5x weight, missing them hurts more
# The 5th (interval 7) is intentionally low: guitarists omit it constantly as noted above

INTERVAL_WEIGHTS = {
    0:  1.5,   # root — critical
    1:  1.2,   # b9 — very characteristic when present
    2:  1.1,   # 9th
    3:  1.4,   # minor 3rd — defines minor quality
    4:  1.4,   # major 3rd — defines major quality
    5:  1.1,   # 4th / 11th
    6:  1.3,   # tritone — very characteristic
    7:  0.5,   # 5th — commonly omitted, low penalty for missing
    8:  1.3,   # #5 / b13 — defines aug
    9:  1.1,   # 6th / 13th
    10: 1.3,   # minor 7th
    11: 1.3,   # major 7th
}


def get_active_intervals(note_vector, root_idx):
    active = set()
    for note_idx, val in enumerate(note_vector):
        if val == 1:
            interval = (note_idx - root_idx) % 12
            active.add(interval)
    return active


def score_quality(active_intervals, quality, note_probs, root_idx):
    # Probability weighted score matching

    data = QUALITY_INTERVALS[quality]
    quality_notes  = set(data["notes"])
    required_notes = set(data["required"])

    score = 0.0
    total_weight = 0.0

    for note_idx in range(12):
        interval = (note_idx - root_idx) % 12
        prob = note_probs[note_idx]
        iw  = INTERVAL_WEIGHTS.get(interval, 1.0)

        # Required notes get an extra multiplier -- this same effective
        # weight (iw * rw) must be used for BOTH the numerator (score) and
        # the denominator (total_weight), otherwise total_weight undercounts
        # relative to what score can accumulate, letting raw_score exceed
        # 1.0 (e.g. a confidence of 1.024).
        rw = 1.5 if interval in required_notes else 1.0
        w  = iw * rw

        if interval in quality_notes:
            score += prob * w
        else:
            # Penalize unexpected active notes (wrong notes for this quality)
            score += (1.0 - prob) * w

        total_weight += w

    raw_score = score / total_weight

    # Apply conventionality bias, this is what prevents Cmaj confusing with Emb6 output
    convention = CONVENTIONALITY.get(quality, 0.60)
    return raw_score * convention


def identify_chord(note_probs, root_idx, threshold=THRESHOLD):
    # Identify chord quality for a given root.
    # Returns best match with method, confidence, and top candidates.

    note_probs = np.array(note_probs, dtype=float)

    note_vector      = (note_probs >= threshold).astype(int)
    active_notes     = [CHROMATIC[i] for i, v in enumerate(note_vector) if v == 1]
    root_name        = CHROMATIC[root_idx]
    active_intervals = get_active_intervals(note_vector, root_idx)

    # First step is required-note matching (handles omitted optionals like 5th)

    SOFT_THRESHOLD = 0.30

    exact_matches = []
    for quality, data in QUALITY_INTERVALS.items():
        required_set = set(data["required"])
        full_set     = set(data["notes"])

        # Check required notes — use soft threshold
        required_ok = all(
            note_probs[(root_idx + interval) % 12] >= SOFT_THRESHOLD
            for interval in required_set
        )
        if not required_ok:
            continue

        # No active (hard-threshold) note should fall outside the full note set
        if not active_intervals.issubset(full_set):
            continue

        exact_matches.append(quality)

    if exact_matches:
        # Among all valid matches, pick highest conventionality-weighted score
        # This is the critical tie-breaker: C major beats Emb6 here
        best_quality = max(
            exact_matches,
            key=lambda q: score_quality(active_intervals, q, note_probs, root_idx)
        )
        chord_name   = normalize(root_name + best_quality)
        alias_names  = [normalize(root_name + a) for a in ALIASES.get(best_quality, [])]
        display_name = " / ".join([chord_name] + alias_names)
        full_set     = set(QUALITY_INTERVALS[best_quality]["notes"])
        method       = "exact" if active_intervals == full_set else "partial"

        return {
            "chord":      display_name,
            "root":       root_name,
            "quality":    best_quality,
            "aliases":    alias_names,
            "confidence": score_quality(active_intervals, best_quality, note_probs, root_idx),
            "method":     method,
            "notes":      active_notes,
            "candidates": [(display_name, 1.0)],
        }

    # Next is weighted scoring fallback (no clean required-note match)
    scores = [
        (quality, score_quality(active_intervals, quality, note_probs, root_idx))
        for quality in QUALITY_INTERVALS
    ]
    scores.sort(key=lambda x: x[1], reverse=True)

    best_quality = scores[0][0]
    best_score   = scores[0][1]
    chord_name   = normalize(root_name + best_quality)
    alias_names  = [normalize(root_name + a) for a in ALIASES.get(best_quality, [])]
    display_name = " / ".join([chord_name] + alias_names)
    candidates   = [(normalize(root_name + q), round(s, 4)) for q, s in scores[:3]]

    return {
        "chord":      display_name,
        "root":       root_name,
        "quality":    best_quality,
        "aliases":    alias_names,
        "confidence": round(best_score, 4),
        "method":     "weighted",
        "notes":      active_notes,
        "candidates": candidates,
    }


def identify_chord_smart(note_logits, root_logits, bass_logits, threshold=THRESHOLD):
  
    # The Main entry point.

    # Fixes the hard-argmax root problem by trying the top 3 root candidates plus the bass note as root. Each candidate's confidence is scaled by how confident the root head was, so a wrong-but-confident root doesn't override a correct-but-less-confident one with a much better note match.

    # Ambiguity handling:
      # - Conventionality bias in score_quality ensures common chords win ties
      # - Bass note used for slash chord / inversion notation
      # - Top alternatives returned for cases of genuine ambiguity

    note_probs = 1 / (1 + np.exp(-np.array(note_logits, dtype=float)))  # sigmoid
    bass_idx   = int(np.argmax(bass_logits))

    # Softmax root logits so confidences sum to 1 and are comparable
    root_exp   = np.exp(np.array(root_logits, dtype=float) - np.max(root_logits))
    root_probs = root_exp / root_exp.sum()

    # Try top 3 roots + bass as root (bass is often correct when root head is wrong)
    top_roots     = list(np.argsort(root_probs)[::-1][:3])
    roots_to_try  = list(dict.fromkeys(top_roots + [bass_idx]))  # deduplicated

    all_candidates = []
    for root_idx in roots_to_try:
        result = identify_chord(note_probs, root_idx, threshold)
        root_conf = float(root_probs[root_idx])
        # Scale by root confidence,a bad root guess can't win on note match alone
        result["confidence"] = result["confidence"] * root_conf
        result["root_conf"]  = round(root_conf, 4)
        all_candidates.append(result)

    all_candidates.sort(key=lambda x: x["confidence"], reverse=True)

    best = all_candidates[0]
    best["bass"]         = CHROMATIC[bass_idx]
    best["alternatives"] = all_candidates[1:3]  # genuine ambiguity candidates

    # Slash chord notation for inversions / pedal tones
    if CHROMATIC[bass_idx] != best["root"]:
        best["chord"] = best["chord"] + "/" + CHROMATIC[bass_idx]

    return best


def identify_chord_from_model_output(note_logits, root_logits, threshold=THRESHOLD):
    # Backwards-compatible wrapper for test scripts
    note_probs = 1 / (1 + np.exp(-np.array(note_logits, dtype=float)))
    root_idx   = int(np.argmax(root_logits))
    return identify_chord(note_probs, root_idx, threshold)


# Tesing it
if __name__ == "__main__":
 
    print("=" * 65)
    print("  MUSIC THEORY - RULE-BASED TEST")
    print("  Using hardcoded model outputs from real wav file tests")
    print("=" * 65)
 
    CHROMATIC_LOCAL = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
 
    def probs_dict_to_array(probs_dict):
        # Returns logits: present notes converted from prob, absent notes at -9.0
        # so sigmoid gives ~0 for notes not in the chord
        def prob_to_logit(p):
            import numpy as np
            p = float(np.clip(p, 1e-6, 1 - 1e-6))
            return float(np.log(p / (1 - p)))
        arr = [-9.0] * 12
        for note, prob in probs_dict.items():
            if note in CHROMATIC_LOCAL:
                arr[CHROMATIC_LOCAL.index(note)] = prob_to_logit(prob)
        return arr
 
    test_cases = [
        # ── PASSING CASES ────────────────────────────────────────────────
        {
            "desc":     "D9 amp recording (expected: D9)",
            "expected": "D9",
            "probs":    {"C": 1.0, "D": 0.984, "E": 1.0, "F#": 0.994, "G": 0.068, "Ab": 0.079},
            "root_idx": CHROMATIC_LOCAL.index("D"),
            "bass_idx": CHROMATIC_LOCAL.index("D"),
        },
        {
            "desc":     "Cmaj mac mic (expected: C)",
            "expected": "C",
            "probs":    {"C": 1.0, "E": 1.0, "G": 1.0},
            "root_idx": CHROMATIC_LOCAL.index("C"),
            "bass_idx": CHROMATIC_LOCAL.index("C"),
        },
        # Ambiguity test: C E G could theoretically be Emb6 — must return C
        {
            "desc":     "Cmaj ambiguity check — must NOT return Emb6 (expected: C)",
            "expected": "C",
            "probs":    {"C": 1.0, "E": 1.0, "G": 1.0},
            "root_idx": CHROMATIC_LOCAL.index("E"),   # root head wrong (said E)
            "bass_idx": CHROMATIC_LOCAL.index("C"),   # bass correctly says C
        },
 
        # ── HARDER CASES ─────────────────────────────────────────────────
        {
            "desc":     "Cmaj7 high pos (expected: Cmaj7)",
            "expected": "Cmaj7",
            "probs":    {"C": 0.638, "E": 1.0, "G": 0.993, "B": 0.744},
            "root_idx": CHROMATIC_LOCAL.index("E"),   # model predicted E (wrong root)
            "bass_idx": CHROMATIC_LOCAL.index("C"),
        },
        {
            "desc":     "Dminor9 mac mic (expected: Dm9)",
            "expected": "Dm9",
            "probs":    {"C": 0.99, "D": 1.0, "E": 00.953, 'F': 0.385},
            "root_idx": CHROMATIC_LOCAL.index("D"),
            "bass_idx": CHROMATIC_LOCAL.index("D"),
        },
        {
            "desc":     "F#maj9 mac mic (expected: F#maj9)",
            "expected": "F#maj9",
            "probs":    {"C#": 0.926, "F": 1.0, "F#": 0.363, "Ab": 0.592, "Bb": 0.652},
            "root_idx": CHROMATIC_LOCAL.index("Bb"),  # model predicted Bb (wrong root)
            "bass_idx": CHROMATIC_LOCAL.index("Bb"),
        },
        {
            "desc":     "Dm7b5 mac mic (expected: Dm7b5)",
            "expected": "Dm7b5",
            "probs":    {"C": 0.84, "C#": 0.318, "Eb": 0.363, "F": 0.31, "Ab": 0.371},
            "root_idx": CHROMATIC_LOCAL.index("Eb"),  # model predicted Eb (wrong root)
            "bass_idx": CHROMATIC_LOCAL.index("Bb"),
        },
    ]
 
    # Build root logits: spike at primary root, but also give secondary root
    # some signal so the multi-root search can find it.
    # In real inference these come from the model's softmax head.
    def make_root_logits(root_idx, secondary_idx=None, spike=3.0, secondary=1.2):
        logits = [-2.0] * 12
        logits[root_idx] = spike
        if secondary_idx is not None and secondary_idx != root_idx:
            logits[secondary_idx] = secondary
        return logits
 
    correct = 0
    total   = 0
 
    for case in test_cases:
        note_probs_arr = probs_dict_to_array(case["probs"])
        root_idx       = case["root_idx"]
        bass_idx       = case["bass_idx"]
        expected       = case["expected"]
        desc           = case["desc"]
 
        # Use identify_chord_smart so multi-root logic kicks in
        root_logits = make_root_logits(root_idx, secondary_idx=bass_idx)
        bass_logits = make_root_logits(bass_idx)
        result = identify_chord_smart(note_probs_arr, root_logits, bass_logits)
 
        chord  = result["chord"]
        method = result["method"]
        conf   = result["confidence"]
        notes  = result["notes"]
        alts   = [a["chord"] for a in result.get("alternatives", [])]
 
        is_correct = chord.lower().replace("/","").startswith(expected.lower()[:4])
        status     = "✓" if is_correct else "✗"
        total     += 1
        if is_correct:
            correct += 1
 
        # Three-layer note breakdown
        note_probs_named = {CHROMATIC_LOCAL[i]: round(float(note_probs_arr[i]), 3) for i in range(12)}
        active_notes = [k for k, v in note_probs_named.items() if v >= THRESHOLD]
        soft_notes   = {k: v for k, v in note_probs_named.items() if 0.30 <= v < THRESHOLD}
 
        # chord may be a DISPLAY string like "A6/9 / A6add9" (primary name
        # + alias, joined with " / ") -- never re-parse that. Build a
        # clean, single, parseable chord name from the structured fields
        # instead, same fix applied to main.py for the identical bug.
        clean_bass = result["bass"] if result["bass"] != result["root"] else None
        clean_chord_name = cp.format_chord(result["root"], result["quality"], clean_bass)

        from chord_info import get_chord_info
        info         = get_chord_info(clean_chord_name) if chord else None
        theory_notes = [iv["note"] for iv in info["intervals"]] if info else []
 
        print(f"  {status} {desc}")
        print(f"      Expected     : {expected}")
        print(f"      Got          : {chord}  [method={method}  conf={conf:.3f}]  Bass: {result['bass']}")
        print(f"      [1] Active notes  (>={THRESHOLD}) : {active_notes}")
        if soft_notes:
            print(f"      [2] Soft notes   (0.30-{THRESHOLD}) : "
                  + "  ".join(f"{k}={v:.3f}" for k, v in sorted(soft_notes.items(), key=lambda x: -x[1])))
        else:
            print(f"      [2] Soft notes   (0.30-{THRESHOLD}) : none")
        print(f"      [3] Theory notes (chord {clean_chord_name}) : {theory_notes}")
        if alts:
            print(f"      Alternatives : {alts}")
        print()
 
    print(f"  {'─' * 60}")
    print(f"  Result: {correct}/{total} correct")
    print("=" * 65)