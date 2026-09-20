"""
Chord tokenizer / normalizer for betterchord_songs.db

Design: parse each chord string into components instead of treating it as
an opaque string:

    ROOT [accidental]  QUALITY-BLOB  [ / BASS [accidental] ]

Then decompose the quality-blob into:
    - base quality   : maj | min | dim | aug | sus2 | sus4 | 5 (power chord)
    - seventh type    : none | 7 | maj7 | dim7  (min7 is implied by base=min + 7)
    - extension       : none | 9 | 11 | 13   (the "highest number" e.g. m11 implies 7,9,11 stacked)
    - alterations     : set of {b5, #5, b9, #9, #11, b13}
    - adds            : set of {add2, add4, add6, add9, add11, add13}
    - no3 / no5       : bool flags (rare, e.g. "5" power chord = no3 implicitly)

Anything that doesn't fit a recognized pattern is returned as UNPARSED with
the reason, rather than being silently guessed at.
"""

import re
import json

# Canonical alteration/add tables live in interval_calculator.py (one-way
# import -- that module doesn't import this one). Reused here, not
# duplicated -- a second, independent copy of these tables would risk
# drifting out of sync with the real one; parse_quality() validates
# alteration/add tokens against them so an unrecognized token fails
# parsing instead of silently no-opping downstream.
from interval_calculator import ALTERATION_MAP, ADD_INTERVAL

# ---------------------------------------------------------------------------
# 1. Root / bass note canonicalization
# ---------------------------------------------------------------------------

def format_chord(root, quality_name, bass=None):
    """Build a canonical chord string from root + quality name, avoiding a
    real ambiguity: if quality_name starts with 'b' or '#' (a bare
    alteration with no base-quality letter in front of it, e.g. the 'b9' in
    'G(b9)'), concatenating it directly (root + quality_name = 'Gb9') reads
    back as root='Gb' (G-flat) + quality='9' -- a completely different
    chord, because FULL_RE's root-accidental group greedily consumes a
    trailing 'b'/'#' as part of the root. Wrapping in parens sidesteps
    this exactly the way real chord notation does ('G(b9)'), and
    parse_quality() already strips parens before parsing the blob, so this
    round-trips correctly.
    """
    if quality_name and quality_name[0] in ("b", "#"):
        s = root + "(" + quality_name + ")"
    else:
        s = root + quality_name
    if bass:
        s += "/" + bass
    return s


CHROMATIC = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

# Every enharmonic spelling we might encounter -> canonical spelling from CHROMATIC
NOTE_ALIASES = {
    "C": "C", "B#": "C",
    "C#": "C#", "Db": "C#",
    "D": "D",
    "D#": "Eb", "Eb": "Eb",
    "E": "E", "Fb": "E",
    "E#": "F", "F": "F",
    "F#": "F#", "Gb": "F#",
    "G": "G",
    "G#": "Ab", "Ab": "Ab",
    "A": "A",
    "A#": "Bb", "Bb": "Bb",
    "B": "B", "Cb": "B",
    # German notation: H is always B-natural, unambiguous in this corpus
    "H": "B",
}

NOTE_RE = r"(?:[A-H])(?:#|b)?"  # matches any root/bass spelling we alias, incl. H


def canon_note(raw):
    """Canonicalize a raw note spelling (e.g. 'D#', 'Eb', 'H') to the
    project's chromatic spelling. Returns None if not a recognized note."""
    return NOTE_ALIASES.get(raw)


# ---------------------------------------------------------------------------
# 2. Pre-cleaning: unicode symbols, stray artifacts, whitespace
# ---------------------------------------------------------------------------

SYMBOL_SUBS = [
    (r"\s+", ""),           # strip all internal/edge whitespace
    (r"[△Δ∆]", "maj"),      # jazz major-7 triangle -> maj  (context finishes the "7")
    (r"[øØ]", "m7b5"),
    (r"°", "dim"),
    # "+" is NOT substituted to "aug" here -- parse_quality() handles bare
    # "+", digit-attached "+5"/"-5", and trailing bare "+"/"-" more
    # precisely. A blanket "+" -> "aug" would break "F7+", which must
    # become "F7" with alteration {"#5"}, not "F7aug".
]

# characters that indicate outright junk / scrape garbage if left over
JUNK_CHARS = set("~\\*")


def preclean(token):
    """Return (cleaned_token, had_junk_chars: bool)"""
    had_junk = any(c in JUNK_CHARS for c in token)
    t = token
    for c in JUNK_CHARS:
        t = t.replace(c, "")
    for pattern, replacement in SYMBOL_SUBS:
        t = re.sub(pattern, replacement, t)
    t = t.strip()
    return t, had_junk


# ---------------------------------------------------------------------------
# 3. Quality-blob normalization
# ---------------------------------------------------------------------------
# Ordered list of (regex, replacement) applied to the quality-blob (the part
# of the token between the root and an optional /bass). Order matters: more
# specific patterns first.

# Word-level aliases (case-insensitive), applied first, longest-match first
WORD_ALIASES = [
    (r"(?i)\bmaj\b", "maj"),
    (r"(?i)maj", "maj"),      # MAJ, Maj, maj -> maj  (word, not the bare letter M)
    (r"(?i)ma(?=\d)", "maj"), # "ma7" -> "maj7" (regional shorthand), guarded so it
                              # never fires on "maj" itself (there 'ma' is followed by 'j', not a digit)
    (r"(?i)\bmin\b", "min"),
    (r"(?i)\bmi\b", "min"),
    (r"(?i)\bsus\b(?!\d)", "sus4"),   # bare "sus" defaults to sus4 (common convention)
    (r"(?i)\baugmented\b", "aug"),
    (r"(?i)\bdiminished\b", "dim"),
    (r"(?i)\bhalfdim\b", "m7b5"),
    (r"(?i)\badd\s+", "add"),  # "add 9" -> "add9"
]

# Matches the trailing alteration/add/sus tokens parse_quality() scans for
# after stripping the base-quality prefix and numeric extension off the
# front of the blob (e.g. "add9", "sus2", "sus4", bare "sus", "#11", "b9").
# Module-level so external callers can detect the same ambiguous-shorthand
# case parse_quality() resolves, without a second drifting copy of the
# pattern. Note: WORD_ALIASES' `\bsus\b(?!\d)` entry only fires when "sus"
# has a real word boundary before it (a bare "Csus"), never inside a
# compound blob like "maj7sus" -- the compound case is resolved by this
# pattern's bare `sus` alternative in parse_quality()'s scanning loop.
ALT_PATTERN = re.compile(
    r"(add[#b]?\d+|sus2|sus4|sus|[#b]\d+|[+-]\d+|no3|no5|[+-]$)"
)


def parse_quality(blob):
    """
    Parse the quality-blob into a structured dict, or return None if it
    doesn't match any recognized chord grammar.
    """
    original = blob
    b = blob

    # normalize case-insensitive words first
    for pat, rep in WORD_ALIASES:
        b = re.sub(pat, rep, b)

    # strip surrounding parens around alterations: "(b5)" -> "b5", "(#9)" -> "#9"
    b = re.sub(r"\(([^()]*)\)", r"\1", b)

    result = {
        "base": None,       # 'maj' | 'min' | 'dim' | 'aug' | 'sus2' | 'sus4' | '5'
        "seventh": None,    # '7' | 'maj7' | 'dim7'
        "ext": None,        # '9' | '11' | '13'  (highest stacked number)
        "alterations": set(),
        "adds": set(),
        "no3": False,
        "no5": False,
        "six": False,       # add6 / 6th chord (not stacked add, the classic "6" chord)
        "sixnine": False,   # 6/9 chord
    }

    # ---- empty blob => plain major triad
    if b == "":
        result["base"] = "maj"
        return result

    # ---- power chord: "5"
    if b == "5":
        result["base"] = "5"
        result["no3"] = True
        return result

    # ---- bare dash = minor triad (e.g. "F-" -> Fm); distinct from "-5"/"-9"
    # digit-attached alterations, which are handled later in alt_pat.
    if b == "-":
        result["base"] = "min"
        return result

    # ---- bare "+" = augmented triad (e.g. "C+" -> Caug), modeled as a
    # major triad with a #5 alteration so the later equivalence-grouping
    # step can unify it with "aug", "maj#5", etc.
    if b == "+":
        result["base"] = "maj"
        result["alterations"].add("#5")
        return result

    # ---- bare "2" / "4" = sus2 / sus4 shorthand (e.g. "E4" -> Esus4)
    if b == "2":
        result["base"] = "sus2"
        return result
    if b == "4":
        result["base"] = "sus4"
        return result

    # ---- Brazilian/Portuguese "7M" notation for major 7th (e.g. "C7M" -> Cmaj7)
    b = re.sub(r"^7M$", "maj7", b)
    b = re.sub(r"^9M$", "maj9", b)
    b = re.sub(r"^13M$", "maj13", b)

    # ---- 6/9 chord (must check before bare "6")
    if re.fullmatch(r"(m|min)?6/?9|(m|min)?69", b):
        result["base"] = "min" if b.startswith(("m", "min")) else "maj"
        result["sixnine"] = True
        return result

    # ---- Work out base quality prefix
    # order matters: check multi-char before single-char
    # third element = dim7-flag marker; fourth = whether this pattern is an
    # EXPLICIT major-quality marker (vs. the no-letter-at-all default, which
    # must NOT be treated as explicit -- "C7" is dominant 7, not maj7)
    base_patterns = [
        (r"^dim7", "dim", "dim7-flag", False),
        (r"^dim", "dim", None, False),
        (r"^(o|O)7", "dim", "dim7-flag", False),
        (r"^(o|O)(?!\w)", "dim", None, False),
        (r"^aug", "aug", None, False),
        (r"^m7b5", "m7b5", None, False),           # half-diminished spelled directly
        (r"^sus2", "sus2", None, False),
        (r"^sus4", "sus4", None, False),
        (r"^min", "min", None, False),
        (r"^m(?!aj)", "min", None, False),          # 'm' not followed by "aj" (avoid eating "maj")
        (r"^maj", "maj", None, True),
        (r"^M(?![0-9]*\d?9sus)", "maj", None, True),  # bare capital M = major indicator
    ]

    rest = b
    base = None
    explicit_major = False
    for pat, label, flag, is_explicit_major in base_patterns:
        m = re.match(pat, rest)
        if m:
            base = label
            explicit_major = is_explicit_major
            rest = rest[m.end():]
            if flag == "dim7-flag":
                result["seventh"] = "dim7"
            break

    if base is None:
        # no explicit quality letter -> defaults to major, rest is the
        # numeric/alteration tail (e.g. "7", "9", "13", "#11")
        base = "maj"

    # "mmaj7" / "minmaj7": minor triad + major 7th (a real, distinct quality).
    # "augmaj7": augmented triad + major 7th (also real, e.g. Caugmaj7).
    # "mM7" / "mM9": same minor-major7 combo, using bare capital M instead of
    # the full "maj" word (a real, distinct notation convention -- e.g.
    # Stevie Wonder's "D#mM7" chart, EmM9, etc.) -- distinct from the 'maj'
    # word check below, so both spellings of the same quality are recognized.
    # The base-loop above already consumed the bare 'm'/'min'/'aug' into
    # base; if 'maj' or bare 'M' still follows, it's one of these combos,
    # not a second major quality collision.
    explicit_maj_seventh = explicit_major
    if base in ("min", "aug") and rest.startswith("maj"):
        rest = rest[len("maj"):]
        explicit_maj_seventh = True
    elif base in ("min", "aug") and re.match(r"^M(?=\d)", rest):
        rest = rest[1:]
        explicit_maj_seventh = True


    if base == "m7b5":
        result["base"] = "min"
        result["seventh"] = "dim7" if result["seventh"] else "m7b5"
        result["alterations"].add("b5")
        base = "min"
        result["seventh"] = "7"
        result["alterations"].add("b5")

    result["base"] = base if base not in ("m7b5",) else "min"

    # ---- pull off numeric extension (6,7,9,11,13) right after quality
    m = re.match(r"^(6|7|9|11|13)", rest)
    if m:
        num = m.group(1)
        rest = rest[m.end():]
        if num == "6":
            result["six"] = True
        elif num == "7":
            result["seventh"] = "maj7" if explicit_maj_seventh else "7"
        else:  # 9, 11, 13 stacked extension (implies m7/dom7/maj7 beneath)
            result["ext"] = num
            result["seventh"] = "maj7" if explicit_maj_seventh else "7"

    # ---- remaining alterations / adds, e.g. "#5", "b9", "#11", "add9",
    # "+5"/"-5" (alternate sharp/flat notation), bare trailing "+"/"-"
    # (shorthand for #5/b5, e.g. "F7+" == "F7#5")
    alt_pat = ALT_PATTERN
    ALT_NORMALIZE = {
        "+5": "#5", "-5": "b5",
        "+9": "#9", "-9": "b9",
        "+11": "#11", "-11": "b11",
        "+13": "#13", "-13": "b13",
        "+": "#5", "-": "b5",  # bare trailing shorthand
    }
    pos = 0
    unknown_tail = []
    for m in alt_pat.finditer(rest):
        if m.start() != pos:
            unknown_tail.append(rest[pos:m.start()])
        token = m.group(1)
        if token.startswith("add"):
            # ALT_PATTERN's `add[#b]?\d+` matches any digit ("add3",
            # "add8", ...), not just the theory-recognized add degrees.
            # Validate against ADD_INTERVAL (the table that actually
            # applies it) so an unreal add degree fails parsing (-> leftover
            # -> None) instead of silently no-opping in compute_intervals().
            if token in ADD_INTERVAL:
                result["adds"].add(token)
            else:
                unknown_tail.append(token)
        elif token in ("sus2", "sus4"):
            # A second sus token means both are present (e.g. "7sus2sus4" or
            # "sus2sus4" itself) -- combine rather than overwrite, or the
            # first sus tone silently disappears.
            if result["base"] in ("sus2", "sus4") and result["base"] != token:
                result["base"] = "sus2sus4"
            else:
                result["base"] = token
        elif token == "sus":
            if result["base"] in ("sus2", "sus4") and result["base"] != "sus4":
                result["base"] = "sus2sus4"
            else:
                result["base"] = "sus4"
        elif token in ("no3", "no5"):
            result[token] = True
        elif token in ALT_NORMALIZE:
            result["alterations"].add(ALT_NORMALIZE[token])
        else:
            # Same as the "add" branch, for the [#b]\d+ / [+-]\d+
            # alternative -- "C+7", "C7#3", "Cm7b4" etc. match the regex
            # but aren't real alterations. compute_intervals() silently
            # skips any token not in ALTERATION_MAP, so "C+7" would resolve
            # to a plain C major triad with the "+7" lost. Validate here so
            # it fails parsing instead, per this file's docstring intent
            # (unrecognized patterns are UNPARSED, not guessed at).
            if token in ALTERATION_MAP:
                result["alterations"].add(token)
            else:
                unknown_tail.append(token)
        pos = m.end()
    if pos != len(rest):
        unknown_tail.append(rest[pos:])
    leftover = "".join(unknown_tail)

    if leftover:
        return None  # could not fully account for the string -> unparsed

    return result


# ---------------------------------------------------------------------------
# 4. Full chord parse: root + quality + bass
# ---------------------------------------------------------------------------

FULL_RE = re.compile(
    rf"^(?P<root>[A-H])(?P<root_acc>#|b)?(?P<quality>.*?)(?:/(?P<bass>[A-H])(?P<bass_acc>#|b)?)?$"
)

SOLFEGE_MAP = {"DO": "C", "RE": "D", "MI": "E", "FA": "F", "SOL": "G", "LA": "A", "SI": "B"}
SOLFEGE_RE = re.compile(r"(?i)^(DO|RE|MI|FA|SOL|LA|SI)")


def looks_like_solfege(token):
    """Heuristic: token's letters (ignoring digits/accidentals/quality suffix)
    start with a solfège syllable and the token does NOT begin with a valid
    English note letter (A-H) -- avoids colliding with real chords like
    'Fadd9' or 'Ab'. Returns the guessed English root or None."""
    m = SOLFEGE_RE.match(token)
    if not m:
        return None
    first_char = token[0].upper()
    if first_char in "ABCDEFGH":
        # letter-ambiguous case (D, F starts overlap with valid English roots);
        # only treat as solfège if the standard parse of this exact token
        # already failed elsewhere -- caller handles that ordering.
        pass
    return SOLFEGE_MAP[m.group(1).upper()]


def parse_chord(raw_token):
    token, had_junk_chars = preclean(raw_token)

    if token == "":
        return {"raw": raw_token, "parsed": False, "reason": "empty_after_clean"}

    m = FULL_RE.match(token)
    if not m:
        guess = looks_like_solfege(token)
        if guess:
            return {"raw": raw_token, "parsed": False, "reason": "solfege_ambiguous",
                     "solfege_guess_root": guess}
        return {"raw": raw_token, "parsed": False, "reason": "no_root_match"}

    root_raw = m.group("root") + (m.group("root_acc") or "")
    root = canon_note(root_raw)
    if root is None:
        return {"raw": raw_token, "parsed": False, "reason": "bad_root"}

    bass = None
    if m.group("bass"):
        bass_raw = m.group("bass") + (m.group("bass_acc") or "")
        bass = canon_note(bass_raw)
        if bass is None:
            return {"raw": raw_token, "parsed": False, "reason": "bad_bass"}

    quality_blob = m.group("quality")

    # Special case: "DO" is a valid English parse (D + dim = "Ddim") AND a
    # valid solfège word (Do = C major). Genuinely ambiguous -- don't guess.
    if token.upper() == "DO":
        return {"raw": raw_token, "parsed": False, "reason": "solfege_ambiguous",
                "solfege_guess_root": "C", "english_alt_parse": "Ddim"}

    quality = parse_quality(quality_blob)
    if quality is None:
        # Before giving up, check the "DO"-style collision: a token that
        # parses under English rules (e.g. "DO" -> D + dim) AND also looks
        # like solfège. We already returned a valid parse above in that
        # case (English wins) -- this branch only fires when English fully
        # failed, so also check solfège here for tokens like "FA#5" where
        # the English attempt (root F, quality "A#5") fails outright.
        guess = looks_like_solfege(token)
        if guess:
            return {"raw": raw_token, "parsed": False, "reason": "solfege_ambiguous",
                     "solfege_guess_root": guess}
        return {"raw": raw_token, "parsed": False, "reason": "bad_quality",
                "root": root, "quality_blob": quality_blob}

    return {
        "raw": raw_token,
        "parsed": True,
        "root": root,
        "bass": bass,
        "quality": quality,
        "had_junk_chars": had_junk_chars,
        # The literal, as-typed quality substring (e.g. "7#5"), before any
        # alias/canonical-name resolution -- for callers that need what
        # the user typed vs. the canonical name it resolves to (see
        # songs.py's resolve_query()/get_songs() "searched_quality").
        "quality_blob": quality_blob,
    }