"""
interval_calculator.py

Turns a structural chord-quality dict (from chord_parser.parse_quality) into
two things:
  - full_intervals     : every pitch class (0-11, root-relative) the chord
                          theoretically contains
  - required_intervals : the subset that MUST be present for the label to
                          apply (the "guide tones") -- the rest is omittable
                          on a 6-string guitar without changing the name

This is a rule engine, not a lookup table, so it can produce an answer for
ANY parsed structure -- including ones that don't exist yet in
music_theory.py's hand-tuned QUALITY_INTERVALS. Where music_theory.py
already has a tuned entry for an equivalent quality, that's compared
against (not silently overridden) by registry_builder.py.

Rules (in application order):
  1. Base triad/dyad sets the starting full+required set.
  2. Seventh (7 / maj7 / dim7) adds one interval, always required.
  3. Extension (9/11/13) adds the NAMED interval as required, and the
     lower "implied" extensions (9 within 11/13, 11 within 13) as
     OMITTABLE -- this is the C13-omits-9-and-11 rule from our discussion.
  4. six / sixnine add their interval(s), always required (naming a 6th
     chord is deliberate, there's nothing to omit).
  5. adds (add9, add11, ...) insert one specific tone, always required.
  6. Alterations (#5, b5, #9, b9, #11, b13, ...) REPLACE their natural
     unaltered counterpart (an altered tone is always required -- that's
     the entire point of naming it).
  7. A plain, UNALTERED 5th is downgraded from required to omittable
     unless the base chord IS the 5th doing the identity work (power
     chord, aug, dim, or an explicit b5/#5 alteration already handled
     the removal in step 6).
  8. no3 / no5 flags strip those tones entirely.
"""

BASE_INTERVALS = {
    "maj":  {"full": [0, 4, 7], "required": [0, 4]},
    "min":  {"full": [0, 3, 7], "required": [0, 3]},
    "dim":  {"full": [0, 3, 6], "required": [0, 3, 6]},
    "aug":  {"full": [0, 4, 8], "required": [0, 4, 8]},
    "sus2": {"full": [0, 2, 7], "required": [0, 2]},
    "sus4": {"full": [0, 5, 7], "required": [0, 5]},
    "sus2sus4": {"full": [0, 2, 5, 7], "required": [0, 2, 5]},
    "5":    {"full": [0, 7],    "required": [0, 7]},
}

SEVENTH_INTERVAL = {"7": 10, "maj7": 11, "dim7": 9}

# alteration -> (new interval to add, natural interval it replaces if present)
ALTERATION_MAP = {
    "b5":  (6, 7),
    "#5":  (8, 7),
    "b6":  (8, 7),   # enharmonic to #5 as a triad alteration (e.g. "mb6" == minor-augmented)
    "#6":  (10, 9),
    "b9":  (1, 2),
    "#9":  (3, 2),
    "b11": (4, 5),
    "#11": (6, 5),
    "b13": (8, 9),
    "#13": (10, 9),
}

ADD_INTERVAL = {
    "add2": 2, "add9": 2,
    "add4": 5, "add11": 5,
    "add6": 9, "add13": 9,
    # Altered-degree adds -- "add(b9)" means "keep the plain triad and add
    # a flat-9 on top", distinct from a b9 ALTERATION (which replaces an
    # existing natural 9th).
    "addb5": 6, "add#5": 8,
    "addb9": 1, "add#9": 3,
    "addb11": 4, "add#11": 6,
    "addb13": 8, "add#13": 10,
}


def compute_intervals(q):
    """q is the 'quality' dict produced by chord_parser.parse_quality().
    Returns (full_intervals: sorted list, required_intervals: sorted list)."""

    base = q["base"] if q["base"] in BASE_INTERVALS else "maj"
    full = set(BASE_INTERVALS[base]["full"])
    required = set(BASE_INTERVALS[base]["required"])

    # -- seventh --
    if q["seventh"]:
        iv = SEVENTH_INTERVAL.get(q["seventh"])
        if iv is not None:
            full.add(iv)
            required.add(iv)

    # -- extension (9/11/13): named tone required, implied lower ones omittable --
    if q["ext"] == "9":
        full.add(2); required.add(2)
    elif q["ext"] == "11":
        full.update({2, 5}); required.add(5)          # 9 implied but omittable
    elif q["ext"] == "13":
        full.update({2, 5, 9}); required.add(9)        # 9, 11 implied but omittable

    # -- six / sixnine --
    if q["six"]:
        full.add(9); required.add(9)
    if q["sixnine"]:
        full.update({2, 9}); required.update({2, 9})

    # -- explicit adds --
    for a in q["adds"]:
        iv = ADD_INTERVAL.get(a)
        if iv is not None:
            full.add(iv); required.add(iv)

    # -- alterations: replace the natural tone, altered tone is always required --
    for alt in q["alterations"]:
        if alt in ALTERATION_MAP:
            new_iv, natural_iv = ALTERATION_MAP[alt]
            full.discard(natural_iv); required.discard(natural_iv)
            full.add(new_iv); required.add(new_iv)
        else:
            # Unreachable for any `q` from chord_parser.parse_quality() --
            # it validates alteration tokens against this exact map before
            # accepting one. A defensive silent no-op for a hand-built `q`:
            # the token stays in q["alterations"] but never touches
            # full/required.
            pass

    # -- no3 / no5 --
    if q.get("no3"):
        full -= {3, 4}; required -= {3, 4}
    if q.get("no5"):
        full -= {6, 7, 8}; required -= {6, 7, 8}

    # -- plain unaltered 5th is omittable unless it IS the chord's identity --
    # (power chord / aug / dim / sus already have 7 doing real identity work,
    # or already replaced via alteration in step above -- so this only
    # downgrades the "plain perfect 5th sitting in a maj/min triad" case)
    if 7 in required and base in ("maj", "min") and not q["alterations"] & {"b5", "#5"}:
        required.discard(7)

    return sorted(full), sorted(required)


# ---------------------------------------------------------------------------
# guide_tone_formula() -- a labeled, structural breakdown of a chord's real
# formula (root/third-or-sus/fifth/seventh/extensions), each tone named in
# this app's voicings.db interval-string vocabulary.
#
# It walks the SAME structured quality fields (q) and lookup tables as
# compute_intervals() (BASE_INTERVALS, SEVENTH_INTERVAL, ALTERATION_MAP,
# ADD_INTERVAL), recording a LABEL at each step, rather than reading labels
# off compute_intervals()'s flat output set. The flat set discards which
# step produced each semitone, and several semitones are ambiguous without
# it: semitone 9 is a dim7's seventh OR a 13th's natural pitch class
# (a flat-set check would tag Cm13/C13/Cmaj13 with a bogus "dim7");
# semitone 6 is b5 OR #11 (7#11 has both 6 and 7 present); semitone 8 is
# #5 OR b13. `full_intervals` is used only as a light cross-check for
# root/plain-5th presence.
# ---------------------------------------------------------------------------

_SEVENTH_LABEL = {"7": "b7", "maj7": "maj7", "dim7": "dim7"}

# Extension-region semitone -> label, used only for tagging the specific
# semitone an "add"/alteration step targets -- never a bare flat set.
_EXT_SEMITONE_LABEL = {1: "b9", 2: "9", 3: "#9", 5: "11", 6: "#11", 9: "13", 10: "#13"}

# ADD_INTERVAL's keys encode the naming intent distinctly ("add4" and
# "add11" both resolve to semitone 5 but a player names the tone "4" vs
# "11") -- labeled from the add's own key, not reverse-derived from its
# target semitone.
_ADD_LABEL = {
    "add2": "9", "add9": "9",
    "add4": "4", "add11": "11",
    "add6": "6", "add13": "13",
    "addb5": "b5", "add#5": "#5",
    "addb9": "b9", "add#9": "#9",
    "addb11": "b11", "add#11": "#11",
    "addb13": "b13", "add#13": "#13",
}


def guide_tone_formula(q, full_intervals, required_intervals=None):
    """Structural, labeled breakdown of a chord's real formula:
      { root: "1",
        third: "m3"|"3"|None,      # None when this is a sus chord (see `sus`)
        sus:   ["sus2"]|["sus4"]|["sus2","sus4"]|[],
        fifth: "b5"|"5"|"#5"|None,
        seventh: "b7"|"maj7"|"dim7"|None,
        extensions: ["9","#11","13",...],   # ordered, de-duplicated
        omittable: ["5", ...] }   # see below, only present when
                                   # required_intervals is given

    `third`/`sus` are mutually exclusive by construction -- a chord's
    formula can never show both, they occupy the same structural slot.
    A bucket/list is empty/None when that slot has no tone in THIS
    chord's formula at all (e.g. a plain triad's `extensions` is `[]`,
    an augmented triad's `seventh` is `None`) -- distinct from a specific
    VOICING merely omitting a tone the formula does have (that's the
    frontend's separate per-voicing omitted-tones check).

    `q` is the parsed quality dict from chord_parser.parse_quality() --
    the same one already passed to compute_intervals(); `full_intervals`
    is that call's own first return value, used only as a light
    cross-check below.

    `required_intervals`: compute_intervals()'s second return value -- the
    guide-tone subset that must be present for the name to apply; the rest
    is omittable on a 6-string guitar without changing the name. Optional
    and additive (default None -> no `omittable` key, so 2-arg callers are
    unaffected). When given, each labeled slot above is cross-checked
    against it via the fixed semitone the label denotes (m3=3, 3=4, b5=6,
    5=7, #5=8, `SEVENTH_INTERVAL` for the sevenths). Root and sus tones are
    never omittable (every BASE_INTERVALS entry keeps them in `required`).
    Extensions use compute_intervals()'s "implied lower extension is
    omittable" rule directly (ext=="11" -> 9th omittable; ext=="13" -> 9th
    and 11th) -- six/sixnine/adds/alterations always land in `required`,
    so this narrower rule covers every real omittable-extension case.
    """
    base = q["base"] if q["base"] in BASE_INTERVALS else "maj"
    present = set(full_intervals)

    # -- third / sus --
    third = None
    sus = []
    if base in ("sus2", "sus4", "sus2sus4"):
        if base in ("sus2", "sus2sus4"):
            sus.append("sus2")
        if base in ("sus4", "sus2sus4"):
            sus.append("sus4")
    elif not q.get("no3"):
        if 3 in present:
            third = "m3"
        elif 4 in present:
            third = "3"

    # -- fifth: determined structurally (base identity + an explicit
    # b5/#5/b6 alteration token), not by semitone presence -- a #11
    # alteration also lands on semitone 6 without touching the fifth, so
    # presence-only detection would misfire. --
    fifth = None
    if not q.get("no5"):
        if base == "dim":
            fifth = "b5"
        elif base == "aug":
            fifth = "#5"
        elif 7 in present or base in ("maj", "min", "sus2", "sus4", "sus2sus4", "5"):
            fifth = "5"
        if "b5" in q["alterations"]:
            fifth = "b5"
        elif q["alterations"] & {"#5", "b6"}:
            fifth = "#5"

    # -- seventh: read q["seventh"] directly -- "7"/"maj7"/"dim7" is
    # unambiguous by construction, never reverse-derived from a semitone
    # that could also mean an extension. --
    seventh = _SEVENTH_LABEL.get(q["seventh"]) if q["seventh"] else None

    # -- extensions: ext / six / sixnine / adds / alterations that don't
    # target the fifth (those are already handled above). Each
    # contributes a named tone; order preserved, de-duplicated. --
    extensions = []

    def add_ext(label):
        if label and label not in extensions:
            extensions.append(label)

    if q["ext"] == "9":
        add_ext("9")
    elif q["ext"] == "11":
        add_ext("9")   # implied by the 11th, omittable in practice -- still
        add_ext("11")  # part of the chord's real formula (see compute_intervals)
    elif q["ext"] == "13":
        add_ext("9")   # implied
        add_ext("11")  # implied
        add_ext("13")

    if q["six"]:
        add_ext("6")
    if q["sixnine"]:
        add_ext("9")
        add_ext("6")

    for a in q["adds"]:
        add_ext(_ADD_LABEL.get(a))

    for alt in q["alterations"]:
        if alt in ("b5", "#5", "b6"):
            continue  # already resolved into `fifth` above, not an extension
        if alt not in ALTERATION_MAP:
            continue
        new_iv, natural_iv = ALTERATION_MAP[alt]
        natural_label = _EXT_SEMITONE_LABEL.get(natural_iv)
        if natural_label in extensions:
            extensions.remove(natural_label)
        add_ext(_EXT_SEMITONE_LABEL.get(new_iv, alt))

    omittable = None
    if required_intervals is not None:
        required_set = set(required_intervals)
        omittable = []
        _THIRD_SEMITONE = {"m3": 3, "3": 4}
        _FIFTH_SEMITONE = {"b5": 6, "5": 7, "#5": 8}
        if third and _THIRD_SEMITONE[third] not in required_set:
            omittable.append(third)
        if fifth and _FIFTH_SEMITONE[fifth] not in required_set:
            omittable.append(fifth)
        if seventh and SEVENTH_INTERVAL[q["seventh"]] not in required_set:
            omittable.append(seventh)
        implied_omittable_ext = set()
        if q["ext"] == "11":
            implied_omittable_ext.add("9")
        elif q["ext"] == "13":
            implied_omittable_ext.update({"9", "11"})
        omittable.extend(ext for ext in extensions if ext in implied_omittable_ext)

    result = {
        "root": "1",
        "third": third,
        "sus": sus,
        "fifth": fifth,
        "seventh": seventh,
        "extensions": extensions,
    }
    if omittable is not None:
        result["omittable"] = omittable
    return result