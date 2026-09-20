# Runtime voicing lookup module. Called at inference time to get guitar voicings for a identified chord
# Returns a list of voicing dicts sorted by priority (Must Know first), excluding inversion-category voicings.
# If model gets slash chord, and no voicings are foundm, falls back to base chord voicings

import os
import json
import sqlite3
import re

import chord_parser as cp
from interval_calculator import compute_intervals
from chord_info import spell_note, LETTER_TO_SEMITONE

_HERE = os.path.dirname(os.path.abspath(__file__))

# Path to voicing db
DB_PATH = os.path.join(_HERE, '..', '..', 'data', 'voicing_data', 'voicings.db')

# Path to the quality registry -- built by registry_builder.py, sits
# alongside this file in betterchord/
QUALITY_REGISTRY_PATH = os.path.join(_HERE, '..', '..', 'data', 'registry', 'quality_registry.json')

# Priority sorting for voicing type
TYPE_PRIORITY = {
    "Must Know": 0,
    "Other":      1,
    "Capo":     2,
}


def _get_connection():
    # Get a SQLite connection. Called fresh each time on runtime
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # allows dict-like access by column name
    return conn


# ---------------------------------------------------------------------------
# Registry-based quality translation
#
# voicings.db stores each chord's quality under whatever string the
# scraper/tagger used (e.g. "7#5"), often NOT the string
# music_theory.py's identify_chord() outputs for the same chord (e.g.
# "aug7") -- same interval content, different name. A literal string match
# on the chord_name handed in would silently return zero rows whenever the
# two conventions disagree (e.g. 28 real aug7-family rows filed under
# "7#5", invisible to an "Aaug7" lookup).
#
# So: translate the query through quality_registry.json's interval-set
# matching -- the same mechanism registry_builder.py uses to reconcile
# music_theory.py names against voicings.db names -- so "Aaug7" resolves
# to "A7#5" before querying.
# ---------------------------------------------------------------------------

_registry_cache = None


def _load_registry():
    global _registry_cache
    if _registry_cache is None:
        with open(QUALITY_REGISTRY_PATH) as f:
            registry = json.load(f)

        intervalset_to_canonical = {
            frozenset(entry["interval_set"]): name for name, entry in registry.items()
        }
        # canonical quality name -> the actual string voicings.db uses for
        # it (only present for qualities that DO have a voicing -- the 34
        # known real gaps correctly have nothing here, not a mistranslation)
        canonical_to_voicing_quality = {
            name: entry["voicing_strings_found"][0]
            for name, entry in registry.items()
            if entry.get("has_voicing_entry") and entry.get("voicing_strings_found")
        }
        _registry_cache = (intervalset_to_canonical, canonical_to_voicing_quality)
    return _registry_cache


def resolve_voicing_chord(chord_name):
    """
    Translate ANY chord string (however it happens to be spelled) into the
    (root, voicing_quality, bass) that actually matches voicings.db's own
    naming convention, using the shared interval-set registry rather than
    a literal string comparison.

    Returns a dict {root, voicing_quality, bass, canonical_quality,
    translated} on success -- 'translated' is True when voicings.db's
    string differs from music_theory.py's canonical name for this chord
    (i.e. this is exactly the case the old code silently failed on).

    Returns None if the chord can't be parsed at all, OR if it parses fine
    but has no matching voicing under ANY known name (a real gap -- see
    registry_builder.py's gap report -- not a lookup bug this can fix).
    """
    parsed = cp.parse_chord(chord_name)
    if not parsed["parsed"]:
        return None

    full, required = compute_intervals(parsed["quality"])
    intervalset_to_canonical, canonical_to_voicing_quality = _load_registry()

    canonical = intervalset_to_canonical.get(frozenset(full))
    if canonical is None:
        # Structural pattern the registry doesn't know at all (parser and
        # registry have drifted since the registry was last built) --
        # nothing to translate against.
        return None

    voicing_quality = canonical_to_voicing_quality.get(canonical)
    if voicing_quality is None:
        # No voicing exists for this chord under ANY name -- a real,
        # already-known gap, not something a smarter lookup can fix.
        return None

    return {
        "root": parsed["root"],
        "voicing_quality": voicing_quality,
        "bass": parsed["bass"],
        "canonical_quality": canonical,
        "translated": voicing_quality != canonical,
    }


def _query_voicings(conn, chord, bass=None):
    # Query voicings for output chord

    # If bass is none (null), query base chord voicings (null bass imples its not a slash chord)
    # If bass is provided, query slash chord voicings for that bass note (filled bass implies its a slash)

    # NOTE (public repo only): `rank` is deliberately not selected/ordered
    # here -- the public release DB doesn't carry that column, and
    # dropping it changes nothing observable: _sort_by_type() below
    # already re-sorts every result by (type_priority, base_fret)
    # regardless of what order these queries return rows in. The
    # production repo's own voicings.py is unaffected by this -- it still
    # serves the full production DB with `rank` intact.
    if bass is None:
        rows = conn.execute("""
            SELECT frets, base_fret, barres, notes, intervals,
                   type, category, capo, source
            FROM voicings
            WHERE chord = ?
              AND bass IS NULL
              AND type IN ('Must Know', 'Other', 'Capo')
        """, (chord,)).fetchall()
    else:
        rows = conn.execute("""
            SELECT frets, base_fret, barres, notes, intervals,
                   type, category, capo, source
            FROM voicings
            WHERE chord = ?
              AND bass = ?
              AND type IN ('Must Know', 'Other', 'Capo')
        """, (chord, bass)).fetchall()
    return rows


def _raw_note_semitone(raw):
    """Semitone (0-11) for ANY note spelling, including double
    accidentals `cp.canon_note()`/NOTE_ALIASES doesn't cover (e.g.
    'Bbb', 'E##') -- built on `chord_info.py`'s own LETTER_TO_SEMITONE
    table (reused, not a new hand-rolled letter-pitch mapping -- a
    note's pitch class doesn't depend on which of several enharmonic
    spellings the raw scrape happened to use) plus a plain sharp/flat
    count. Returns None if `raw` isn't a recognizable note token at all
    (never raises -- callers already have an "unrecognized, leave as
    is" fallback for that case).
    """
    if not raw:
        return None
    base = LETTER_TO_SEMITONE.get(raw[0].upper())
    if base is None:
        return None
    accidentals = raw[1:]
    if accidentals and not all(c in "#b" for c in accidentals):
        return None
    return (base + accidentals.count("#") - accidentals.count("b")) % 12


def _canon_notes(raw_notes, root, quality=""):
    # voicings.db's `notes` column was scraped independently of
    # chord_parser.py's canonical-spelling table and can disagree with it
    # (e.g. every C#/F#-root voicing is scraped in the flat spelling
    # Db/Gb, even though get_voicings()'s `displayed` field says "C#"/
    # "F#"). Respell every note key-aware via chord_info.spell_note()
    # (reused rather than building a second, independent spelling system),
    # so a sharp-key chord's tones read "G#" not "Ab".
    #
    # The root note (semitone 0) is unchanged -- spell_note() returns
    # root_name verbatim for interval 0 -- so a voicing's root always
    # matches the page header. Notes are respelled relative to the root:
    # spell_note() needs the offset FROM the root, not an absolute pitch
    # class.
    #
    # canon_note() only resolves the 12 single-accidental pitch classes.
    # A double-accidental raw note (dim7/m7b5 data's legitimate "Ebb"/
    # "Bbb" etc.) has its semitone computed by letter + accidental-count
    # arithmetic (`_raw_note_semitone` -- a pitch doesn't depend on which
    # enharmonic spelling scraped it) so it routes through the same
    # spell_note() call. A genuinely unrecognizable token falls through
    # unchanged, never crashes.
    root_idx = cp.CHROMATIC.index(root) if root in cp.CHROMATIC else None
    out = []
    for n in raw_notes:
        canon = cp.canon_note(n)
        if canon is not None and root_idx is not None:
            semitone = (cp.CHROMATIC.index(canon) - root_idx) % 12
            out.append(spell_note(root, semitone, quality))
        elif canon is not None:
            # root itself didn't resolve (rare) -- use the fixed canonical
            # spelling.
            out.append(canon)
        elif root_idx is not None:
            abs_semitone = _raw_note_semitone(n)
            if abs_semitone is None:
                out.append(n)
            else:
                # Relative to the root, same as the canon_note() branch
                # above -- an absolute pitch class alone (e.g. 9 for
                # "Bbb") isn't what spell_note() wants; it needs the
                # offset FROM the root (e.g. 3, for F#'s minor third).
                semitone = (abs_semitone - root_idx) % 12
                out.append(spell_note(root, semitone, quality))
        else:
            out.append(n)
    return out


def _rows_to_dicts(rows, root=None, quality=""):
    # Convert sqlite3.Row objects to plain dicts with parsed JSON fields.
    # `root`/`quality` (the resolved chord's canonical root + registry
    # quality name) drive the key-aware note respelling in _canon_notes --
    # optional so a caller with no resolved chord still returns something
    # sane rather than crashing on a missing argument.
    result = []
    for row in rows:
        result.append({
            "frets":     row["frets"],
            "base_fret": row["base_fret"],
            "barres":    json.loads(row["barres"]) if row["barres"] else [],
            "notes":     _canon_notes(json.loads(row["notes"]), root, quality) if row["notes"] else [],
            "intervals": json.loads(row["intervals"]) if row["intervals"] else [],
            "type":      row["type"],
            "category":  row["category"],
            "capo":      row["capo"] or 0,
            "source":    row["source"],
        })
    return result


def _sort_by_type(voicings):
    # Sort voicings by type priority (Must Know, Other, Capo), then by
    # base_fret ascending within each type group -- was `rank` before, but
    # frontend now renders every voicing as its own diagram grouped by
    # type/section, so ordering low-to-high on the fretboard within a
    # section reads better than the old rank order.
    return sorted(voicings, key=lambda v: (TYPE_PRIORITY.get(v["type"], 99), v["base_fret"]))


def _get_voicings_literal(conn, chord_name):
    """
    Literal-string-match lookup -- a fallback for cases the registry can't
    resolve at all (unparseable input, or a chord whose interval set the
    registry has never seen), so the registry path can only ever find
    MORE voicings, never fewer.
    """
    slash_match = re.search(r'/([A-G][b#]?)$', chord_name)

    fallback  = False
    displayed = chord_name

    if slash_match:
        bass_note  = slash_match.group(1)
        base_chord = chord_name[:slash_match.start()]
        rows = _query_voicings(conn, chord_name, bass=bass_note)

        if not rows:
            rows = _query_voicings(conn, base_chord, bass=None)
            fallback  = True
            displayed = base_chord
    else:
        rows = _query_voicings(conn, chord_name, bass=None)

    # Parsed fresh from whatever was queried (no `resolved` dict on this
    # registry-less path) -- feeds the same key-aware note respelling in
    # _canon_notes. Quality is left "" rather than re-derived: parse_chord
    # returns quality as a structured dict, not the string name
    # spell_note() wants ("m7b5"), and quality only affects spell_note()'s
    # disambiguation of a few altered qualities (b5-vs-#11, #5-vs-b13) --
    # "" just falls back to the fixed spelling for those. A root that
    # fails to parse falls to _canon_notes's None-root handling (notes
    # left as scraped).
    parsed_for_display = cp.parse_chord(displayed if slash_match and fallback else chord_name)
    literal_root = parsed_for_display["root"] if parsed_for_display["parsed"] else None

    voicings = _rows_to_dicts(rows, root=literal_root, quality="")
    sorted_voicings = _sort_by_type(voicings)

    # `bass`: the resolved slash-bass note backing the returned rows,
    # canonicalized through cp.canon_note -- None when the rows are
    # root-position (no slash, or a fallback stripped the bass). A
    # top-level fact about the whole response, not per-voicing (every row
    # came from one resolved query). Not re-derived frontend-side by
    # splitting on "/" -- a compound quality like "6/9" (A6/9) contains a
    # "/" that isn't a bass separator -- only chord_parser.py's real
    # parser handles this correctly.
    bass_out = cp.canon_note(bass_note) if (slash_match and not fallback) else None

    return {
        "voicings":        sorted_voicings,
        "fallback":        fallback,
        "displayed":       displayed,
        "translated":      False,
        "translated_from": None,
        "bass":            bass_out,
    }


def get_voicings(chord_name):
    # Get all voicings for a chord, sorted Must Know, Other, Capo
    # Excludes inversion-category voicings.

    # If chord_name is a slash chord (e.g. Cmaj7/E) and no voicings are found, falls back to the base chord (Cmaj7).

    conn = _get_connection()

    try:
        resolved = resolve_voicing_chord(chord_name)

        if resolved is None:
            # Couldn't parse, or the registry has no entry for this exact
            # interval set -- fall back to the literal string match.
            return _get_voicings_literal(conn, chord_name)

        root              = resolved["root"]
        bass              = resolved["bass"]
        voicing_quality   = resolved["voicing_quality"]
        canonical_quality = resolved["canonical_quality"]  # e.g. "m7b5" -- feeds spell_note()'s
                                                             # b5-vs-#11 / #5-vs-b13 disambiguation
                                                             # in _canon_notes; harmless if it doesn't
                                                             # match spell_note()'s quality vocabulary
                                                             # exactly (only affects a few altered
                                                             # qualities, falls back to the fixed
                                                             # spelling).
        translated_chord = root + voicing_quality  # e.g. "A7#5", not "Aaug7"

        fallback = False
        if bass:
            full_query = f"{translated_chord}/{bass}"
            rows = _query_voicings(conn, full_query, bass=bass)
            displayed = full_query
            if not rows:
                rows = _query_voicings(conn, translated_chord, bass=None)
                fallback  = True
                displayed = translated_chord
        else:
            rows = _query_voicings(conn, translated_chord, bass=None)
            displayed = translated_chord

        # Safety net: the registry said this should have a voicing, but if
        # the lookup still comes up empty for an edge case (e.g. a
        # slash-bass spelling voicings.db doesn't have), try the raw
        # literal input before giving up.
        if not rows:
            literal = _get_voicings_literal(conn, chord_name)
            if literal["voicings"]:
                return literal

        voicings = _rows_to_dicts(rows, root=root, quality=canonical_quality)
        sorted_voicings = _sort_by_type(voicings)

        # `bass` -- see _get_voicings_literal's identical field for the
        # full reasoning (top-level, not per-voicing; None whenever a
        # fallback already stripped the bass off). `resolved["bass"]` is
        # already canon_note-normalized (resolve_voicing_chord parses it
        # via cp.parse_chord, whose own "bass" field is canon_note'd --
        # see chord_parser.py:449), so no second normalization is needed
        # here.
        return {
            "voicings":        sorted_voicings,
            "fallback":        fallback,
            "displayed":       displayed,
            "translated":      resolved["translated"],
            "translated_from": chord_name if resolved["translated"] else None,
            "bass":            bass if not fallback else None,
        }
    finally:
        conn.close()


def get_voicings_by_type(chord_name, vtype="Must Know"):
    # Get voicings filtered to a specific type only

    result = get_voicings(chord_name)
    return {
        "voicings":  [v for v in result["voicings"] if v["type"] == vtype],
        "fallback":  result["fallback"],
        "displayed": result["displayed"],
    }


def get_must_know(chord_name):
    # returns Must Know voicings only
    return get_voicings_by_type(chord_name, "Must Know")


if __name__ == "__main__":
    # Quick diagnostic - test a few chords, including ones that exercise
    # the new translation path (Aaug7, C7b13, etc.)
    test_chords = ["Cmaj7", "Gm7", "F#m7b5", "Cmaj7/E", "Am7", "G", "D9",
                   "C6/9", "D6/9/E", "Aaug7", "Caug7", "C7b13"]

    for chord in test_chords:
        result    = get_voicings(chord)
        voicings  = result["voicings"]
        must_know = [v for v in voicings if v["type"] == "Must Know"]
        fallback  = result["fallback"]
        displayed = result["displayed"]
        translated = result.get("translated")
        print(f"\n{chord}: {len(voicings)} total, {len(must_know)} must know  "
              f"fallback={fallback}  displayed='{displayed}'  translated={translated}")
        for v in voicings[:3]:
            print(f"  [{v['type']:<10}] {v['frets']:<22} capo={v['capo']} notes={v['notes']}")