"""
registry_builder.py

Reconciles the three drifted sources of chord-quality truth in BetterChord:

  1. music_theory.py  - QUALITY_INTERVALS (66 entries, hand-tuned required/notes)
  2. voicings.db      - distinct `quality` strings actually stored (59 entries)
  3. betterchord_songs.db - every quality that REAL songs actually use,
     via chord_parser.py's structural parse (2,592 distinct patterns)

Everything gets run through ONE calculator (interval_calculator.py) so
comparisons are apples-to-apples instead of trusting whichever source
happens to have an opinion. Output:

  - quality_registry.json  : canonical entries, grouped by interval-set
                              equivalence, with every known alias attached
  - printed gap report      : what's missing where, and where the three
                              sources actively disagree
"""

import json
import os
import re
import sqlite3
from collections import defaultdict, Counter

# chord_parser.py and interval_calculator.py live in the sibling config/
# folder, not here -- Python won't find them without this.
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config"))

import chord_parser as cp
from interval_calculator import compute_intervals

# Matches your actual BetterChord layout:
#   betterchord/registry_builder.py   (this file, and music_theory.py lives here too)
#   data/voicing_data/voicings.db
#   data/song_data/betterchord_songs.db
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MUSIC_THEORY_PATH = os.path.join(_HERE, "..", "config", "music_theory.py")
DEFAULT_VOICINGS_DB_PATH = os.path.join(_HERE, "..", "..", "data", "voicing_data", "voicings.db")
DEFAULT_SONGS_DB_PATH = os.path.join(_HERE, "..", "..", "data", "song_data", "betterchord_songs.db")


# ---------------------------------------------------------------------------
# Source 1: music_theory.py's own QUALITY_INTERVALS (loaded, not re-typed)
# ---------------------------------------------------------------------------

def load_music_theory_registry(path=None):
    path = path or DEFAULT_MUSIC_THEORY_PATH
    text = open(path, encoding="utf-8-sig").read()
    ns = {}
    exec(text.split("if __name__")[0], ns)
    return ns["QUALITY_INTERVALS"], ns.get("ALIASES", {})


# ---------------------------------------------------------------------------
# Source 2: voicings.db distinct quality strings
# ---------------------------------------------------------------------------

def load_voicings_qualities(path=None):
    path = path or DEFAULT_VOICINGS_DB_PATH
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute("SELECT quality, COUNT(*) FROM voicings GROUP BY quality")
    return {q: c for q, c in cur.fetchall()}


def voicings_by_intervalset(voicing_qualities):
    """Run every voicings.db quality string through the SAME shared engine
    used for music_theory.py and songs.db, so coverage is checked by actual
    musical content instead of by whether the string happens to match."""
    by_intervalset = defaultdict(list)
    unparseable = []
    for qkey, count in voicing_qualities.items():
        probe = cp.format_chord("C", qkey)
        parsed = cp.parse_chord(probe)
        if not parsed["parsed"]:
            unparseable.append((qkey, count))
            continue
        full, required = compute_intervals(parsed["quality"])
        by_intervalset[frozenset(full)].append((qkey, count))
    return by_intervalset, unparseable


# ---------------------------------------------------------------------------
# Source 3: songs.db, run through chord_parser -> structural signatures
# ---------------------------------------------------------------------------

def signature_key(quality_dict):
    """A hashable, order-independent key for a parsed quality structure."""
    return (
        quality_dict["base"],
        quality_dict["seventh"],
        quality_dict["ext"],
        frozenset(quality_dict["alterations"]),
        frozenset(quality_dict["adds"]),
        quality_dict["six"],
        quality_dict["sixnine"],
        quality_dict.get("no3", False),
        quality_dict.get("no5", False),
    )


def load_songs_db_signatures(path=None):
    path = path or DEFAULT_SONGS_DB_PATH
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute("SELECT unique_chords FROM songs WHERE unique_chords IS NOT NULL")

    sig_counts = Counter()
    sig_examples = {}
    for (uc,) in cur.fetchall():
        try:
            chords = json.loads(uc)
        except Exception:
            continue
        for tok in chords:
            r = cp.parse_chord(tok)
            if not r["parsed"]:
                continue
            sig = signature_key(r["quality"])
            sig_counts[sig] += 1
            sig_examples.setdefault(sig, tok)
    return sig_counts, sig_examples


# ---------------------------------------------------------------------------
# Render a display name from a structural signature (for entries with no
# existing name in music_theory.py -- best-effort, flagged as such)
# ---------------------------------------------------------------------------

def render_name(sig):
    base, seventh, ext, alterations, adds, six, sixnine, no3, no5 = sig

    BASE_LABEL = {"maj": "", "min": "m", "dim": "dim", "aug": "aug",
                  "sus2": "sus2", "sus4": "sus4", "5": "5"}
    label = BASE_LABEL.get(base, base)

    if sixnine:
        label += "6/9"
    elif six:
        label += "6"

    if base == "dim" and seventh == "dim7":
        label += "7"
    elif seventh == "maj7":
        label += "maj7"
    elif seventh == "7":
        label += "7"
    elif seventh == "dim7":
        label += "dim7"

    if ext:
        # replace the trailing 7-ish digit with the higher extension number
        # (maj7 + ext9 -> maj9, 7 + ext9 -> 9, m7 + ext11 -> m11)
        label = re.sub(r"7$", "", label) if label.endswith("7") else label
        label = re.sub(r"maj$", "maj", label)
        label += ext

    for a in sorted(adds):
        label += a
    for a in sorted(alterations):
        label += a
    if no3:
        label += "no3"
    if no5:
        label += "no5"

    return label if label else "maj"


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def build_registry():
    mt_qualities, mt_aliases = load_music_theory_registry()
    voicing_qualities = load_voicings_qualities()
    songs_sigs, songs_examples = load_songs_db_signatures()

    # Compute interval sets for every music_theory.py entry by RE-PARSING its
    # own key through chord_parser, so it's judged by the same calculator as
    # everything else (apples to apples) rather than trusting its "notes" field.
    mt_computed = {}       # quality_key -> (full, required) computed by OUR engine
    mt_declared = {}       # quality_key -> (notes, required) as music_theory.py wrote it
    mt_disagreements = []
    for qkey, data in mt_qualities.items():
        probe = cp.format_chord("C", qkey)
        parsed = cp.parse_chord(probe)
        if not parsed["parsed"]:
            mt_computed[qkey] = None
            continue
        full, required = compute_intervals(parsed["quality"])
        mt_computed[qkey] = (full, required)
        mt_declared[qkey] = (sorted(data["notes"]), sorted(data["required"]))
        if full != sorted(data["notes"]):
            mt_disagreements.append((qkey, sorted(data["notes"]), full, sorted(data["required"]), required))

    # Group songs.db signatures by their computed full-interval-set
    groups = defaultdict(list)   # frozenset(full) -> list of (sig, count, example)
    for sig, count in songs_sigs.items():
        base, seventh, ext, alterations, adds, six, sixnine, no3, no5 = sig
        qdict = {"base": base, "seventh": seventh, "ext": ext,
                 "alterations": set(alterations), "adds": set(adds),
                 "six": six, "sixnine": sixnine, "no3": no3, "no5": no5}
        full, required = compute_intervals(qdict)
        groups[frozenset(full)].append((sig, count, songs_examples[sig], tuple(required)))

    # For each interval-set group, find the best existing name (prefer
    # music_theory.py's, since that's what the CNN already outputs in prod)
    mt_by_intervalset = defaultdict(list)
    for qkey, computed in mt_computed.items():
        if computed:
            mt_by_intervalset[frozenset(computed[0])].append(qkey)

    voicing_strings = set(voicing_qualities.keys())
    voicing_by_intervalset, voicing_unparseable = voicings_by_intervalset(voicing_qualities)

    # Walk the UNION of all three sources' interval sets, not just songs.db's
    # -- a quality with real voicings.db and/or music_theory.py coverage but
    # ZERO current songs.db occurrences (e.g. its only real song got deleted)
    # must still get a registry entry, not silently vanish. Confirmed this
    # was a real bug: sus2add#11 had 438 real voicings.db rows and a
    # music_theory.py name, but disappeared entirely from a rerun after its
    # only songs.db source song was removed, because the old loop only ever
    # visited interval sets that came from `groups` (songs.db-derived).
    all_interval_sets = set(groups.keys()) | set(mt_by_intervalset.keys()) | set(voicing_by_intervalset.keys())

    registry = {}
    for interval_set in all_interval_sets:
        entries = groups.get(interval_set, [])
        mt_names = mt_by_intervalset.get(interval_set, [])
        voicing_matches = voicing_by_intervalset.get(interval_set, [])

        if entries:
            entries.sort(key=lambda e: -e[1])
            best_sig, best_count, best_example, required = entries[0]
            total_occurrences = sum(e[1] for e in entries)
            example_spellings = [e[2] for e in entries[:5]]
        else:
            # No songs.db occurrence at all -- fall back to whatever
            # required-tone info the other two sources can provide, rather
            # than needing songs.db data to exist at all.
            total_occurrences = 0
            example_spellings = []
            if mt_names:
                required = tuple(sorted(mt_qualities[mt_names[0]]["required"]))
            elif voicing_matches:
                # derive required by re-parsing one of the actual voicings.db
                # quality strings through the same shared engine
                probe = cp.parse_chord("C" + voicing_matches[0][0])
                _, required = compute_intervals(probe["quality"])
                required = tuple(sorted(required))
            else:
                required = tuple(sorted(interval_set))  # shouldn't happen, safe fallback

        if mt_names:
            canonical = mt_names[0]
        elif entries:
            canonical = render_name(entries[0][0])
        elif voicing_matches:
            canonical = voicing_matches[0][0]
        else:
            canonical = "unknown_" + "_".join(str(i) for i in sorted(interval_set))  # should never trigger

        # Coverage check by ACTUAL INTERVAL SET, not string match -- this is
        # what catches "aug7" (music_theory's name) and "7#5" (voicings.db's
        # name for the identical chord) as the same coverage, instead of
        # reporting a false gap.
        voicing_hit = len(voicing_matches) > 0
        voicing_strings_found = sorted(set(v[0] for v in voicing_matches))

        registry[canonical] = {
            "interval_set": sorted(interval_set),
            "required": list(required),
            "omittable": sorted(set(interval_set) - set(required)),
            "songs_db_occurrences": total_occurrences,
            "songs_db_example_spellings": example_spellings,
            "music_theory_names": mt_names,
            "has_voicing_entry": voicing_hit,
            "voicing_strings_found": voicing_strings_found,
        }

    return registry, mt_disagreements, mt_qualities, voicing_qualities, songs_sigs


if __name__ == "__main__":
    registry, disagreements, mt_qualities, voicing_qualities, songs_sigs = build_registry()

    with open("quality_registry.json", "w") as f:
        json.dump(registry, f, indent=2)

    print(f"Built registry: {len(registry)} canonical quality groups "
          f"(from {len(songs_sigs)} distinct structural patterns actually used in songs.db)\n")

    print("=" * 70)
    print("FIXED BY INTERVAL MATCHING: qualities that LOOKED like gaps under a")
    print("string comparison, but actually have voicings under a different name")
    print("=" * 70)
    renamed = [(name, e) for name, e in registry.items()
               if e["has_voicing_entry"] and name not in e["voicing_strings_found"]]
    renamed.sort(key=lambda x: -x[1]["songs_db_occurrences"])
    for name, e in renamed:
        print(f"  {e['songs_db_occurrences']:6d}  {name:15s} -> voicings.db has it as {e['voicing_strings_found']}")
    if not renamed:
        print("  (none)")

    print()
    print("=" * 70)
    print("REAL GAP: qualities real songs use with NO voicing under ANY name")
    print("(checked by interval set, not string -- these are genuine gaps)")
    print("=" * 70)
    gap = [(name, e) for name, e in registry.items() if not e["has_voicing_entry"]]
    gap.sort(key=lambda x: -x[1]["songs_db_occurrences"])
    for name, e in gap[:30]:
        mt_tag = f"(music_theory calls it: {e['music_theory_names']})" if e["music_theory_names"] else "(NEW - not in music_theory.py either)"
        print(f"  {e['songs_db_occurrences']:6d}  {name:20s} {mt_tag}")
    print(f"  ... {len(gap)} total gaps")

    print()
    print("=" * 70)
    print("DISAGREEMENT: music_theory.py's declared interval set vs. what")
    print("the SAME quality string computes to when run through the shared engine")
    print("=" * 70)
    for qkey, declared_notes, computed_full, declared_req, computed_req in disagreements:
        print(f"  quality={qkey!r}")
        print(f"    music_theory.py declares notes    = {declared_notes}")
        print(f"    shared engine computes full        = {computed_full}")
        print(f"    music_theory.py declares required = {declared_req}")
        print(f"    shared engine computes required    = {computed_req}")
        print()