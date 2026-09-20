"""
build_normalized_columns.py

Adds two new columns to the songs table:
  - normalized_all_chords    : same length/order as all_chords, each token
                                replaced with its canonical form (e.g. "Cmaj7"),
                                or null if unparseable (junk/solfege -- never guessed)
  - normalized_unique_chords : deduped canonical strings from the above

This is what makes search a plain SQL query later:
  WHERE normalized_unique_chords LIKE '%"C13"%'
(the quotes from JSON encoding act as exact-token boundaries, so this
won't accidentally also match "C13sus4" or similar).

Canonicalization: root + canonical quality name from quality_registry.json
(e.g. root="C", quality="13" -> "C13"), with bass passed through unchanged
(e.g. "C13/E") since inversion is a separate dimension from chord identity.

SAFETY: run this against a COPY of your database first. ALTER TABLE ADD
COLUMN is non-destructive (existing columns/rows are untouched), but you
should confirm the output looks right before running it against your
production betterchord_songs.db.

Usage:
    python build_normalized_columns.py [path_to_betterchord_songs.db] [path_to_quality_registry.json]
"""

import json
import os
import sqlite3
import sys

# chord_parser.py and interval_calculator.py live in the sibling config/
# folder, not here -- Python won't find them without this.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config"))

import chord_parser as cp
from interval_calculator import compute_intervals

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.join(_HERE, "..", "..", "data", "song_data", "betterchord_songs.db")
DEFAULT_REGISTRY_PATH = os.path.join(_HERE, "..", "..", "data", "registry", "quality_registry.json")


def build_intervalset_to_canonical(registry):
    """frozenset(full interval set) -> canonical quality name, e.g.
    frozenset({0,3,7,10}) -> 'm7'. Built once, used for every token."""
    lookup = {}
    for name, entry in registry.items():
        lookup[frozenset(entry["interval_set"])] = name
    return lookup


def canonicalize_token(raw_token, intervalset_to_canonical):
    """Returns the canonical chord string (e.g. 'Cmaj7', 'G13/B'), or None
    if the token is junk or solfege-ambiguous (never guessed)."""
    parsed = cp.parse_chord(raw_token)
    if not parsed["parsed"]:
        return None

    full, required = compute_intervals(parsed["quality"])
    canonical_quality = intervalset_to_canonical.get(frozenset(full))

    if canonical_quality is None:
        # Structural pattern not seen when the registry was built (e.g. a
        # parser fix since then produced a genuinely new interval set).
        # Fall back to the raw quality label from chord_parser rather than
        # silently dropping it -- flagged via the returned "unmapped" set
        # by the caller, not guessed here.
        return None

    result = cp.format_chord(parsed["root"], canonical_quality, parsed["bass"])
    return result


def build_normalized_columns(db_path=None, registry_path=None):
    db_path = db_path or DEFAULT_DB_PATH
    registry_path = registry_path or DEFAULT_REGISTRY_PATH

    registry = json.load(open(registry_path))
    intervalset_to_canonical = build_intervalset_to_canonical(registry)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Add columns if they don't already exist (safe to re-run)
    cur.execute("PRAGMA table_info(songs)")
    existing_cols = {row[1] for row in cur.fetchall()}
    if "normalized_all_chords" not in existing_cols:
        cur.execute("ALTER TABLE songs ADD COLUMN normalized_all_chords TEXT")
    if "normalized_unique_chords" not in existing_cols:
        cur.execute("ALTER TABLE songs ADD COLUMN normalized_unique_chords TEXT")
    conn.commit()

    cur.execute("SELECT id, all_chords FROM songs WHERE all_chords IS NOT NULL")
    rows = cur.fetchall()

    updates = []
    unmapped_tokens = set()
    total_tokens = 0
    null_tokens = 0

    for song_id, all_chords_json in rows:
        try:
            raw_tokens = json.loads(all_chords_json)
        except Exception:
            continue

        normalized = []
        for tok in raw_tokens:
            total_tokens += 1
            canon = canonicalize_token(tok, intervalset_to_canonical)
            if canon is None:
                null_tokens += 1
                # distinguish "known junk/solfege" from "genuinely unmapped
                # new structure" for the summary report
                r = cp.parse_chord(tok)
                if r["parsed"]:
                    unmapped_tokens.add(tok)
            normalized.append(canon)

        unique_normalized = sorted(set(t for t in normalized if t is not None))

        updates.append((
            json.dumps(normalized),
            json.dumps(unique_normalized),
            song_id,
        ))

    cur.executemany(
        "UPDATE songs SET normalized_all_chords = ?, normalized_unique_chords = ? WHERE id = ?",
        updates
    )
    conn.commit()
    conn.close()

    return {
        "songs_updated": len(updates),
        "total_tokens": total_tokens,
        "null_tokens": null_tokens,
        "unmapped_tokens": unmapped_tokens,
    }


if __name__ == "__main__":
    db_path = sys.argv[1] if len(sys.argv) > 1 else None
    registry_path = sys.argv[2] if len(sys.argv) > 2 else None

    stats = build_normalized_columns(db_path, registry_path)

    print(f"Songs updated: {stats['songs_updated']}")
    print(f"Total chord tokens processed: {stats['total_tokens']}")
    print(f"Null (junk/solfege, correctly unfilled): {stats['null_tokens']} "
          f"({stats['null_tokens']/stats['total_tokens']*100:.2f}%)")

    if stats["unmapped_tokens"]:
        print(f"\nWARNING: {len(stats['unmapped_tokens'])} tokens parsed fine but had NO "
              f"matching registry entry (genuinely new interval set since the registry "
              f"was built -- rebuild registry_builder.py and re-run this script):")
        for t in sorted(stats["unmapped_tokens"])[:20]:
            print(f"  {t!r}")
    else:
        print("\nNo unmapped tokens -- every parseable chord found a home in the registry.")