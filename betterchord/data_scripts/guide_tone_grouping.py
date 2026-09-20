"""
guide_tone_grouping.py

Step 5: group registry entries by matching REQUIRED (guide-tone) sets,
separately from the exact-full-interval-set grouping already done in
registry_builder.py.

Two qualities land in the same guide-tone group when they require the
exact same guide tones, even if their full theoretical interval sets
differ (e.g. "13" requires {root,3rd,b7,13} and so does "7add13" -- their
full sets differ by the natural 5th, but that's exactly the kind of
difference a guitarist's omission-heavy voicing choice shouldn't be
penalized for).

This does NOT change canonical naming or merge registry entries -- it
produces a separate map of "which canonical names should be shown/
searched together", used later by search (step 6) and UI display (step 7).

Usage:
    python guide_tone_grouping.py [path_to_quality_registry.json]
"""

import json
import os
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REGISTRY_PATH = os.path.join(_HERE, "..", "..", "data", "registry", "quality_registry.json")


def build_guide_tone_groups(registry):
    groups = defaultdict(list)
    for name, entry in registry.items():
        key = tuple(sorted(entry["required"]))
        groups[key].append(name)

    # Only keep groups with 2+ members -- singletons don't need merging
    multi_groups = {k: v for k, v in groups.items() if len(v) > 1}
    return multi_groups


def build_name_to_group_map(multi_groups):
    """Flat lookup: canonical name -> list of OTHER names in its group."""
    lookup = {}
    for members in multi_groups.values():
        for name in members:
            lookup[name] = [m for m in members if m != name]
    return lookup


if __name__ == "__main__":
    registry_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_REGISTRY_PATH
    registry = json.load(open(registry_path))

    multi_groups = build_guide_tone_groups(registry)
    lookup = build_name_to_group_map(multi_groups)

    output_path = os.path.join(os.path.dirname(registry_path) or ".", "guide_tone_groups.json")
    with open(output_path, "w") as f:
        json.dump({
            "groups": [sorted(v) for v in multi_groups.values()],
            "name_to_related": lookup,
        }, f, indent=2)

    print(f"Found {len(multi_groups)} guide-tone groups "
          f"(qualities with different full interval sets, but identical "
          f"required guide tones):\n")

    for required_key, members in sorted(multi_groups.items(), key=lambda x: -sum(
            registry[m]["songs_db_occurrences"] for m in x[1])):
        total = sum(registry[m]["songs_db_occurrences"] for m in members)
        print(f"  required={list(required_key)}  (total songs_db occurrences: {total})")
        for m in members:
            e = registry[m]
            print(f"      {m:20s} full={e['interval_set']}  "
                  f"songs_db_occurrences={e['songs_db_occurrences']}  "
                  f"has_voicing={e['has_voicing_entry']}")
        print()

    if not multi_groups:
        print("  (none found)")

    print(f"\nWrote guide_tone_groups.json -- 'name_to_related' is the lookup"
          f" search/UI code should use: given a canonical name, get every"
          f" other name that should be shown alongside it.")