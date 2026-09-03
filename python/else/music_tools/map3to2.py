import json
from pathlib import Path

IN_PATH = Path("data/Golden/ExpertStandard.dat")
OUT_PATH = Path("data/Golden/ExpertStandard_v2.dat")


def load_json(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, p: Path):
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


if not IN_PATH.exists():
    raise FileNotFoundError(f"Input file not found: {IN_PATH}")

data = load_json(IN_PATH)

# Prepare output structure
out = {}

# Version
out["_version"] = "2.2.0"

# Transfer top-level customData if exists, renaming to _customData
if "customData" in data:
    out["_customData"] = data["customData"]
else:
    out["_customData"] = {}


# Helper mappers
def convert_note(v3note):
    # fields in v3 note: b, x, y, a, c, d, customData (optional)
    note = {}
    if "b" in v3note:
        note["_time"] = v3note["b"]
    # map coordinates
    if "x" in v3note:
        note["_lineIndex"] = v3note["x"]
    if "y" in v3note:
        note["_lineLayer"] = v3note["y"]
    if "c" in v3note:
        note["_type"] = v3note["c"]
    if "d" in v3note:
        note["_cutDirection"] = v3note["d"]
    if "a" in v3note:
        # v3 'a' usually corresponds to angle offset; keep as _angleOffset
        note["_angleOffset"] = v3note["a"]
    # carry over any customData nested in the note
    if "customData" in v3note:
        note["_customData"] = v3note["customData"]
    return note


def convert_event(v3event):
    # v3 event fields: b, et, i, f (float)
    event = {}
    if "b" in v3event:
        event["_time"] = v3event["b"]
    if "et" in v3event:
        event["_type"] = v3event["et"]
    # value (i) -> _value
    if "i" in v3event:
        event["_value"] = v3event["i"]
    # float value -> _floatValue (only include if present)
    if "f" in v3event:
        event["_floatValue"] = v3event["f"]
    # carry customData if exists
    if "customData" in v3event:
        event["_customData"] = v3event["customData"]
    return event


# Convert colorNotes -> _notes if present
if "colorNotes" in data:
    out["_notes"] = [convert_note(n) for n in data["colorNotes"]]
else:
    out["_notes"] = []

# Convert basicBeatmapEvents -> _events if present
if "basicBeatmapEvents" in data:
    out["_events"] = [convert_event(e) for e in data["basicBeatmapEvents"]]
else:
    out["_events"] = []

# Also attempt to copy other common arrays if present but rename keys to v2 style where sensible
# Examples: bombNotes -> _bombNotes, obstacles -> _obstacles, waypoints -> _waypoints
key_renames = {
    "bombNotes": "_bombNotes",
    "obstacles": "_obstacles",
    "waypoints": "_waypoints",
    "bpmEvents": "_BPMChanges",
    "rotationEvents": "_rotationEvents"  # keep name but prefix underscore
}

for k, newk in key_renames.items():
    if k in data:
        # generic shallow copy (no internal renaming) - BPM events might need different shape but keep it simple
        out[newk] = data[k]

# Save result
save_json(out, OUT_PATH)

# Provide summary and show first 8 entries for notes/events
notes_preview = out["_notes"][:8]
events_preview = out["_events"][:8]
summary = {
    "input_path": str(IN_PATH),
    "output_path": str(OUT_PATH),
    "notes_count_v3": len(data.get("colorNotes", [])),
    "notes_count_v2": len(out["_notes"]),
    "events_count_v3": len(data.get("basicBeatmapEvents", [])),
    "events_count_v2": len(out["_events"]),
    "notes_preview": notes_preview,
    "events_preview": events_preview
}

print("Conversion complete")
# print(json.dumps(summary, ensure_ascii=False, indent=2))

# # Print first 3 lines of the output file to confirm
# with OUT_PATH.open("r", encoding="utf-8") as f:
#     for i in range(20):
#         line = f.readline()
#         if not line:
#             break
#         print(line.rstrip())
