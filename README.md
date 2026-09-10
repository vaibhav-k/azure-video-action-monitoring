# Video Activity Analyzer (proof of concept)

A small CLI that uploads pre-recorded videos to Azure AI Video Indexer,
waits for processing, and turns the resulting insights into three
artifacts per video: the raw insights JSON, a normalized activities/people
JSON, and a plain-text report suitable for a client demo.

**Read "What this prototype can reliably demonstrate" and "Known
limitations" below before showing this to anyone.** The honest, most
important fact about this whole prototype is: Azure AI Video Indexer has
no dedicated human-action-recognition model, and it never tells you which
tracked person did which thing. Everything this tool presents as an
"activity" or a "person doing an activity" is built, transparently, on top
of two things Video Indexer *does* actually give you (visual labels, and
tracked-but-unidentified people), plus this script's own inference from
overlapping timestamps. Nothing here is fabricated, but nothing here is a
verbatim Video Indexer fact either -- see below for exactly where the line
is.

## Setup

1. **Azure AI Video Indexer resource.** You need an existing Video Indexer
   account (ARM-based, not the legacy trial/API-key kind) in the Azure
   Portal, and an Azure identity with at least `Contributor` on it.

2. **Python 3.11+**, then:

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # .venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

3. **Configure.**

   ```bash
   cp .env.example .env
   ```

   Fill in `AVI_SUBSCRIPTION_ID`, `AVI_RESOURCE_GROUP`, `AVI_ACCOUNT_NAME`,
   `AVI_ACCOUNT_ID`, and `AVI_LOCATION` from your Video Indexer resource's
   page in the Azure Portal (`.env.example` says exactly where to find
   each one). This script authenticates as an Azure identity, not with a
   Video Indexer API key, so also make sure one of:

   * you're logged in locally (`az login`), or
   * `AZURE_CLIENT_ID` / `AZURE_TENANT_ID` / `AZURE_CLIENT_SECRET` are set
     for a service principal

   actually has `Contributor` on the account.

## Usage

Batch mode -- processes every supported video already in `input/`:

```bash
python analyze_video.py
```

Single-video mode -- the path can be inside `input/` or anywhere else:

```bash
python analyze_video.py --video "/path/to/video.mp4"
```

Force reprocessing (by default, a video is skipped if its three output
files already exist):

```bash
python analyze_video.py --force
python analyze_video.py --video "/path/to/video.mp4" --force
```

By-ID mode -- analyze a video that's already been uploaded and indexed
(by this tool or otherwise), using its Video Indexer video ID instead of
a local file. No upload happens; this just waits out any remaining
processing and fetches the finished insights:

```bash
python analyze_video.py --video-id "1l6cptolbd"
```

The output filename base is the video's own registered name in Video
Indexer when it has one, otherwise the video ID itself. `--video-id` is
single-video mode only -- it can't be combined with `--video`, and
`--force` works the same way as above.

`input/` and `output/` are created automatically if missing. Supported
extensions: `.mp4 .mov .avi .mkv .wmv .m4v .mpg .mpeg`.

## Console output

While processing, this script also prints a one-paragraph plain-English
summary and which activities it recognized and which person (if any)
each was associated with, right after that video finishes -- the same
text and lines that get written into its `.report.txt`, so what you see
in the terminal and what's saved to disk always match:

```
Processing customer_demo.mp4 ...
  uploaded, video id=1l6cptolbd
  video state: Processing
  video state: Processed
  summary: This 47-second clip tracks three people. Video Indexer's labels flag two activities here: running and walking. Of those, one lines up with a specific tracked person by timestamp overlap; the other one doesn't, because either no one or more than one person was in frame at that point. Worth keeping in mind: these are Video Indexer's own visual labels, not output from a dedicated action-recognition model, and the person-to-activity links above are this script's own inference from overlapping timestamps -- never something Video Indexer states outright. See NOTES below for the full picture.
  activities recognized:
    00:04 - 00:12   walking   confidence: 0.91
    00:18 - 00:27   Person 2: running   confidence: 0.87
  done: 3 people, 2 activities -> customer_demo.insights.json / customer_demo.activities.json / customer_demo.report.txt
```

## Output

For `input/customer_demo.mp4`, this produces:

```
output/
├── customer_demo.insights.json      # the complete raw Video Indexer response
├── customer_demo.activities.json    # normalized people/activities/objects/labels
└── customer_demo.report.txt         # human-readable summary
```

A video supplied from outside `input/` still uses its filename as the
output base name. The original video is never copied into `output/`.

### Example report

This is real output from this script (`render_report`), not a mockup --
generated from a synthetic insights payload with three labels
("walking", "running", "talking") and three observed people with
overlapping appearance windows:

```
VIDEO ACTIVITY REPORT
======================

Video: customer_demo.mp4
Duration: 00:47

People detected: 3
Activities detected: 3

SUMMARY
-------
This 47-second clip tracks three people. Video Indexer's labels flag three activities here: running, talking, and walking. Of those, two line up with a specific tracked person by timestamp overlap; the other one doesn't, because either no one or more than one person was in frame at that point. Worth keeping in mind: these are Video Indexer's own visual labels, not output from a dedicated action-recognition model, and the person-to-activity links above are this script's own inference from overlapping timestamps -- never something Video Indexer states outright. See NOTES below for the full picture.

PEOPLE
------
  Person 1     seen 00:00 - 00:20 (tracked only, not recognized)
  Person 2     seen 00:15 - 00:30 (tracked only, not recognized)
  Person 3     seen 00:28 - 00:47 (tracked only, not recognized)

00:04 - 00:12   Person 1: walking   confidence: 0.91
00:18 - 00:27   running   confidence: 0.87
00:32 - 00:47   Person 3: talking   confidence: 0.79

ACTIVITY SUMMARY
----------------
running: 1 occurrence
talking: 1 occurrence
walking: 1 occurrence

NOTES
-----
- "Activities" are Video Indexer's own visual labels, filtered to those
  that read as actions (e.g. "running", "-ing"-ending tags) -- not output
  from a dedicated human-action-recognition model. Video Indexer's current
  API has no such model; only actions its general label vocabulary
  happens to name can appear here.
- Person <-> activity pairings shown above are this script's own
  inference from overlapping timestamps, made only when exactly one
  observed person's appearance overlaps the activity -- never a
  relationship Video Indexer itself provides. See each activity's
  "association" field in the .activities.json file for the exact
  reasoning, including cases left unassigned.
- Video Indexer's current API does not document a bounding-box/spatial
  field for labels, objects, or observed people -- only start/end
  timestamps. Bounding boxes are therefore always reported as unavailable
  (null) rather than invented.
```

Notice "running" is reported *without* a person attached, even though two
people (1 and 2) were both in frame during that window. That's not a bug
-- it's the whole point. Person 1's window (00:00-00:20) and Person 2's
window (00:15-00:30) both overlap "running" (00:18-00:27), so this script
has no basis to say which of them the label actually describes, and
reports the activity unassigned rather than guessing. "walking" and
"talking" each overlap exactly one tracked person's window, so those get
attributed. This is the honest behavior the whole design leans on: an
association is only ever made when the timing evidence is actually
unambiguous, per the person-association section of `.activities.json`.

### `customer_demo.activities.json` shape

```json
{
  "video": "customer_demo.mp4",
  "duration_seconds": 47.0,
  "people": [
    {
      "id": 2,
      "label": "Person 2",
      "matched_face_confidence": null,
      "appearances": [
        {"start_seconds": 15.0, "end_seconds": 30.0, "start": "00:15", "end": "00:30",
         "confidence": null, "bounding_box": null}
      ]
    }
  ],
  "activities": [
    {
      "name": "running",
      "start_seconds": 18.0, "end_seconds": 27.0, "start": "00:18", "end": "00:27",
      "confidence": 0.87, "bounding_box": null,
      "person_id": null, "person_label": null,
      "association": {
        "basis": null,
        "note": "2 people (ids: 1, 2) overlapped this time range -- ambiguous, not assigned",
        "provided_by_video_indexer": false
      },
      "source": "labels"
    }
  ],
  "objects": [],
  "labels": [ ]
}
```

`matched_face_confidence` and `label` are populated with a real matched
name only when Video Indexer's own `matchingFace` link says so (see
"Known limitations" -- this needs Face Recognition access approval on
your account); otherwise a person is just "Person N".

## What this prototype can reliably demonstrate to a potential client

* **End-to-end pipeline against the real, current Azure AI Video Indexer
  API** -- upload, poll to completion, retrieve the detailed (non-
  summarized) insights, all using the currently-supported ARM-based auth
  flow (no legacy API key).
* **People detection and tracking**: Video Indexer's `observedPeople`
  insight reliably tells you how many distinct people it tracked and the
  time windows each was in frame -- this is real, general-purpose body
  tracking, not fabricated.
* **A large, genuinely useful visual-labels vocabulary**, some of which
  names actions (Microsoft's own documented example: "swimming") -- so for
  video where the action is visually obvious and generic (walking,
  running, sitting, etc.), Video Indexer's labels will often catch it.
* **Transparent, inspectable person<->activity association** built from
  real timestamp overlap -- and, just as importantly, a system that is
  honest and visibly conservative about *not* claiming an association
  when the evidence doesn't support one (see the "running" example
  above). This is the strongest thing to walk a client through: point at
  an unassigned activity and its `association.note`, and show that the
  tool won't guess.
* **A clean, inspectable JSON contract** (`*.activities.json`) a real
  application could build a UI or downstream pipeline on, independent of
  Video Indexer's much larger raw response.

## Known limitations (be upfront about these)

* **No dedicated action/activity-recognition model.** Video Indexer has
  no API that takes "what is this person doing" as its actual question.
  "Activities" here are a subset of the general `labels` insight,
  filtered by this script's own display heuristic (an "-ing" suffix, or a
  small curated word list -- see `_looks_like_activity` /
  `_ACTIVITY_WORD_HINTS` / `_NON_ACTIVITY_LABELS` in `analyze_video.py`).
  Real, specific actions a client might ask about by name (a particular
  gesture, a safety-relevant motion, a sport-specific move) will very
  likely **not** be reliably detected unless Video Indexer's general
  label vocabulary happens to name that exact thing. If a client needs
  that, it needs a dedicated action-recognition model layered on top --
  see "Extending this" below. This same heuristic is also why a compound
  scene/place label like "swimming pool" is curated out of
  `_NON_ACTIVITY_LABELS` explicitly (confirmed against a real account: a
  "swimming" video's labels also included "swimming pool" alongside it)
  -- Video Indexer never marks a label as an object vs. an action, so a
  label containing a genuine activity word can still just name a place.
* **No bounding boxes / spatial coordinates.** Checked directly against
  Microsoft's current documentation for `labels`, `detectedObjects`, and
  `observedPeople` while building this: none of the three documents a
  bounding-box or (x, y) field on their instances, only start/end
  timestamps. This script always reports `"bounding_box": null` rather
  than inventing spatial data the API doesn't provide. If a future API
  version adds it, or you plug in a real object-tracking/pose model, that
  field is exactly where to wire it in.
* **Person<->activity association is inferred, not given.** Video Indexer
  never links a label occurrence to a specific tracked person anywhere in
  its schema. Every association in this tool's output is this script's
  own conservative overlap inference (see above) -- always labeled as such
  in both the JSON (`association.provided_by_video_indexer: false`) and
  the report's NOTES section, and left unassigned whenever more than one
  person overlaps, or none do.
* **Face recognition / matched names typically need account approval.**
  `observedPeople[].matchingFace` (Video Indexer's own link from a
  tracked body to a specific named face) is documented as needing Face
  Recognition access approval on the account. Until/unless that's
  granted, expect every person to show up as a generic "Person N", not a
  name -- which is itself a reasonable, honest default for a client demo
  that hasn't done that approval process.
* **`Advanced` indexing preset costs more and takes longer** than
  `Default` -- it's required here because `observedPeople` (this
  prototype's people-tracking) is Advanced-only. Set
  `AVI_INDEXING_PRESET=Default` in `.env` only if you don't need person
  tracking at all.
* **Filename-based output naming can collide.** Two videos with the same
  filename from different folders will overwrite each other's output in
  `output/` (this is intentionally minimal -- see "Project structure").
  Rename before processing if that's a real risk for your input set.
* **No retry/resume mid-processing.** If this script is interrupted while
  Video Indexer is still processing a video, re-running it will not reuse
  that Video Indexer video ID -- it uploads again. For a proof of concept
  this is an acceptable simplification; a production version would want
  to persist the video ID between runs.

## Extending this

The parsing layer is intentionally factored so a real activity-
recognition model could be dropped in without touching the Video Indexer
integration:

* `_extract_labels` / `_extract_objects` / `_extract_people` each build a
  plain dataclass from Video Indexer's raw insights -- unrelated to how
  "activities" get built.
* `_build_activities(labels, people)` is the one function that currently
  defines "activity" as "a label that looks like one". A second model's
  output (e.g. frame-level pose/action classifications) could be turned
  into the same `Activity` dataclass and merged in here, with its own
  `source` value (instead of `"source": "labels"`) so the report and JSON
  keep being explicit about where each detection actually came from.
* `_instance_dict` is the one place a real bounding box would get added,
  the moment a source that actually provides one exists.

## Testing

```bash
pip install -r requirements.txt
pytest tests/ -v
```

No Azure account or network access needed: `tests/test_analyze_video.py`
builds synthetic raw Video Indexer payloads (shaped like the real
`videos[0].insights.{labels,observedPeople,faces,detectedObjects}`
schema) and exercises parsing, normalization, activity/person association
(including the ambiguous- and no-overlap cases), report rendering, and
the batch/skip/`--force` orchestration via a hand-rolled fake client --
never the real network-calling `VideoIndexerClient`.

## Project structure

```
project/
├── input/                    # drop videos here for batch mode
├── output/                   # *.insights.json / *.activities.json / *.report.txt
├── tests/
│   └── test_analyze_video.py
├── analyze_video.py
├── requirements.txt
├── .env.example
└── README.md
```

Deliberately just one script -- no package, no framework, no database, no
web server, no Docker. Everything Video Indexer-specific (auth, upload,
polling) and everything parsing/reporting-specific lives in
`analyze_video.py`, organized top to bottom as: constants/config -> the
Video Indexer client -> insight extraction/normalization -> JSON/report
output -> batch orchestration -> CLI.
