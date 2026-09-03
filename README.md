# Azure Action Monitoring — Prototype

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests: pytest](https://img.shields.io/badge/tests-pytest-informational)](#running-the-tests)

A working Python prototype that uploads a video to **Azure AI Video Indexer**,
waits for it to be processed, then scans the resulting insights for a
specific action of interest (default: **"jumping"**) and produces an
oversight report: a console summary, a JSON file, and a self-contained HTML
timeline you can open in any browser. A second CLI reports *every* action
Video Indexer detected (not just one), and a third burns the results into an
annotated copy of the video.

Built and tested against the sample video `people_jumping.mp4` you provided
(jumping, id = qxxv5h1be9) (boxing, id = tuxagrberr).

## Contents

- [How it works](#how-it-works)
- [Project structure](#project-structure)
- [Requirements](#requirements)
- [Azure setup (one-time)](#azure-setup-one-time)
- [Setup](#setup)
- [Run it](#run-it)
- [Detecting every action in a video](#detecting-every-action-in-a-video)
  - [What if the action/object I need isn't detected?](#what-if-the-actionobject-i-need-isnt-detected)
  - [Composite (derived) actions](#composite-derived-actions)
- [Saving an annotated video](#saving-an-annotated-video)
  - [Quick recipes: annotated video with composite actions](#quick-recipes-annotated-video-with-composite-actions)
- [Running the tests](#running-the-tests)
- [Troubleshooting: authentication & permissions](#troubleshooting-authentication--permissions)
- [Limitations & honest caveats](#limitations--honest-caveats)
- [License](#license)
- [Sources consulted](#sources-consulted)

## How it works

```
your video ──▶ Video Indexer (upload + AI analysis) ──▶ insights JSON
                                                              │
                                                              ▼
                                          action_analyzer.py (match "jumping"
                                          against labels/keywords + timestamps)
                                                              │
                                                              ▼
                                     report.py ──▶ report.json + report.html
```

- `src/video_indexer_client.py` — handles Azure AD auth, uploads the video,
  polls until processing finishes, fetches the insights index.
- `src/action_analyzer.py` — pulls out every timestamped instance of your
  action of interest (plus a small synonym list, e.g. "jump" also matches
  "jumping"/"leap"/"hop"; "cash" matches text printed on currency, read via
  OCR), counts distinct people Video Indexer tracked, and (`analyze_all`)
  extracts *every* detected label/keyword/object/OCR-text for the "all
  actions" report below.
- `src/report.py` — renders the results as text/JSON/HTML, for both the
  single-action and all-actions reports.
- `src/video_annotator.py` — burns a per-frame "active actions" overlay into
  a copy of the video, for actions above a confidence threshold.
- `main.py` — the CLI that searches for one action of interest.
- `detect_all_actions.py` — the CLI that reports every action Video Indexer
  detected in the video (see "Detecting every action in a video" below).
- `annotate_video.py` — the CLI that saves an annotated video (see "Saving
  an annotated video" below).
- `tests/` — unit tests that verify the extraction/reporting/annotation
  logic against a realistic sample insights payload (and a tiny synthetic
  clip for the annotator), so you can trust the code without needing a live
  Azure account to run the test suite.

## Project structure

```
azure-video-action-monitoring/
├── main.py                  # CLI: find one action of interest (default "jumping")
├── detect_all_actions.py    # CLI: report every action detected, timestamped
├── annotate_video.py        # CLI: burn detected actions into a copy of the video
├── src/
│   ├── config.py            # Loads/validates Settings from .env / environment
│   ├── video_indexer_client.py  # Azure AD auth + Video Indexer upload/poll/fetch
│   ├── action_analyzer.py   # Turns raw insights into ActionReport / AllActionsReport
│   ├── report.py            # Renders reports as console text / JSON / HTML
│   └── video_annotator.py   # Burns overlays into video frames (OpenCV + ffmpeg)
├── tests/                   # Unit tests (no live Azure account required)
├── input/                   # Your local video files (gitignored)
├── output/                  # Generated reports / annotated videos (gitignored)
├── .env.example             # Template for required environment variables
└── requirements.txt
```

## Requirements

- Python 3.10+ (developed and tested on 3.14).
- A **standard (paid) Azure AI Video Indexer account** — see
  [Azure setup](#azure-setup-one-time) below.
- [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli)
  installed and available on `PATH`, for local `az login` authentication
  (not required if you're using a service principal instead — see
  [Troubleshooting](#troubleshooting-authentication--permissions)).
- `ffmpeg` on `PATH`, optional — only needed by `annotate_video.py` to keep
  the original audio track in the annotated output. Without it, the
  annotated video is still produced, just silent.

## Azure setup (one-time)

You need a **standard (paid) Azure AI Video Indexer account** — the old free
trial accounts can't be driven from the API. Steps:

1. In the [Azure portal](https://portal.azure.com), create a resource of
   type **Azure AI Video Indexer**. This is the modern "ARM-based" account
   type; give it a resource group, name, and region (e.g. `eastus`).
2. Once created, open the resource's **Overview** page and copy the
   **Account ID** (a GUID — different from the resource name).
3. Note the **Subscription ID**, **Resource Group** name, **Account name**
   (the ARM resource name), and **Location/region** — you'll put all of
   these into `.env`.
4. Grant your Azure identity (your own `az login` user, or a service
   principal) at least **Contributor** access on the Video Indexer resource
   (or its resource group), since this app calls the ARM `generateAccessToken`
   operation on your behalf. See
   [Troubleshooting](#troubleshooting-authentication--permissions) if you hit
   permission errors here — it's the single most common setup snag.

Authentication uses `azure-identity`'s `DefaultAzureCredential`, so whatever
you already use to talk to Azure works here too:

- Local development: run `az login` once; no extra env vars needed.
- Unattended/CI use: create a service principal and set
  `AZURE_TENANT_ID`, `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET` in the
  environment.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env with your AVI_SUBSCRIPTION_ID / AVI_RESOURCE_GROUP /
# AVI_ACCOUNT_NAME / AVI_ACCOUNT_ID / AVI_LOCATION

az login                         # or set AZURE_CLIENT_ID/SECRET/TENANT_ID
```

Once configured, sanity-check your setup without uploading anything:

```bash
python main.py --check-auth
```

## Run it

```bash
python main.py --video people_jumping.mp4 --action jumping
```

First run uploads and indexes the video (this can take a few minutes
depending on length — a short clip is usually done in under 2 minutes). The
CLI prints the video ID; on later runs against the same video, skip
re-uploading with:

```bash
python main.py --video people_jumping.mp4 --action jumping --video-id <id-from-first-run>
```

Outputs land in `./output/`:

- `<video>.report.json` — machine-readable event list + summary stats.
- `<video>.report.html` — open this in a browser for the timeline view.
- `<video>.raw_insights.json` — only with `--save-raw-insights`, the full
  Video Indexer response, useful for debugging or extending the analyzer.

Useful flags:

- `--action <term>` — change what you're looking for (e.g. `--action fall`).
- `--synonym <term>` — add extra matching terms (repeatable), e.g.
  `--synonym trampoline --synonym hop`.
- `--indexing-preset <preset>` — Video Indexer preset to upload with, e.g.
  `Advanced` for richer object/people insights (fresh uploads only — see
  [What if the action/object I need isn't
  detected?](#what-if-the-actionobject-i-need-isnt-detected)).
- `--min-overlap-seconds <n>` — for a composite `--action` like `"using
  phone"` or `"handling cash"`, drop matches whose underlying overlap is
  shorter than this — see [Composite (derived)
  actions](#composite-derived-actions).
- `--check-auth` — verify Azure auth + account access only, no upload.
- `-v` — verbose/debug logging.

## Detecting every action in a video

`main.py` searches for one action of interest at a time. If you instead want
a full "what happened, and when" timeline — every label/keyword Video
Indexer tagged anywhere in the video, not filtered to a single term — use
`detect_all_actions.py`:

```bash
python detect_all_actions.py --video people_jumping.mp4
```

It uses the same upload/auth/polling client as `main.py` (so `--video-id` to
reuse an already-indexed video, `-v` for debug logging, `--check-auth`, and
`--save-raw-insights` all work the same way), then reports every detected
action grouped by name — occurrence count, first/last timestamp, total
flagged time, and max confidence per action — plus the full chronological
occurrence log. This includes three insight buckets: `labels`, `keywords`,
and `detectedObjects` (Video Indexer's object detector — logged with
source `objects`).

Outputs land in `./output/`:

- `<video>.all_actions.json` — machine-readable per-action summaries + the
  full occurrence list.
- `<video>.all_actions.html` — open in a browser for a per-action timeline
  (one lane per distinct action), a summary table, and the full log.

Extra flags:

- `--min-confidence <0.0-1.0>` — drop occurrences whose confidence score is
  below this threshold. Occurrences with no confidence value (some insight
  buckets don't score every item) are always kept.
- `--indexing-preset <preset>` — see below.
- `--min-overlap-seconds <n>` / `--merge-gap-seconds <n>` — tune composite
  (derived) actions like "person using phone" — see [Composite (derived)
  actions](#composite-derived-actions) below.

Same caveat as above applies here too: this reports whatever Video Indexer's
default model tagged in `labels`/`keywords`/`detectedObjects`/`ocr` — it is
not a dedicated action classifier, so it's a strong first pass, not a
guaranteed-recall detector.

### What if the action/object I need isn't detected?

First, check whether the concept exists in Video Indexer's vocabulary at
all — no flag or synonym list can surface something that was never in
scope to begin with:

- `labels`/`keywords` tag general scenes, objects, and a limited set of
  visual concepts (see [labels
  insight](https://learn.microsoft.com/en-us/azure/azure-video-indexer/labels-identification-insight)).
- `detectedObjects` uses a fixed ~80-class vocabulary shared with
  Microsoft's generic vision object detector — everyday things like
  vehicles, furniture, food, and sports equipment (see [object detection
  insight](https://learn.microsoft.com/en-us/azure/azure-video-indexer/object-detection-insight)).
- `ocr` reads printed/on-screen text, independent of what it's printed
  on — this one *isn't* a fixed concept vocabulary, so it's the exception
  to the rule below.

None of `labels`/`keywords`/`detectedObjects` include domain-specific
concepts like **money, cash, currency, or the act of counting it** — a
video of someone counting cash will only get tagged with whatever's
generically in frame (`person`, `clothing`, `indoor`, `office supplies`,
etc.), never "cash" itself, no matter what `--synonym` you add. This isn't
a bug or a config issue; the concept simply isn't one those models were
trained to recognize.

**`ocr` is how "cash" becomes detectable anyway.** A banknote has no
"money" object class, but it does have printed text — a denomination
number, a motto, a serial number — and OCR reads that regardless. Running
`detect_all_actions.py --indexing-preset Advanced --save-raw-insights` on
an actual dollar-bill clip surfaced exactly this:

```
00:00.08  00:03.00  ocr  100                 0.996   <- denomination
00:00.08  00:00.60  ocr  WE TRUST            0.998   <- "IN GOD WE TRUST"
00:01.04  00:07.32  ocr  06161268 A          0.9905  <- serial number
```

`action_analyzer.py` now reads this bucket too (source `"ocr"`), and
`DEFAULT_SYNONYMS["cash"]`/`["money"]` (in `src/constants.py`) match
phrases known to be printed on US currency (`"we trust"`,
`"federal reserve"`, `"united states of america"`, `"legal tender"`,
`"dollar"`, `"currency"`, `"banknote"`), so `--action cash` picks up hits
like `WE TRUST` above. A bare denomination number like `"100"` deliberately
isn't a match target — too ambiguous alone (room numbers, percentages,
prices) to auto-match — so treat those OCR rows as something to eyeball in
`--save-raw-insights` output rather than something the CLI flags for you.

Treat this as a manual-review aid, not a reliable detector: OCR on video
frames is noisy (motion blur, rotation, and partial reads mean the same
text often shows up fragmented or duplicated across frames, as the serial
number above illustrates), and it only catches text that's both in frame
and legible — cash lying face-down, out of focus, or a banknote design
with different printed text (non-US currency, older series) won't match
the same way.

If the concept genuinely isn't visible as text and isn't in any of these
vocabularies either, the only reliable fix is a custom-trained model — see
[Limitations & honest caveats](#limitations--honest-caveats) below for that
upgrade path (e.g. an Azure Custom Vision classifier trained on "cash
visible" vs. not, plugged in alongside `video_indexer_client.py` as a
second insights source).

### Composite (derived) actions

Some things worth reporting — "a person using a phone", "a person handling
cash" — aren't any single label, keyword, or object Video Indexer returns.
`action_analyzer.py`'s `_derive_composite_actions` synthesizes these
(source `"derived"`) from **temporal overlap** between two independently
detected things already in the report. A person-like label and a
`cell phone` object both present at the same moment is a reasonable,
inspectable proxy for "person using phone" — Video Indexer never tagged
that concept directly, but the two things it did tag line up in time.

Confidence is the **minimum** of the two overlapping items' confidences (a
composite claim is only as strong as its weaker piece of evidence), and
each derived row records its `evidence` — the two contributing item names —
so you can see why it fired without re-running `--save-raw-insights`.
Current composites, defined in `COMPOSITE_ACTIONS` (`src/constants.py`):

| Derived action | Left | Right | Caveat |
| --- | --- | --- | --- |
| `person using phone` | person-like label | `cell phone` object | Overlap is circumstantial: they could be nearby without the phone being touched. |
| `person driving car` | person-like label | `car`/`outdoor vehicle`/`vehicle` object | Same circumstantial caveat — a person and a car in frame together isn't proof the person is driving it (could be a pedestrian, passenger, or bystander). |
| `person playing sports` | person-like label | `sports equipment`/`athletic game`/`ball`/`bat`/`racket` object | Same caveat, generalized further — any of these objects near a person matches, not necessarily one they're actively using. |
| `person handling cash` | person-like label | OCR text matching `DEFAULT_SYNONYMS["cash"]` | Only fires when currency text is both in frame and legible (see OCR caveats above) — misses cash that's present but unreadable. |
| `person at register` | person-like label | `keyboard`/`laptop`/`computer`/`monitor`/`tv` object | Weakest of the five — Video Indexer has no "cash register"/"POS terminal" class, so this leans on the closest generic object classes available. A person near any keyboard or screen matches, not specifically one behind a checkout counter. |

`COMPOSITE_ACTIONS` entries reference a `CompositeSide`, which matches
either exact detected names (`names={"cell phone"}`) or a synonym family
by substring, same as `DEFAULT_SYNONYMS` (`synonyms=(...)`) — the latter is
what makes `person handling cash` possible at all, since OCR text varies
per banknote and no fixed set of exact strings could cover it. Add a new
row to extend detection to another person+object overlap without touching
the derivation logic itself.

Verified against a real ~28s clip of a person standing and using their
phone (`detect_all_actions.py --video-id f9m21irr9l`):

```
   Start       End  Source    Action                    Confidence  Evidence
00:00.08  00:28.44  labels    person                    1.00        -
00:00.00  00:18.84  objects   cell phone                0.76        -
00:00.08  00:18.84  derived   person using phone        0.76        person (labels) + cell phone (objects)
00:19.00  00:19.44  objects   cell phone                0.71        -
00:19.00  00:19.44  derived   person using phone        0.71        person (labels) + cell phone (objects)
00:19.60  00:28.48  objects   cell phone                0.81        -
00:19.60  00:28.44  derived   person using phone        0.81        person (labels) + cell phone (objects)
```

(Trimmed to the relevant rows; the full report also lists `car`, `building`,
`clothing`, `outdoor`, all unrelated to the phone.) Note this run used
`--min-overlap-seconds` at its default — a fourth, 0.04-second overlap that
showed up in the raw data (a single-frame detector blip, not a real
occurrence) was filtered out. That flag and its companion:

- `--min-overlap-seconds <n>` (`detect_all_actions.py` / `main.py`) — drop
  a derived overlap shorter than this many seconds. Defaults to `0.15`;
  pass `0` to see every overlap Video Indexer's raw timestamps produce,
  jitter included.
- `--merge-gap-seconds <n>` (`detect_all_actions.py` only) — off by
  default. When set, collapses occurrences of the *same* action within
  that many seconds of each other (or overlapping) into one combined
  occurrence — useful once you're past debugging and just want "the
  cashier picked up their phone 3 separate times" rather than a wall of
  near-duplicate rows from Video Indexer's frame-level fragmentation.

**Honest limitation: this can't tell *which* person is which.** A derived
`person using phone` (or `person at register`) means *some* detected person
overlapped the other evidence at that time — not a specific, identified
person. If more than one person is ever in frame (e.g. a cashier *and* a
customer), nothing here can currently attribute the action to one over the
other. This isn't a gap in this project's code: Video Indexer's own
documented schemas for [observed
people](https://learn.microsoft.com/en-us/azure/azure-video-indexer/observed-matched-people-insight)
and [faces](https://learn.microsoft.com/en-us/azure/azure-video-indexer/face-detection-insight)
carry only start/end timestamps, no bounding-box or other spatial
coordinates — confirmed both against those docs and against this project's
own `--save-raw-insights` captures. If distinguishing roles ever becomes a
requirement, the honest options are: check whether a different Video
Indexer tier or API surface exposes spatial data before building around
it, or add a separate, local computer-vision pass (e.g. a lightweight
person tracker establishing a fixed "counter region") as a second insights
source alongside `video_indexer_client.py` — real additional engineering,
not a config flag.

## Saving an annotated video

`annotate_video.py` renders a copy of the video with a burned-in overlay
listing every action active at each moment, for whichever actions clear a
confidence threshold you choose:

```bash
# Only have a video ID? No local file needed -- its source gets downloaded for you.
python annotate_video.py --video-id qxxv5h1be9 --min-confidence 0.6

# Have the local file too? Skip the download and read frames straight from it.
python annotate_video.py --video people_jumping.mp4 --video-id qxxv5h1be9 --min-confidence 0.6
```

You need **at least one** of `--video` / `--video-id` — not both required:

- `--video` alone — uploads and indexes it from scratch, then annotates that
  same local file.
- `--video-id` alone — reuses an already-indexed video's insights, and
  downloads its source file from Video Indexer to get frames from (saved to
  `<out-dir>/<video-id>.source.mp4`, reused on subsequent runs).
- both — reuses the insights (skips re-uploading) *and* reads frames from
  the local file (skips the download); the fastest combination if you have
  both on hand.

The actions to draw come from the same insights call as above, unless you
add `--report` to reuse a report already saved to disk instead (skips the
insights call, though a video source is still needed for frames per the
rules above). `--report` accepts either report file this project writes:

- `--report output/<video>.all_actions.json` — every action `detect_all_actions.py` found.
- `--report output/<video>.report.json` — just the occurrences of one action
  `main.py --action <...>` searched for (e.g. a `--action "using phone"` run,
  to annotate only the derived "person using phone" moments rather than
  every action in the video).

Important honest caveat: Video Indexer's `labels`/`keywords` insights are
frame-level tags with no spatial coordinates for the general model used
here, so the overlay is a caption box listing what's active (e.g. "boxing
(0.97)"), not a box drawn around the person doing it.

Flags:

- `--min-confidence <0.0-1.0>` — only draw actions with confidence at or
  above this. Default: `0.5`. Unlike the other two scripts, actions with no
  confidence score are **excluded** by default here (a threshold can't be
  compared against an unknown score) — pass `--include-unscored` to draw
  them regardless of the threshold.
- `--only-composite` — drop every raw label/keyword/object/ocr detection
  and draw **only** composite/derived actions (source `"derived"`, e.g.
  `person using phone`, `person handling cash`) — see [Composite (derived)
  actions](#composite-derived-actions). The default output filename gets a
  `.composite_actions` suffix (`<name>.composite_actions.annotated.mp4`) so
  it doesn't overwrite a plain annotated run of the same video.
- `--max-lines <n>` — cap simultaneous action lines drawn per frame before
  collapsing the rest into a "+N more" line. Default: `6`.
- `--out <path>` — output video path. Default: `<out-dir>/<name>.annotated.mp4`.
- `--out-dir <dir>` — directory for default outputs, including a downloaded
  source file when `--video` is omitted. Default: `./output`.
- `--no-audio` — skip audio muxing even if `ffmpeg` is available (faster,
  silent output).

Requires `opencv-python-headless` (in `requirements.txt`) to render frames,
and **`ffmpeg` on your `PATH`** to carry the original audio track over —
without it, the output is silent and a warning is logged; the video itself
still renders fine either way.

### Quick recipes: annotated video with composite actions

Two ways to get an annotated video that includes composite ("derived")
actions like `person using phone` or `person handling cash` — assuming
your source video is in `input/` (swap `input/your_video.mp4` for your
actual file). Both are two-command recipes: index once, then annotate
without any further Azure calls.

**Recipe A — annotate with every detected action** (composites plus
everything else Video Indexer tagged — `car`, `person`, `outdoor`, etc.):

```bash
python detect_all_actions.py --video input/your_video.mp4
python annotate_video.py --video input/your_video.mp4 --report output/your_video.all_actions.json
```

The first command uploads, indexes, and writes
`output/your_video.all_actions.json` (every action, composites included,
per [Composite (derived) actions](#composite-derived-actions) above). The
second reuses that saved report and the local file for frames, so it
makes no Azure calls at all — just burns in the overlay.

**Recipe B — annotate with only one composite action isolated** (a
cleaner overlay when you just want to show, say, phone use, without every
other label cluttering the video). Reuse the video ID the first command
above printed to skip a second upload:

```bash
python main.py --video input/your_video.mp4 --video-id <id-from-recipe-A> --action "using phone"
python annotate_video.py --video input/your_video.mp4 --report output/your_video.report.json
```

`--action` takes the composite's name (or any distinctive substring of
it) — swap `"using phone"` for `"driving car"`, `"playing sports"`,
`"handling cash"`, or `"at register"` for the other composites currently
defined in `COMPOSITE_ACTIONS` (`src/constants.py`). If you haven't run
Recipe A first and don't have a video ID yet, drop `--video-id` from the
`main.py` command — it'll upload the video itself instead (one Azure
upload either way, just on whichever command runs first).

Watch out: `--action` is a substring match against **every** detection,
not just composites — `--action phone` also matches the raw `cell phone`
object on its own, so Recipe B's overlay can still include a non-composite
hit alongside the composite you meant to isolate.

**Recipe C — annotate with every composite action, and nothing else** (the
direct answer if what you want is "show me only the derived/composite
actions, never the raw labels/objects/ocr they're built from"):

```bash
python detect_all_actions.py --video input/your_video.mp4
python annotate_video.py --video input/your_video.mp4 --report output/your_video.all_actions.json --only-composite
```

Same two commands as Recipe A, plus `--only-composite` on the second one.
Unlike Recipe B, this can't accidentally pick up a non-composite match —
it filters strictly on `source == "derived"`, so the overlay only ever
shows actions actually synthesized by `COMPOSITE_ACTIONS`, across however
many composite types the video contains at once.

## Running the tests

No Azure account needed — these test the extraction and report-rendering
logic against a realistic sample insights payload in
`tests/fixtures/sample_insights_jumping.json`:

```bash
pip install pytest
python -m pytest tests/ -v
```

## Troubleshooting: authentication & permissions

All three CLIs authenticate the same way (`src/video_indexer_client.py`,
via `azure-identity`'s `DefaultAzureCredential`), and hit Azure in two
stages — an Azure AD token, then an ARM call to mint a Video Indexer
data-plane token. Almost every setup issue shows up as one of these:

| Symptom | Cause | Fix |
| --- | --- | --- |
| `DefaultAzureCredential failed to retrieve a token from the included credentials` (every credential in the chain fails, including `AzureCliCredential: Failed to invoke the Azure CLI`) | No credential source is available on this machine — most often Azure CLI isn't installed or isn't on `PATH` (this specific message means the `az` executable couldn't be found/run, not just "you're logged out"). | Install the [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli), open a **fresh** terminal so `PATH` updates, then run `az login`. For unattended/CI use instead, set `AZURE_TENANT_ID` / `AZURE_CLIENT_ID` / `AZURE_CLIENT_SECRET` (a service principal) in the environment. |
| `DefaultAzureCredential acquired a token from AzureCliCredential` but then `generateAccessToken failed (404) ... SubscriptionNotFound` | Either `AVI_SUBSCRIPTION_ID` and `AVI_ACCOUNT_ID` got swapped/mistyped in `.env` (both are GUIDs), or `az login` authenticated into a different Azure AD tenant than the one owning the subscription. | Re-copy the Subscription ID from the resource's Overview page in the portal. Run `az account show` to see the active tenant/subscription, and `az login --tenant <tenant-id>` if it's the wrong one. |
| `generateAccessToken failed (403): AuthorizationFailed ... does not have authorization to perform action 'Microsoft.VideoIndexer/accounts/generateAccessToken/action'` | Your identity authenticated fine but has no RBAC role on the Video Indexer resource that includes this action. There's no narrower built-in role for it — it needs **Contributor** (or Owner); the read-only **Video Indexer Restricted Viewer** role explicitly excludes token generation. | Have someone with Owner/User Access Administrator on the resource (or its resource group) run: `az role assignment create --assignee "<your-email>" --role "Contributor" --scope "/subscriptions/<sub-id>/resourceGroups/<rg>/providers/Microsoft.VideoIndexer/accounts/<account>"` |
| `az role assignment create` itself fails with `AuthorizationFailed ... Microsoft.Authorization/roleAssignments/write` | Assigning a role is a separate, higher permission than using a resource — `Contributor` alone doesn't grant it. | Find who has Owner/User Access Administrator with `az role assignment list --scope <rg-scope> --include-inherited --output table`, and ask them to run the command above (or do it in the Portal: resource → **Access control (IAM)** → **Add role assignment**). |

If you're stuck past all of the above, `python main.py --check-auth` (or the
same flag on `detect_all_actions.py`) isolates the auth/permissions path
from the upload, so you're not waiting on a large file transfer to find out
whether config or access is the problem.

## Limitations & honest caveats

- **Video Indexer's default model tags general visual concepts** (people,
  objects, scenes, activities it recognizes) via its `labels` and `keywords`
  insights — it is not a dedicated, guaranteed-recall classifier for one
  specific motion like "a person jumping." If your clip doesn't get tagged
  with a jumping-related label, `main.py --save-raw-insights` lets you
  inspect exactly what Video Indexer did detect, and you can widen matching
  with `--synonym`.
- **For reliable, purpose-built action detection** (e.g. compliance-grade
  "did anyone jump the fence" monitoring), the recommended next step is a
  custom-trained model: label a set of clips and train an **Azure Custom
  Vision** classifier, or a pose/action-recognition model deployed via
  **Azure Machine Learning**, and swap `video_indexer_client.py` /
  `action_analyzer.py` for a client against that model. The rest of this
  app (upload/poll pattern, report rendering, CLI, tests) stays the same.
- **Person tracking** here is just a count of Video Indexer's
  "observed people" — it does not currently attribute a specific jump event
  to a specific tracked person. That association is possible with Video
  Indexer's per-person timeline data if needed later; flagged here rather
  than silently implied.
- No live Azure credentials were available in the environment this was
  originally built in, so the upload/auth/polling path
  (`video_indexer_client.py`) was implemented against the current,
  documented ARM-based API before being exercised against a live account.
  The extraction/report logic (`action_analyzer.py`, `report.py`) *is*
  fully unit-tested independent of that (see [Running the
  tests](#running-the-tests)).

## License

[MIT](LICENSE) — see the `LICENSE` file for the full text.

## Sources consulted

- [Azure AI Video Indexer REST API reference](https://learn.microsoft.com/en-us/rest/api/videoindexer/)
- [Generate Access Token (ARM) operation reference](https://learn.microsoft.com/en-us/rest/api/videoindexer/generate/access-token)
- [Azure AI Video Indexer — use APIs](https://learn.microsoft.com/en-us/azure/azure-video-indexer/video-indexer-use-apis)
- [Azure AI Video Indexer FAQ](https://learn.microsoft.com/en-us/azure/azure-video-indexer/faq)
- [Azure AI Video Indexer — restricted viewer role](https://learn.microsoft.com/en-us/azure/azure-video-indexer/restricted-viewer-role)
