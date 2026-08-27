# Azure Action Monitoring — Prototype

A working Python prototype that uploads a video to **Azure AI Video Indexer**,
waits for it to be processed, then scans the resulting insights for a
specific action of interest (default: **"jumping"**) and produces an
oversight report: a console summary, a JSON file, and a self-contained HTML
timeline you can open in any browser.

Built and tested against the sample video `people_jumping.mp4` you provided 
(jumping, id = qxxv5h1be9) (boxing, id = tuxagrberr).

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
  "jumping"/"leap"/"hop"), counts distinct people Video Indexer tracked, and
  (`analyze_all`) extracts *every* detected label/keyword for the "all
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
   operation on your behalf.

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
reuse an already-indexed video, `-v` for debug logging, and
`--save-raw-insights` all work the same way), then reports every detected
action grouped by name — occurrence count, first/last timestamp, total
flagged time, and max confidence per action — plus the full chronological
occurrence log.

Outputs land in `./output/`:

- `<video>.all_actions.json` — machine-readable per-action summaries + the
  full occurrence list.
- `<video>.all_actions.html` — open in a browser for a per-action timeline
  (one lane per distinct action), a summary table, and the full log.

Extra flag:

- `--min-confidence <0.0-1.0>` — drop occurrences whose confidence score is
  below this threshold. Occurrences with no confidence value (some insight
  buckets don't score every item) are always kept.

Same caveat as above applies here too: this reports whatever Video Indexer's
default model tagged in `labels`/`keywords` — it is not a dedicated action
classifier, so it's a strong first pass, not a guaranteed-recall detector.

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

You need **at least one** of `--video` / `--video-id` — not both:

- `--video` alone — uploads and indexes it from scratch, then annotates that
  same local file.
- `--video-id` alone — reuses an already-indexed video's insights, and
  downloads its source file from Video Indexer to get frames from (saved to
  `<out-dir>/<video-id>.source.mp4`, reused on subsequent runs).
- both — reuses the insights (skips re-uploading) *and* reads frames from
  the local file (skips the download); the fastest combination if you have
  both on hand.

The actions to draw come from the same insights call as above, unless you
add `--report output/<video>.all_actions.json` to reuse a report already
saved by `detect_all_actions.py` instead (skips the insights call, though a
video source is still needed for frames per the rules above).

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

## Running the tests

No Azure account needed — these test the extraction and report-rendering
logic against a realistic sample insights payload in
`tests/fixtures/sample_insights_jumping.json`:

```bash
pip install pytest
python -m pytest tests/ -v
```

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
  built in, so the upload/auth/polling path (`video_indexer_client.py`) is
  implemented against the current, documented ARM-based API but has not
  been exercised against a live account. The extraction/report logic
  (`action_analyzer.py`, `report.py`) *is* fully tested (see above). Run
  `python main.py -v --video people_jumping.mp4` for verbose logs the first
  time you point it at a real account, in case any endpoint detail needs a
  small adjustment.

## Sources consulted

- [Azure AI Video Indexer REST API reference](https://learn.microsoft.com/en-us/rest/api/videoindexer/)
- [Generate Access Token (ARM) operation reference](https://learn.microsoft.com/en-us/rest/api/videoindexer/generate/access-token)
- [Azure AI Video Indexer — use APIs](https://learn.microsoft.com/en-us/azure/azure-video-indexer/video-indexer-use-apis)
- [Azure AI Video Indexer FAQ](https://learn.microsoft.com/en-us/azure/azure-video-indexer/faq)
