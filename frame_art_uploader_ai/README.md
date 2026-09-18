# Frame Art Uploader Add-on

This add-on keeps Samsung Frame Art Mode aligned with the artwork flows you are building in Home Assistant.
It can upload the newest AI-generated image, generate music-inspired widescreen art, selectively show built-in Samsung gallery pieces, and keep local and TV-side artwork catalogs tidy over time.

TV control is direct over the local network and does not use the SmartThings API.

## JSignals local art selection (4.1.9)

Each ambient pick now writes `/share/frame_art_jsignals_local_art.json` with a
local image path and selection time. This happens even when the Frame TV's pick
uses the Samsung gallery or is queued while Art Mode is hidden. Holiday art is
preferred when present; if that holiday folder is empty, the local selection
falls back to the current seasonal ambient folder. The record contains no
Samsung gallery artwork or image data. Home Assistant can expose it with a
command-line sensor for JSignals to read.

## Playlist and radio session backgrounds (4.1.8)

Playlist and radio requests can set `preserve_album: false`, provide a stable
`collection_name`, and include up to 20 representative `context_tracks`. The
add-on generates and caches one full-frame gallery image for the collection
without resolving or compositing an album cover. Album requests keep the
existing cover-derived pipeline. Session prompts explicitly exclude musical
equipment and music-listening objects so the Frame artwork does not visually
compete with equipment in the room. Radio station names are treated as musical
identity clues instead of literal scene requests, and the prompt steers away
from generic staged coffee, wellness, sunset, and interior-design imagery.

Before generating a new collection image, the add-on asks the configured
`openai_context_model` to create a concise curator profile from the collection
name and representative tracks. Web search is available when public context
would improve the result—for example, resolving a community station to its
place and culture—but the planner can simply reason from the music when search
would add nothing. Profiles are cached in
`/data/frame_art_session_profiles.json`, so regenerating artwork does not repeat
the research step. `session_context_planning` and
`session_context_web_search` can disable either capability independently.

## Old prompt button (4.1.5)

The HA Music Actions button **Old prompt background** regenerates the current
cover background using the original reference-background prompt and legacy
unmasked generation/rendering path. It uses the main model and forces a fresh
generation. This is a one-request override; the configured `music_pipeline`
and the frontier buttons keep their existing behavior. Normal generation error
fallbacks still apply.

Queue/API callers can set `use_legacy_prompt: true` on a `music_feedback` or
`cover_art_reference` request. It can also be combined with `use_frontier_model`
when calling the request API. Update the add-on to 4.1.5 before using the button.

## Frontier repairs (4.1.4)

The add-on Configuration menu has two independent model fields:

- **OpenAI model** (`openai_model`, default `gpt-image-2.5-flare`) for ordinary generation.
- **Frontier OpenAI model** (`openai_frontier_model`, default `gpt-image-2.5-sunburst`) for explicit one-time repairs.

The HA Music Actions buttons **Fix with frontier** and **Frontier background**
use the frontier setting for that request. Fix refreshes the cover/generation;
background regenerates around the selected cover. Both force a new generation
instead of reusing the cached final image. Existing buttons keep using the main setting.

A frontier model error is reported in the HA status sensor and a persistent
notification. The current TV artwork, existing files, and match are retained;
there is no automatic retry with another model or local replacement image.
Subsequent ordinary requests still use the main setting. Frontier outputs use
separate filenames so a failed generation cannot overwrite the current artwork.

Queue/API callers can set `use_frontier_model: true` on a `music_feedback`
request (`regen_now` or `regen_background`) or a `cover_art_reference` request.
Omitting it preserves ordinary behavior. Install add-on 4.1.4 before using the
new HA buttons, then pull/reload the HA configuration as usual.

## Local TV connection

- Keep `tv_ws_port: 8001` for the normal tokenless local connection.
- If the TV requires secure local pairing, set `tv_ws_port: 8002`. Approve the prompt on the TV the first time it connects.
- The TV-issued token is then retained at `tv_token_file` (default `/data/frame_tv_token.txt`). This is a local television token, not a SmartThings PAT, and it is preserved across add-on restarts.
- The Samsung WebSocket library is pinned to a reviewed commit so rebuilding the add-on cannot silently pull different connection behavior.

Version 4.1.3 reconnects after a Samsung `ms.channel.clientDisconnect` event
during upload setup, using the configured retry limit and backoff. The upload
reuses the saved image without repeating generation. Authentication failures and
missing upload acknowledgements still stop for inspection.

## What it does

- Uploads new artwork to Samsung Frame Art Mode and can switch to it immediately after upload.
- Preserves a true TV-friendly presentation by using 16:9 full-bleed outputs and 4K-sized artwork handling for generated cover-art flows.
- Supports a Samsung/AI blend, so automations can sometimes choose curated built-in Samsung gallery selections instead of always showing the newest generated image.
- Lets you define seasonal and ambient Samsung art pools in the add-on configuration, including holiday groups and time-of-day seasonal collections.
- Processes restore and sync requests through a durable queue so Home Assistant can safely fire repeated requests without losing work.

## Current artwork flows

### AI and local artwork uploads

The add-on watches your configured inbox folder for images that match the configured filename prefix, uploads the newest eligible item to the TV, and keeps retention cleanup under control with separate limits for TV uploads and local files.

### Samsung gallery selections

When `pick_samsung_pct` is greater than `0`, the add-on can intentionally choose from built-in Samsung artwork collections instead of uploading a fresh image.
Those selections are driven by the `Samsung Art Gallery Selections` configuration area, which now supports friendlier labels for holidays, seasons, and time-of-day buckets.

### Music cover-art generation

The add-on also supports music-focused restore jobs that can:

- look up album metadata and artwork from iTunes
- generate original full-bleed backgrounds inspired by album art
- outpaint and upscale the result into a Frame-friendly widescreen image
- reuse cached associations when possible
- queue follow-up regeneration requests from feedback flows

### Seed sync jobs

Bulk sync jobs are supported for:

- `ambient_seed`
- `holiday_seed`
- `music_seed`

These jobs scan configured folders, compare them against stored catalogs, upload missing items, optionally delete pending removals, and stream live progress so Home Assistant dashboards can show what is happening while the sync is still running.

## Queue and worker behavior

The add-on treats `/share/frame_art_restore_request.json` as an inbox file and immediately moves it into the durable queue directory at `/share/frame_art_restore_queue` using an atomic rename.

- Each consumed inbox request becomes its own JSON work item.
- Requests are processed in FIFO order.
- Multiple rapid writes are preserved as separate queue entries.
- Malformed work items are finalized without blocking later valid requests.
- A single worker lock at `/share/frame_art_uploader_worker.lock` prevents overlapping processing while still allowing new requests to be safely enqueued.

## Status output

For each queued request, the add-on writes progress and final status to `/share/frame_art_uploader_last.json`.
This payload always includes the core outcome fields below:

- `ts`
- `kind`
- `ok`
- `error`
- `requested_at`

Seed sync jobs also stream richer progress fields such as:

- `phase`, `phase_action`, `phase_status`
- `phase_index`, `phase_total`, `phase_item`
- `uploaded_count`, `skipped_count`, `failed_count`
- `deletion_candidates`, `deletion_processed`, `deletion_failed`
- `auto_queued_missing_count`

That makes the file useful both as a final status record and as a live progress source for dashboard cards and automations.

## Logging

- `debug_logging: false` keeps logs compact and readable during normal operation.
- `debug_logging: true` turns on verbose step-by-step logging for troubleshooting.
- `FRAME_ART_LOG_JSON=1` can be added as an environment variable when you also want JSON log lines.

## Local development on macOS

To make local setup repeatable across Macs, this repo includes `make` targets that create a repo-local virtual environment and install the Python dependencies used for local runtime work.

```bash
cd /Users/jsands/Documents/Code/frame-art-addon
make setup
make test
```

- `make setup` creates `.venv` and installs `requests`, `pillow`, `rapidfuzz`, and `samsungtvws`.
- `make test` runs `python -m unittest discover -s tests -v` using that local virtual environment, so you do not need to activate it first.

## Reviewed music pipeline (4.1)

`music_pipeline: seamless` uses Flare (`openai_model: gpt-image-2.5-flare`)
with the reviewed continuation prompt and a real mask: a fully opaque 614 px
cover centered on a 1536×1024 reference, with a transparent editable exterior.
Black title bands are continued as graphic design rather than painted over.
The original cover is restored locally at 1536 px on the final 3840×2160 image.
The finish matches the original library renderer: black shadow alpha 88/255,
26 px blur, 16 px downward offset, and **no edge feather**. The review viewer's
corresponding preset is **Original standard**, with feather Off.

Applicable definitive model/compatibility errors can fall back through Sunburst,
GPT Image 2, and 1.5 with continuation, followed by the legacy prompt on 2 and 1.5.
Safety refusals are not replayed against other models, and uncertain network or
server outcomes do not trigger another paid request. The existing local fallback
remains available with the same final finish. The known Nevermind cover uses a
water-only reference; the full original is composited locally. Every successful
continuation records its actual model, prompt, mask, usage and finish beside the
source as a `.recipe.json` file. Sources, backgrounds, widescreen PNGs and TV JPEGs
remain separate. `music_pipeline: legacy` preserves the former generation path.

## Reviewed-library migration

This is a separate, explicit maintenance import; the dashboard's ordinary Library
Sync button does not launch it. `tools/frame_art_review/build_release.py` freezes
confirmed selections and hashes. Prepared 4K JPEGs are uploaded without a second
lossy encode, and the TV's returned content ID is used directly. Ordinary sync now
recognizes changed files when their previously recorded source hash differs.

A control file at `/share/frame_art_migration/active.json` contains:

```json
{"release":"/media/frame_art_release", "run":"/share/frame_art_migration/runs/release-1", "paused":false}
```

While this file exists, the launcher holds normal queued work and runs
`migration.py` under the shared worker lock. The release is verified, then current
media and metadata are copied into the run's `backup` directory and checksummed.
The worker deletes only mapped superseded music IDs, verifies absence, uploads
one approved replacement, verifies its acknowledged ID in the TV inventory, and commits
its mapping. There is at most one outstanding album replacement. Operations are
paced at least ten seconds apart, with a one-minute rest after the first album and
each following five albums. These are conservative operating settings, not a
Samsung rate-limit claim. Saved unchanged images and unrelated TV art are retained.

Progress and recovery state are in `status.json` and `journal.json` under the run
directory. On failure the control file is paused; after resolving the failure,
set `paused` to false to continue from the journal. An `upload_started` entry with
no acknowledged ID **must be reconciled manually** against its saved pre-upload
inventory before resuming; the worker will never blindly upload it again. Backups
are retained for restoring deleted artwork if needed; restored uploads get new TV
IDs. Final catalog/index/manifest/alias updates are deterministic and repeatable
under maintenance. Only after final verification is the control file removed and
normal queue processing restored.
