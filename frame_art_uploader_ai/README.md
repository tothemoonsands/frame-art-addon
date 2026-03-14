# Frame Art Uploader Add-on

This add-on keeps Samsung Frame Art Mode aligned with the artwork flows you are building in Home Assistant.
It can upload the newest AI-generated image, generate music-inspired widescreen art, selectively show built-in Samsung gallery pieces, and keep local and TV-side artwork catalogs tidy over time.

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
