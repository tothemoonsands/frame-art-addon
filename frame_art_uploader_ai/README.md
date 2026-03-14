# Frame Art Uploader Add-on

## Restore request queue semantics

The add-on now treats `/share/frame_art_restore_request.json` as an **inbox file** and immediately moves it into a durable queue directory at `/share/frame_art_restore_queue` using an atomic rename (`os.replace`).

### Queue behavior

- Each consumed inbox request becomes a unique JSON work item filename.
- Requests are processed in FIFO order (oldest queue file first).
- Multiple rapid writes are preserved as independent queue items.
- Malformed queue items are not allowed to block later valid items; each item is finalized and removed after processing.

### Concurrency/worker lock

- A single worker lock file (`/share/frame_art_uploader_worker.lock`) ensures only one `addon_start` invocation processes requests at a time.
- If another invocation starts while processing is active, it will still enqueue any inbox request and then exit without overlapping processing.

## Status contract (`/share/frame_art_uploader_last.json`)

For each restore queue item, the add-on writes `frame_art_uploader_last.json` as work progresses and again on completion. The payload always includes at least:

- `ts`: monotonic timestamp (`time.monotonic()`)
- `kind`: normalized request kind (if available)
- `ok`: boolean success/failure
- `error`: error string on failures
- `requested_at`: request timestamp from payload (if provided)

Seed sync requests (`ambient_seed`, `holiday_seed`, `music_seed`) also stream live progress fields that are useful in Home Assistant dashboards:

- `phase`: `scan`, `delete`, `upload`, or `done`
- `phase_action`: short machine-friendly action label such as `checking_catalog_entry`, `delete_completed`, or `uploading_cached_item`
- `phase_status`: short result label such as `queued_for_delete`, `deleted`, `already_synced`, `uploaded`, or `failed`
- `phase_index` / `phase_total`: progress counters for the current phase
- `phase_item`: current relative file or catalog key being processed
- `total_files`, `scan_total`, `scan_checked_count`
- `uploaded_count`, `skipped_count`, `failed_count`
- `deletion_candidates`, `deletion_processed`, `deletion_failed`
- `auto_queued_missing_count`

The file reflects the most recently updated request state, which means dashboards can read it for live in-flight seed progress instead of waiting for completion.

## Logging

- `debug_logging` (add-on option, default `false`): when `false`, console logs are compact and human-readable.
- Set `debug_logging: true` for full per-step verbose logs (including detailed generation pipeline fields).
- Optional: set environment variable `FRAME_ART_LOG_JSON=1` to also emit JSON log lines alongside readable lines.
