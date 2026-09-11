# Local Frame Art review

This companion app snapshots the existing library, generates versioned candidates, and stores review decisions in SQLite. Generation and review stay local. The separate, explicitly started release builder and staging tools hand approved artwork to the Home Assistant migration worker, which controls the TV under maintenance.

The pilot and batch descriptions below document this run's history, not a universal recipe for future runs. See [the repeatable-upgrade plan](UPGRADE_WORKFLOW.md) for current capabilities, lessons learned, and remaining implementation work. Production behavior and the migration contract are documented in [the add-on README](../../frame_art_uploader_ai/README.md#reviewed-library-migration).

## Run

The post-deployment manual review portal is at `/manual-review`. It lists all active
canonical albums, using the release's finished JPEG where the selected asset still
matches and the latest selected candidate for later repairs. It uses the original
standard finish. It does not query TV thumbnails or launch generation/deployment.

Click the artwork to add up to 30 numbered problem markers. Click a marker to remove
it; describe the numbered issues in the notes, select optional tags, and choose
Looks good (A), Needs fix (F), or Later (S). Decisions save to SQLite and advance.
Search, status filters, a full-size preview, and Undo last save support revisiting
decisions. Saved progress survives reloads; unsaved edits prompt before navigation.

`GET /api/manual-review/export` returns the flagged queue with album IDs, selected
assets, notes, issue tags, and normalized `[x,y]` marker coordinates (top-left origin).
The queue is stored in `manual_reviews`, with history in `manual_review_events`.
Review revisions protect concurrent edits; a different selected asset returns an
album to unreviewed while retaining prior guidance. For each queued repair, preserve
the source cover, repair the surrounding background, composite the exact resized
source back over it, verify cover pixels in the lossless output, and show the user
the preview before deploying. Record approved replacements as new candidates so the
portal displays the new version. Keep all prior layers and deployment journals.

From the repository root:

```sh
.venv/bin/python tools/frame_art_review/server.py
```

Open http://127.0.0.1:8766/. Data is stored in `/Users/jsands/Documents/Frame Art Review` by default (`~/Documents/Frame Art Review` on another account). Set `FRAME_REVIEW_ROOT` to choose another directory. Keep it outside the repository, and back it up along with the images.

## Data layout

- `snapshot/music/{source,background,widescreen,widescreen-compressed}`: original HA files, unchanged.
- `snapshot/music/{manifest,index}.json` and `snapshot/share/`: original metadata and associations.
- `snapshot/inventory.json`: remote file sizes, timestamps, and SHA-256 checksums.
- `recovered/source/`: covers recovered by recorded Apple collection ID, or explicitly labeled crops requiring user confirmation.
- `review.sqlite3`: albums, generation attempts, batches, decisions, source approvals, and append-only review events.
- `candidates/<attempt>/source`, `background`, `widescreen`, `widescreen-compressed`: separate candidate assets; PNG/JPEG rendering variants can be produced from the same raw background.
- `candidates/<attempt>/{recipe,response}.json`: exact request, source checksum, model, geometry, response usage, request ID, and estimated cost.
- `recipe/`: frozen copy of the existing cover-art module used by this review run.
- `backups/`: consistent SQLite backups. The UI can also export all structured records as JSON.

The existing add-on compression function enforces its 4 MiB JPEG limit. Changing a shadow preview renders locally and never calls OpenAI. Acceptance pins a generated candidate; the final global shadow choice and production export are later steps.

The current pilot uses `gpt-image-2.5-flare` with the frozen add-on's original `REFERENCE_BACKGROUND_PROMPT`, its reference canvas, and no mask. Final compositing has no edge feathering and defaults to the original regular shadow. The previous masked pilot was stopped; its completed images and failed/canceled attempts remain in history. The second batch compares exactly the same 25 albums, without adding previous rejection feedback to the prompt. A separate, user-requested single-album batch tests Pink Floyd with `gpt-image-2`; its model is recorded per batch and candidate and shown in the viewer. The historical HA manifest records the existing Pink Floyd image as `gpt-image-1.5`, generated February 10, 2026, with no mask and a 1024-pixel reference cover. Today’s tests use the current add-on reference geometry, keeping it fixed for model comparisons. Those pilot attempts remain unchanged in history.

## Review

The viewer now opens on all albums. The pilot filter still shows the selected 25 albums; Generation failures shows terminal API errors or uncertain requests, and Used a fallback shows successful later stages. Compare a completed candidate with its current saved artwork, optionally preview a shadow, then Accept, Reject with reasons, or Review later. Notes save with the decision. Clear decision lets you reconsider; older candidates remain selectable. Conflicting saves from stale tabs fail rather than overwriting newer decisions.

The user-authorized full run includes seven provisional recovered crops without marking them confirmed. They remain flagged in Needs source cover; ordinary reviewer retries require confirmation first. They may have less detail than original downloads. One mismatching duplicate alias requires particular care; the source note explains it. Exact-ID downloads are not replaced with a different edition through fuzzy search.

Regenerate rejected queues up to 25 albums. For the full-run pipeline it advances each rejected album to the next fallback stage, retaining its review guidance. Each API rejection can then advance through the remaining stages. Albums that exhausted all stages require inspection before another run. It excludes accepted albums and requires the current candidate to be rejected. Correct source-cover problems before retrying them. Queued/running or uncertain interrupted requests cannot be charged again implicitly. Pause stops after the current request; resume continues the saved queue. For fallback batches, definitive HTTP 400 errors advance to the next model/prompt stage; timeouts, connection failures, HTTP 5xx, and failures after API success remain uncertain and do not advance. HTTP 429 uses at most three attempts with bounded waits; authentication, exhausted rate limits, disk errors, or the spend guard pause the batch.

Each generation run has a $50 estimated-spend allowance. Clicking Resume on a paused batch starts a fresh allowance; repeated Resume requests while already running do not reset it. A separate $200 lifetime estimated-spend limit spans all batches and cannot be reset by Resume. Recorded costs and historical estimates remain intact. Each request is assigned to its run when claimed, so queued fallback requests count against the run that actually starts them. Both limits include reservations for active requests and conservative allowances for missing usage; the viewer shows recorded costs, the current run guard, and the lifetime guard separately. Each active request reserves $0.50. Failed requests with missing usage reserve $0.10 and uncertain requests reserve $0.50 for the spend guard. This is a conservative scheduling guard, not an API-enforced billing cap. Cost uses returned usage and the recorded pricing rates. The first batch contains exactly 25 attempts, with no automatic retries.

## Full-library run

Batch 4 contains all 660 catalog entries, with two concurrent workers. Its persisted plan is: masked Flare, masked GPT Image 2, masked GPT Image 1.5, original-prompt GPT Image 2, original-prompt GPT Image 1.5. Each album stops at its first successfully saved candidate for human review. Source, raw output, generated background, 4K PNG, compressed JPEG, exact prompt/model, costs, errors, and fallback stage remain separate. A successful API response is not a quality approval. Regular shadow is the initial preview; shadows remain locally adjustable. Masked/water composites retain the tested 24-pixel feathering; legacy composites use zero feathering.

Nevermind uses a separately inspected, water-only color reference sampled from the outermost source edges. Only that reference is sent to the generation API; the original cover is composited locally. Its route tries the three models with a water-background prompt and never falls back to sending the original cover. Source/reference checksums and sampling provenance are recorded.

The queue and every transition are durable in SQLite. A process lock prevents duplicate workers, and transactional claims prevent duplicate requests between worker threads. A stopped/restarted worker marks uncertain in-flight requests for inspection instead of silently charging again. `full-run-plan.json` records the original scope; SQLite `spend_runs` and `spend_settings` record the current run allowances and lifetime limit. The viewer reports album successes/failures separately from attempt counts. Download failures and source issues remain visible.

## Maintenance commands

```sh
.venv/bin/python tools/frame_art_review/library.py report
.venv/bin/python tools/frame_art_review/library.py verify
.venv/bin/python tools/frame_art_review/library.py recover
.venv/bin/python tools/frame_art_review/library.py recover --country gb
.venv/bin/python tools/frame_art_review/library.py recover-centers
.venv/bin/python -m unittest discover -s tools/frame_art_review -p 'test_*.py' -v
```

`library.py import` imports an existing local snapshot without overwriting existing review records. `library.py pilot` selects and queues a single representative pilot, and refuses to select it twice. `pipeline.py <batch-id>` runs a saved batch. Never reset an interrupted request to queued without checking its response files and provider outcome: the request may already have been billed. Saved raw output can be rendered without another API call.

The server binds only to loopback, validates the Host and Origin of mutations, requires JSON plus a custom request header, and serves only image files from allowed data directories. The add-on key is retrieved over existing HA SSH into worker memory and never saved or sent to the browser. Optional feature-detected WebMCP read/review tools mirror the same API; browser registration is unverified where WebMCP is unavailable. No generation tools are exposed through WebMCP.

TV replacement, rewriting catalogs, new production options, and changes to the live pipeline are deliberately not implemented by this review tool.

## Curation after the first pass

The home page compares two finished versions, defaulting to the saved original/kept artwork and the newest completed generation. Its Second-pass results filter and live progress bar surface the ongoing run. Both version menus include older passes. `/review` retains the generation status, candidate reviews, and rejection history. The original comparison serves the exact saved JPEG; its baked-in shadow/edges are preserved. One persistent global shadow/feather setting (off, 12 px, 24 px) applies to every generated version and to the reconciliation export, including previously kept candidates. Previews render locally from the immutable source/background pair. **Keep this version** chooses the artwork; styling is controlled globally. Original finished JPEGs retain their existing appearance. Changing these settings does not generate images or alter the running generation requests.

- **Possible duplicates** groups normalized artist/title matches, including aliases and edition suffixes. These are suggestions, not automatic deletions. Select a catalog entry and a comparison pane's artwork, then merge locally; all assets and review records remain. **Keep as separate releases** dismisses a suggestion. **Unmerge entries** removes aliases and preserves current artwork choices.
- **Compare original first** highlights entries whose written notes mention preferring original/previous artwork. Notes and reasons remain visible beside the comparisons and can be edited into guidance for a new attempt.
- **Fix source cover** includes wrong-cover rejections and provisional recovered covers. Inspect existing source options, recover a crop from the saved original, or upload a replacement. Explicitly select/confirm the correct source; old generations remain intact.
- **Second-pass draft** stores selected albums, recipes, source hashes, and guidance without API calls. Clear a kept choice before staging that album. Resolve duplicates and wrong covers before staging. **Generate next draft batch** explicitly launches up to 25 staged albums with the existing $50/run and $200/lifetime estimated guards. The water-only Nevermind reference route is retained.
- **Export reconciliation plan** produces a local canonical-entry/alias map, chosen artwork/settings, original metadata and source checksums. It does not apply changes to HA manifests or rebuild the TV cache. Those are later steps after the choices are complete.

Preparation is idempotent and adds suggestions/options only:

```sh
PYTHONPATH=tools/frame_art_review .venv/bin/python -c 'import store,curation; store.backup(); store.init(); curation.discover_duplicates(); curation.prepare_sources()'
```

Checks: `.venv/bin/python -m unittest discover -s tools/frame_art_review -p 'test_*.py'` (temporary databases and mocked generation requests; no API spend).

## Full manual review

The home page is now a full-library workbench; `/curation` retains the earlier two-version tool and `/review` retains generation/rejection history. All finished original and generated versions appear in a two-column gallery, including versions from merged entries. The saved global finish remains shared across every generated preview.

Every canonical record begins **To review**, even when artwork was previously kept. **Confirm kept & next** or **Use this & next** confirms it for this full review. Queuing regeneration, flagging an album fix/duplicate, or choosing Later records that task and advances. Queued regeneration removes the current pin so the worker can make a new version; all old artwork and review history remain, and Undo restores the earlier pin and queue state. Generation still requires the separate launch button.

The editor supports artist/title corrections, source-cover recovery/upload/selection, and an optional immediate staging action. Cache keys and snapshot metadata stay intact. Manual deduplication allows searching another canonical record, inspecting all versions from both, then choosing which image to retain under the target record. Aliases are flattened, and no files are deleted.

`full_review` stores review status independently of old acceptance decisions. Album edits or a new generation invalidate stale confirmation. `workflow_events` records before/after snapshots; Undo is permitted only while the affected records, generation state, and (for merges) duplicate groups still match. Exports include full-review status and remaining records. No workflow endpoint writes to HA or the TV.

Keyboard shortcuts outside form fields: Enter confirms the kept image, 1–9 chooses a loaded version, R stages regeneration, F flags an album fix, D opens deduplication, S defers, U undoes, and arrow keys navigate. Notes/reasons and the last record are retained locally across refreshes. The queue filters distinguish To review, Confirmed, Regeneration, Fixes, Duplicates, Later, and latest-generation failures.

## Targeted repair pilot

`repairs.py` stages an explicit subset of up to six records and leaves other drafts alone. It requires current revisions, an unkept regeneration decision, a verified source, a completed latest request, and resolved duplicates. API refusals/uncertain requests and the separate water-reference route cannot enter this scene-repair route. Frozen references include the existing widescreen scene, an opaque original center at the established API coordinates, source/base checksums, and a specific repair instruction. Each request uses Sunburst high with one stage and the normal spend guards. There is no automatic fallback after refusal. Source, background, final image and review history remain separate.

A deterministic solid-color repair creates a zero-cost candidate through the same local compositor and global finish. The viewer labels these versions `Repair · local color match` and `Repair · Sunburst scene edit`.

`workflow.correct_catalog` applies a verified source option to a record flagged for a fix, updates its local collection ID, and preserves the kept finished original. The change is journaled and undoable. The record reopens for review. Reconciliation exports include the corrected collection ID alongside the unchanged cache key and original snapshot metadata, so a later migration can reconcile associations deliberately.

## Independent background alternatives

`repairs.queue_backgrounds` freezes explicit, album-specific scenery/design prompts for an authorized selection of unkept regeneration records. The worker uses the Image API generations endpoint with a JSON text prompt and uploads **no cover or mask**. This makes a separate decorative background rather than another outpainting of the cover. The original cover stays in the local source folder and is composited afterward with the global shadow/feather settings. The worker records `api_input: text_only`, the prompt, model, source checksum, raw response, background, final images and usage. Source changes since staging stop before an API request.

The route has one Sunburst high stage, the existing $50/run and $200/lifetime guards, no automatic model fallback, and no automatic acceptance. A previous refusal can receive a distinct scenery-only design without resubmitting its refused image input. Any newly refused or uncertain request stays for inspection. Local solid-color alternatives cost zero. Confirmed records and prior image files remain protected. The viewer labels generated alternatives `Background alternative · Sunburst`.

### Final manual-fix comparison

Run `.venv/bin/python tools/frame_art_review/final_comparison.py` and open
`http://127.0.0.1:8767/`. The frozen 23-album comparison manifest lives under
`reports/final-23-comparison/manifest.json` in the review data directory. It pairs
pre-upgrade snapshot images with approved manual repairs (or current released
images where generation was rejected). Rejections are highlighted. Choices are
saved atomically in `selections.json` with exact asset paths and SHA-256 hashes;
stale tabs cannot overwrite newer choices. This page does not deploy or modify
the library. Verify the chosen hashes and retain the source/recipe when preparing
the eventual deployment.
