# Repeatable library upgrades

Status: implementation plan, recorded during the September 2026 migration. The current migration remains in progress; this document does not certify completion. Proposed commands and features below are not implemented unless explicitly listed as existing.

## Intended next-upgrade workflow

1. Create a named upgrade run from a fresh HA snapshot and TV inventory. Preserve the previous release and all review history.
2. Select a versioned generation recipe and budget. Run a small representative pilot, including prior problem cases, before authorizing the full library.
3. Generate candidates without changing active HA media or TV artwork. Reuse existing source covers and retain every attempt, raw response, background, and finished image.
4. Review current artwork alongside all candidates. Keep, choose, fix source, resolve duplicates, or queue regeneration with notes; advance automatically. Apply one global finish without another paid generation.
5. Freeze the approved release and preview its exact changes: logical albums, unique TV images, kept IDs, uploads, retirements, disk use, and estimated duration. Shared artwork can retain separate album identities.
6. Stage and verify the files, then explicitly start the HA-owned migration. It continues without the laptop. Delete and replace serially, save returned IDs, reconnect after rests, and preserve unrelated TV art.
7. Verify final catalog/index/manifest/alias consistency and actual TV inventory. Save a completion report and promote the approved recipe as the production default. Keep an earlier recipe available for future selection.

## What already exists

- Durable SQLite candidates, reviews, rejection notes, source corrections, duplicate decisions, workflow history, and undo.
- Separate source, raw generation, background, widescreen PNG, and compressed JPEG assets; locally rendered finish variants.
- Bounded generation workers, estimated spend reservations, per-run and lifetime guards, and recorded request/model/usage metadata.
- Frozen releases with file checksums, verified HA backups, paced delete-first TV replacement, an acknowledged-ID journal, and deterministic metadata reconciliation.
- TV connection refresh after long rests and bounded reconnection for read-only requests. Mutations with uncertain acknowledgments are not blindly retried.
- Production continuation and legacy modes, with the approved original finish and an applicable model fallback chain.

## Work to complete before calling this reusable

### 1. Replace one-run assumptions with a run configuration

`build_release.py`, `stage_and_start.py`, `migration_status.py`, and `ha_connection.py` currently contain run-specific names, paths, a host address, or an exact add-on version. Introduce a versioned run document containing run ID, local data root, HA SSH destination, remote release/run paths, recipe ID, budgets, and pacing. Keep credentials outside it. Check migration schema/capability compatibility rather than one literal add-on version.

Provide one documented entry point for create/snapshot, pilot, generate, review, plan, stage, deploy, status, pause, resume, amend, and restore. These are proposed operations, not existing CLI commands. Staging a release with an existing journal must never reset the migration.

### 2. Share the generation recipe between review and production

The review runner and `seamless.py` currently implement overlapping model, prompt, fallback, geometry, and finishing behavior. Extract a shared recipe/renderer with explicit versioned inputs. Pin each historical run to its original recipe; do not rewrite past attempts when defaults change.

A recipe should declare model preference, quality, prompt, reference/mask geometry, output format, finishing settings, fallback eligibility, and pricing provenance. Adding a future model should normally be configuration plus a pilot; new API capabilities may still require an adapter change. Verify model availability and pricing at that time rather than assuming a future model name or price.

Transport uncertainty, explicit refusals, unsupported-model errors, and visually rejected results are different outcomes. A successful API response is not acceptance. Preserve these distinctions and do not automatically replay refused content across models.

### 3. Make post-freeze amendments a supported operation

This run required audited manual amendments to retain two originals and share approved Dark Side of the Moon artwork between editions. Implement an amendment transaction that pauses at a safe boundary, previews affected records, saves the old release/journal, verifies completed work, stages changed assets, updates the release hash and review choice together, and resumes.

Cover both untouched records and already-uploaded artwork. Preserve all old layers and decisions. Do not clear an uncertain upload merely to let a run continue. If a pause happens before an upload is sent, represent that as prepared—not `upload_started`. Reserve the uncertain state for requests that may actually have been transmitted.

### 4. Make retention and progress understandable

Keep logical album identity separate from source identity, selected artwork identity, and TV content ID. Multiple editions may deliberately share one finished image and TV ID without merging their music metadata.

Catalog hashes can be stale. Distinguish a metadata discrepancy from a proven file difference or missing TV ID. Reconcile with snapshot checksums, provenance, inventory, and explicit review decisions; do not claim a TV image differs merely because a catalog hash differs.

Show separate counters for records reconciled, uploads completed/remaining, existing images retained, duplicate IDs retired, and shared artwork. Report preflight, intentional rest, reconnect, paused, and complete distinctly. Keep thumbnail downloads out of routine migration verification; use acknowledged IDs and inventory.

### 5. Finish recovery, retention, and packaging

Package the local reviewer and HA worker as maintained parts of this repository. The HA status endpoint should eventually be accessible without the local reviewer; the migration worker itself already runs independently of the laptop.

Add a documented restore operation using the verified backup. Restored uploads receive new TV IDs, so restoring a JSON database alone is insufficient. Preserve sources, backgrounds, finished files, recipes, reviews, aliases, old/new ID mappings, and generation metadata. Keep secrets out of exported archives and require explicit selection before pruning historical runs.

## Verification and rollout

First finish this migration and user testing. Produce a baseline release report with checksums, actual inventory, selected recipe, amendments, counts, errors, and backup locations. Do not refactor the live worker during that final validation.

Then implement configuration and shared recipes, followed by the amendment/recovery interface. Exercise them against a saved fixture and simulated TV before another small live canary. Regression scenarios must include:

- Connection loss after a rest, read retry exhaustion, and restart between every migration state.
- Pause before upload invocation versus loss of acknowledgment after invocation; neither may create duplicate paid requests or duplicate TV uploads.
- Completed uploads skipped on resume; retained/shared IDs never retired; unrelated art preserved.
- Stale catalog hashes, edition aliases, corrected source IDs, and explicit keep decisions.
- Repeated staging, interrupted amendments, changed release hashes, missing layers, and failed backup verification.
- Finish parity between review and production; rejected/uncertain generation requests and spend-guard resume behavior.

Completion criterion: start a second named run from a fresh snapshot using the documented interface, pilot a different configured recipe, review and amend it, deploy/resume/restore it in simulation, and produce its audit report without editing Python constants, database rows, or journal hashes by hand.
