# Agent Notes for Frame Art Add-on Repo

This repository contains the production Home Assistant add-on and a separate local artwork-review/migration tool. The add-on controls Samsung Frame TVs and can invoke paid OpenAI requests; keep runtime state, generated assets, and review data distinct from source code.

## Quick commands (run from repo root)

- **Set up local Python:** `make setup` (creates `.venv` and installs the add-on dependencies).
- **Run add-on tests:** `make test` (`unittest` discovery under `tests/`).
- **Run review-tool tests:** `.venv/bin/python -m unittest discover -s tools/frame_art_review -p 'test_*.py' -v`.
- **Run the local review portal:** `.venv/bin/python tools/frame_art_review/server.py` (normally at `http://127.0.0.1:8766/`).
- **Refresh Home Assistant context (only when needed):** `make ha-context`.

## Project structure

- `frame_art_uploader_ai/` — production add-on: `run.sh` supervises the worker; `uploader.py` owns queues, TV operations, catalogs, and status; `cover_art.py` and `seamless.py` handle artwork lookup/generation; `migration.py` handles explicitly released library migrations.
- `frame_art_uploader_ai/config.yaml` — add-on metadata, options, and schema. Keep user-facing option changes synchronized with the code, schema, and `translations/en.yaml`.
- `frame_art_uploader_ai/README.md` — runtime behavior, queue/status contracts, and migration guidance.
- `tests/` — focused add-on unit tests, generally using mocks and temporary paths.
- `tools/frame_art_review/` — local review UI, generation pipeline, SQLite store, release/staging helpers, and review-tool tests. Read its `README.md` and `UPGRADE_WORKFLOW.md` before changing deployment behavior; the latter distinguishes existing features from proposals.
- `Makefile` — local environment and test entry points.

## Working rules

- Preserve existing `/data`, `/share`, and `/media` contracts, queue/status JSON shapes, catalog/recipe metadata, and migration recovery behavior unless the request explicitly changes them.
- Follow the surrounding Python style; there is no configured formatter or linter. Add or update focused `unittest` coverage for behavior changes and mock external services rather than contacting them.
- Keep `config.yaml` and the runtime add-on version (`ADDON_VERSION`) aligned for releases, and document new user-facing behavior or options.
- Keep review databases, images, reports, and other local runtime data outside the repository. Check `.gitignore` and `git status` before creating local artifacts or committing changes. Do not edit `.venv`, caches, or generated/vendor files.
- Treat model/prompt pipelines, the pinned `samsungtvws` dependency, and migration journals as compatibility-sensitive; do not silently refactor or reset them.

## Boundaries

- ✅ **Always:** read the relevant README/tests first, use local mocks for validation, and verify behavior without assuming a successful API or TV operation was idempotent.
- ⚠️ **Ask first:** before live TV changes, release staging/deployment, migration or catalog cleanup, paid image generation/web search, changing model/pipeline settings, or destructive file operations.
- 🚫 **Never:** commit secrets or local review data, blindly retry an uncertain generation/upload request, or treat commands described as proposed in `UPGRADE_WORKFLOW.md` as implemented.
