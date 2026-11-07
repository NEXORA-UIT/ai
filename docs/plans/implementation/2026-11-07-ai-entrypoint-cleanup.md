# AI Entrypoint Cleanup Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development or superpowers:executing-plans.

**Goal:** Make the existing preview and operational entrypoints easier to navigate and verify.

**Architecture:** Keep the public module paths. Preview owns arguments and orchestration, rendering owns console and JSON, and the ingestion pipeline owns embedding metadata construction.

**Tech Stack:** Python 3.12, argparse, pytest, Docker Compose.

**Spec:** [AI maintainability review](../../refactor/ai-maintainability-review.md)

## Global Constraints

- Current AI feature branch only. Preserve preview flags, formats and exit codes.
- Preserve model revision, embedding fingerprint, Board scope and the existing HTTP contracts.
- Use only disposable services and a separate database for live integration tests.

## Review Focus

- Preview never needs private model internals.
- JSON omits complete vectors by default and includes them only when requested.
- CLI help and retrieval exports still work when models are unavailable.
- Documentation distinguishes the AI demo from the planned queue integration.
- Offline and isolated live tests are reported separately.

## Tasks

- [x] Cover preview model loading and JSON output. The old CLI called the private model method; the new CLI uses the public model lifecycle and shares embedding metadata with ingestion. Focused ingestion and embedding run: 117 passed, 16 deselected.
- [x] Replace file inventory assertions with CLI and Docker fixture contracts. Preserve the existing import tests. Update README, architecture and local demo environment examples.
- [x] Verify CLI entrypoints, Ruff, Compose syntax and diff checks. Final offline suite: 390 passed, 31 deselected.
- [x] Build the demo image from the lockfile and check the CLI help, OpenAPI routes, system prompt, and SQL fixture inside an isolated container.
- [x] Run the live suite against the disposable Compose project. First run identified the required seeder database name and an outdated test double. Final run: 30 passed, 383 deselected, with the live DeepSeek test excluded. The real smoke covers Tika, BGE-M3, pgvector, CPU reranking, citation source fields and isolation from another Board. The test project was removed afterward.
