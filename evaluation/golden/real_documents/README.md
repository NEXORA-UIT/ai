# Real-document evaluation corpus

This directory is intentionally a placeholder until the user chooses a source
for a small, non-sensitive PDF/DOCX corpus. Do not put production documents,
private user data, generated text, or unreviewed labels here.

When a source is selected, check in a versioned manifest with source URLs,
license and provenance notes, SHA-256 checksums, a frozen chunk snapshot, and
manually reviewed query labels. Run it only through the guarded evaluation
seeder against `TEST_DATABASE_URL`. Synthetic fixtures under `../synthetic/`
are correctness tests and cannot be reported as real-document quality results.
