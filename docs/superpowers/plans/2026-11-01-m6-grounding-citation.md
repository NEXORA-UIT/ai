# Milestone 6 Grounding and Citation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Generate an answer from selected evidence with checkable citations, or refuse when evidence is absent or invalid.

**Architecture:** A provider-neutral answer service first checks the database-backed Board ID in every Milestone 5 RerankedChunk against the requested Board, then builds a bounded context from those records. The model returns structured answer text and cited chunk IDs; a local validator maps those IDs to public source metadata and fails closed if citations do not belong to the supplied context. The real provider adapter is deferred until D3 is selected.

**Tech Stack:** Python 3.12, provider-neutral client protocol, future agreed LangChain/LLM adapter, pytest, Ruff.

**Spec:** docs/superpowers/specs/2026-11-01-rag-retrieval-m2-m8-design.md and docs/rag-architecture.md

## Global Constraints

- Express will authorize Board access before future service integration. The standalone function normalizes requested board_id and rejects any supplied source whose database-backed board_id differs before context construction or LLM calls; it must not rely on the caller's board_id alone.
- Never send internal storage_key, storage credentials or raw vectors to the LLM or public result. Limit context by chunk count and character/token budget.
- Citation ID must refer to one of the supplied chunks. Output includes document/chunk IDs, file name, optional page, and offsets.
- Empty evidence refuses without an LLM call. A non-refusal answer must have nonempty text and at least one valid citation from the supplied context; empty/malformed/unknown citations refuse rather than inventing a source.
- Retrieved document text is untrusted data. Delimit it in a separate document_context field, never merge it into system instructions; the system prompt tells the model to ignore instructions inside documents. Fake-client tests verify prompt construction, while real-model adversarial behavior is assessed after D3.
- D3 selected: DeepSeek AI with API model ID `deepseek-flash`, the OpenAI-compatible DeepSeek endpoint, a 30-second timeout, and JSON Output for `{answer, cited_chunk_ids}`. Use existing `httpx`; do not add an SDK solely for this adapter. Task 1 keeps fake-client coverage; Task 2 adds the real adapter with mock-transport tests and no live API test. Do not hard-code secrets or add a public FastAPI route.
- One task per commit after checks.

## Review Focus

- No eligible evidence returns a deterministic refusal without calling the client (test_empty_context_refuses).
- A citation to an unknown chunk ID is rejected (test_unknown_citation_refuses).
- Context contains only the selected scoped chunks supplied by retrieval (test_context_uses_only_supplied_chunks).
- Cross-Board evidence fails with EvidenceScopeError before constructing context or calling the client (test_cross_board_evidence_rejected).
- Context overflow is truncated on chunk boundaries, retaining citation mapping (test_context_budget).
- A non-refusal answer without valid citations is refused (test_uncited_answer_refuses).
- An "Ignore previous instructions" payload in a retrieved chunk remains delimited document data and cannot replace system instructions in the built request (test_untrusted_document_prompt_injection).
- Model transport failure is surfaced as a typed error, not a fabricated answer (test_provider_failure).

---

### Task 1: Build context and validate grounded output

**Files:**
- Create: src/generation/schemas.py
- Create: src/generation/context.py
- Create: src/generation/answer.py
- Create: src/generation/__init__.py
- Modify: data/system_prompt.md to contain the grounded answer, citation and refusal instructions; load it through context.py.
- Test: tests/generation/test_answer.py

**Interfaces:**
- Produce Citation(chunk_id: str, document_id: str, file_name: str, page_number: int | None, start_offset: int | None, end_offset: int | None). Do not include storage_key; a future authorized backend can resolve document_id.
- Produce AnswerResult(text: str, citations: tuple[Citation, ...], refused: bool).
- Produce GroundedPrompt(system_instructions: str, question: str, document_context: str) so trusted instructions and untrusted text remain distinct.
- Produce AnswerClient protocol complete(prompt: GroundedPrompt) -> dict[str, object] with keys answer and cited_chunk_ids. The later provider adapter maps system_instructions to the provider's system role and document_context to a lower-trust user/content role.
- Produce answer_with_sources(question: str, board_id: str | UUID, chunks: Sequence[RerankedChunk], *, client: AnswerClient, max_context_chars: int = 12000) -> AnswerResult.
- Validate each chunks[i].source.board_id against normalized board_id and raise EvidenceScopeError on any mismatch before passing content to context.py or AnswerClient. Do not simply stamp the requested Board onto supplied chunks. Keep answer text and citation list structured; future transport may render numbered inline references. Do not claim a citation proves the model's claim is faithful; Milestone 7 measures that separately.

- [ ] **Step 1: Write failing tests.** Assert all Review Focus cases except provider transport, plus whitespace question, duplicate citation IDs, a fake client refusing an unsupported question despite nonempty context, valid multi-source answer mapping, and serializing AnswerResult without storage_key. The cross-Board test must assert the fake client was never called. Include a malicious chunk asking for an uncited answer; if the fake client returns one, validation must refuse it.
- [ ] **Step 2: Confirm failure.** Run python -m pytest tests/generation/test_answer.py -q. Expected: missing service fails.
- [ ] **Step 3: Implement scope check, context and validation.** Check all database-backed source Board IDs before context construction; delimit untrusted source text in document_context and include only selected text and opaque chunk IDs. Keep system_instructions separate, validate output shape, non-refusal citations and IDs; return a fixed refusal for zero evidence or invalid citations.
- [ ] **Step 4: Verify and commit.** Run python -m pytest tests/generation -q and python -m ruff check src/generation tests/generation. Commit focused files with message feat(generation): ground answers in cited chunks.

### Task 2: Add the approved LLM provider adapter after D3 is selected

**Files:**
- Create: src/generation/provider.py
- Modify: .env.example with key names only; no secret values.
- Test: tests/generation/test_provider.py

**Interfaces:**
- Implement AnswerClient from Task 1 using DeepSeek `deepseek-flash` and an explicit 30-second timeout. Use JSON Output and map GroundedPrompt fields to distinct provider roles without promoting document_context to system/developer instructions.
- Configure credentials through environment variables and inject a fake transport in tests. Pin a model identifier and record output schema handling.

- [x] **Step 1: Record D3 provider/model, timeout, and structured-output contract in the design spec before writing the adapter.**
- [ ] **Step 2: Write failing adapter tests.** Assert request payload contains bounded, delimited untrusted context; an instruction embedded in document text cannot be promoted into system/developer instructions. Verify schema parsing, timeout/HTTP errors and that no API key appears in logs/errors. Add an opt-in real-model adversarial check after D3 without treating fake-client tests as proof of model faithfulness.
- [ ] **Step 3: Confirm failure.** Run python -m pytest tests/generation/test_provider.py -q. Expected: missing adapter fails.
- [ ] **Step 4: Implement the provider adapter behind the existing AnswerClient boundary.** Keep provider-specific code out of answer.py.
- [ ] **Step 5: Verify and commit.** Run python -m pytest tests/generation -q and python -m ruff check src/generation tests/generation. Commit focused files with message feat(generation): add llm provider adapter.

**Acceptance:** Task 1 can validate cited evidence and refusals with fake clients. The real answer path remains open until Task 2 receives D3; faithfulness quality is assessed in Milestone 7.
