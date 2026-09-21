// Phase 8 (aq_017) — DEEPENING last-resort fallback fix.
//
// Root cause: "Can you elaborate on the memory cost tradeoff?" (8 words) matched
// IB_DEEPENING_RE but was rejected by isBareFollowUp() on the 5-word cap.
// With claims.size=0, the last-resort at line 1123 of turn-classifier.ts fired and
// added DOCUMENT_FACT/PRIVATE_SOURCE_REQUIRED, causing noEvidenceNotice to inject
// an evidence disclaimer into the prompt → voice score 3/5.
//
// Fix: inside the last-resort, `isDeepening = IB_DEEPENING_RE.test(resolvedQuestion)`
// routes to GENERAL_TECHNICAL instead of DOCUMENT_FACT for DEEPENING questions.
//
// Tests:
//   A: aq_017 exact question — no PRIVATE_SOURCE_REQUIRED, no noEvidenceNotice,
//      correct intent (follow_up_generic) and strategy (deepening_elaboration).
//   B: a longer DEEPENING question (>5 words, different DEEPENING phrase) — same.
//   C: non-DEEPENING document-dependent question — DOCUMENT_FACT path still intact.
//   D: existing bare follow-up (≤5 words) — FOLLOW_UP behavior unchanged.

import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const base = path.resolve(process.cwd(), 'dist-electron/electron/context-intelligence');
const { decide }        = await import(pathToFileURL(path.join(base, 'orchestration/orchestrator.js')).href);
const { composePrompt } = await import(pathToFileURL(path.join(base, 'generation/prompt-composer.js')).href);
const { MODE_POLICIES } = await import(pathToFileURL(path.join(base, 'policies/mode-policy-registry.js')).href);

const POLICY     = MODE_POLICIES['technical-interview'];
const EVAL_SCOPE = { userId: 'eval', modeId: 'technical-interview' };

// Markers produced by noEvidenceNotice (from prompt-composer.ts).
// Any of these in a composed prompt means the notice fired.
const NO_EVIDENCE_MARKERS = [
  'No supporting evidence was retrieved',
  'has NO reference material attached',
  'This question requires a source the active mode does not authorize',
];

function hasNoEvidenceNotice(composed) {
  const full = (composed.system ?? '') + ' ' + (composed.user ?? '');
  return NO_EVIDENCE_MARKERS.some((m) => full.includes(m));
}

let _seq = 0;
const mkDecide = (question, extra = {}) => decide({
  requestId:       `deep-fallback-${++_seq}`,
  requestSequence: 1,
  surface:         'manual_chat',
  modeId:          'technical-interview',
  scope:           EVAL_SCOPE,
  sessionId:       `deep-fallback-sid-${_seq}`,
  manualQuestion:  question,
  ...extra,
});

// ── Test A: aq_017 exact question ────────────────────────────────────────────

describe('A: aq_017 exact question — DEEPENING must not produce PRIVATE_SOURCE_REQUIRED', () => {
  const Q = 'Can you elaborate on the memory cost tradeoff?';
  const d = mkDecide(Q);
  const composed = composePrompt({ decision: d, policy: POLICY, evidence: [] });

  test('A1: interviewerBehavior is DEEPENING', () => {
    assert.equal(
      d.interviewIntent?.interviewerBehavior, 'DEEPENING',
      `expected DEEPENING; got ${d.interviewIntent?.interviewerBehavior}`,
    );
  });

  test('A2: intent is follow_up_generic', () => {
    assert.equal(
      d.interviewIntent?.intent, 'follow_up_generic',
      `expected follow_up_generic; got ${d.interviewIntent?.intent}`,
    );
  });

  test('A3: strategy is deepen_explanation (DEEPENING strategy)', () => {
    assert.equal(
      d.answerStrategy?.id, 'deepen_explanation',
      `expected deepen_explanation; got ${d.answerStrategy?.id}`,
    );
  });

  test('A4: claimRequirements does NOT include PRIVATE_SOURCE_REQUIRED', () => {
    const hasPrivate = (d.claimRequirements ?? []).some(
      (c) => c.authority === 'PRIVATE_SOURCE_REQUIRED',
    );
    assert.ok(
      !hasPrivate,
      `DEEPENING question must not produce PRIVATE_SOURCE_REQUIRED — got: ${JSON.stringify(d.claimRequirements)}`,
    );
  });

  test('A5: composePrompt does NOT fire noEvidenceNotice (even with no evidence)', () => {
    assert.ok(
      !hasNoEvidenceNotice(composed),
      `noEvidenceNotice must NOT fire for DEEPENING question — found notice text in composed prompt`,
    );
  });
});

// ── Test B: longer DEEPENING question, different phrase ───────────────────────

describe('B: longer DEEPENING question (>5 words, "tell me more" phrase) — still no PRIVATE_SOURCE_REQUIRED', () => {
  const Q = 'Tell me more about the eviction policy you mentioned.';
  const d = mkDecide(Q);

  test('B1: interviewerBehavior is DEEPENING', () => {
    assert.equal(
      d.interviewIntent?.interviewerBehavior, 'DEEPENING',
      `expected DEEPENING; got ${d.interviewIntent?.interviewerBehavior}`,
    );
  });

  test('B2: claimRequirements does NOT include PRIVATE_SOURCE_REQUIRED', () => {
    const hasPrivate = (d.claimRequirements ?? []).some(
      (c) => c.authority === 'PRIVATE_SOURCE_REQUIRED',
    );
    assert.ok(
      !hasPrivate,
      `Longer DEEPENING question must not produce PRIVATE_SOURCE_REQUIRED — got: ${JSON.stringify(d.claimRequirements)}`,
    );
  });

  test('B3: composePrompt does NOT fire noEvidenceNotice', () => {
    const composed = composePrompt({ decision: d, policy: POLICY, evidence: [] });
    assert.ok(
      !hasNoEvidenceNotice(composed),
      `noEvidenceNotice must NOT fire for DEEPENING question`,
    );
  });
});

// ── Test C: non-DEEPENING document-dependent question — DOCUMENT_FACT intact ──

describe('C: non-DEEPENING document question — DOCUMENT_FACT path still intact', () => {
  // "How does the heartbeat failure get detected?" — cited in the turn-classifier
  // comment at line 1111 as the canonical example the last-resort was added for.
  // 8 words, not DEEPENING, not techTask, claims.size=0 → last-resort fires →
  // d01Blocked=false, howWouldYouGeneral=false, isDeepening=false → DOCUMENT_FACT.
  const Q = 'How does the heartbeat failure get detected?';
  const d = mkDecide(Q);

  test('C1: claimRequirements DOES include PRIVATE_SOURCE_REQUIRED', () => {
    const hasPrivate = (d.claimRequirements ?? []).some(
      (c) => c.authority === 'PRIVATE_SOURCE_REQUIRED',
    );
    assert.ok(
      hasPrivate,
      `Non-DEEPENING document question must still produce PRIVATE_SOURCE_REQUIRED — got: ${JSON.stringify(d.claimRequirements)}`,
    );
  });

  test('C2: claimRequirements includes DOCUMENT_FACT', () => {
    const hasDocFact = (d.claimRequirements ?? []).some(
      (c) => c.claimType === 'DOCUMENT_FACT',
    );
    assert.ok(
      hasDocFact,
      `Non-DEEPENING document question must retain DOCUMENT_FACT claim — got: ${JSON.stringify(d.claimRequirements)}`,
    );
  });
});

// ── Test D: existing bare follow-up (≤5 words) — behavior unchanged ───────────

describe('D: existing bare follow-up (≤5 words) — FOLLOW_UP behavior unchanged', () => {
  // "Can you explain?" — 3 words, matches FOLLOW_UP_RE (starts with "can you"),
  // isBareFollowUp() returns true → last-resort at line 1123 skipped entirely.
  // This test proves the fix did not break pre-existing short follow-up handling.
  const Q = 'Can you explain?';
  const d = mkDecide(Q, { isFollowUp: true });

  test('D1: claimRequirements does NOT include PRIVATE_SOURCE_REQUIRED', () => {
    const hasPrivate = (d.claimRequirements ?? []).some(
      (c) => c.authority === 'PRIVATE_SOURCE_REQUIRED',
    );
    assert.ok(
      !hasPrivate,
      `Bare follow-up must not produce PRIVATE_SOURCE_REQUIRED — got: ${JSON.stringify(d.claimRequirements)}`,
    );
  });

  test('D2: interviewerBehavior is FOLLOW_UP (not DEEPENING)', () => {
    assert.equal(
      d.interviewIntent?.interviewerBehavior, 'FOLLOW_UP',
      `Bare follow-up must have interviewerBehavior=FOLLOW_UP — got: ${d.interviewIntent?.interviewerBehavior}`,
    );
  });
});
