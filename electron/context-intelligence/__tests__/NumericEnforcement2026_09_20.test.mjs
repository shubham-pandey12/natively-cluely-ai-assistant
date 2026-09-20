// Phase 8 Step 23 — deterministic tests for the numeric enforcement module.
//
// All tests are pure: no LLM calls, no IO, no streaming.
// The `regenerate` callback is a mock function under test control.
//
// T1: valid answer → no enforcement, finalValid true, retryCount 0
// T2: invalid answer + no regenerate → no enforcement, finalValid false, original answer
// T3: invalid + regenerate returns valid → enforced, retryCount 1, finalValid true
// T4: invalid + regenerate returns still-invalid → enforced, retryCount 1, finalValid false
// T5: correct CORRECTION_HINT passed to regenerate
// T6: regenerate called at most once (MAX_NUMERIC_RETRIES = 1 guarantee)
// T7: MAX_NUMERIC_RETRIES constant equals 1
// T8: answer with no numbers → initialValid true, no enforcement

import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const base = path.resolve(process.cwd(), 'dist-electron/electron/context-intelligence');
const { applyNumericEnforcement, MAX_NUMERIC_RETRIES, CORRECTION_HINT } =
  await import(pathToFileURL(path.join(base, 'validation/numeric-enforcement.js')).href);

describe('applyNumericEnforcement: core enforcement logic', () => {

  // ── T1: valid answer → no enforcement ──────────────────────────────────────

  test('T1: valid answer returns immediately without enforcement', async () => {
    const answer  = 'The service handles 5000 requests/sec.';
    const sources = ['handles 5000 requests/sec'];
    const result  = await applyNumericEnforcement(answer, sources);

    assert.strictEqual(result.initialValid,  true,  'initialValid must be true');
    assert.strictEqual(result.enforced,      false, 'enforced must be false');
    assert.strictEqual(result.retryCount,    0,     'retryCount must be 0');
    assert.strictEqual(result.finalValid,    true,  'finalValid must be true');
    assert.deepStrictEqual(result.initialUnsupported, [], 'initialUnsupported must be empty');
    assert.deepStrictEqual(result.finalUnsupported,   [], 'finalUnsupported must be empty');
    assert.strictEqual(result.finalAnswer, answer,  'finalAnswer must be original');
  });

  // ── T2: invalid answer + no regenerate → observability-only ────────────────

  test('T2: invalid answer with no regenerate callback — no enforcement, original answer preserved', async () => {
    const answer  = 'The system processes 9999 requests/sec.';
    const sources = ['handles 5000 requests/sec'];
    const result  = await applyNumericEnforcement(answer, sources);

    assert.strictEqual(result.initialValid,  false, 'initialValid must be false');
    assert.strictEqual(result.enforced,      false, 'enforced must be false (no callback)');
    assert.strictEqual(result.retryCount,    0,     'retryCount must be 0');
    assert.strictEqual(result.finalValid,    false, 'finalValid must be false');
    assert.ok(result.initialUnsupported.includes('9999'),
      `initialUnsupported must include "9999"; got ${JSON.stringify(result.initialUnsupported)}`);
    assert.ok(result.finalUnsupported.includes('9999'),
      `finalUnsupported must include "9999" (no change); got ${JSON.stringify(result.finalUnsupported)}`);
    assert.strictEqual(result.finalAnswer, answer, 'finalAnswer must be original (unmodified)');
  });

  // ── T3: invalid + regenerate returns valid ──────────────────────────────────

  test('T3: invalid answer + regenerate returns valid answer — enforced, finalValid true', async () => {
    const originalAnswer = 'The service runs at 9999 requests/sec.';
    const retryAnswer    = 'The service runs at 5000 requests/sec.';
    const sources        = ['handles 5000 requests/sec'];

    const regenerate = async (_hint) => retryAnswer;
    const result = await applyNumericEnforcement(originalAnswer, sources, regenerate);

    assert.strictEqual(result.initialValid,  false, 'initialValid must be false');
    assert.strictEqual(result.enforced,      true,  'enforced must be true');
    assert.strictEqual(result.retryCount,    1,     'retryCount must be 1');
    assert.strictEqual(result.finalValid,    true,  'finalValid must be true after retry');
    assert.deepStrictEqual(result.finalUnsupported, [], 'finalUnsupported must be empty after valid retry');
    assert.strictEqual(result.finalAnswer, retryAnswer, 'finalAnswer must be the retry answer');
  });

  // ── T4: invalid + regenerate returns still-invalid ─────────────────────────

  test('T4: invalid answer + regenerate returns still-invalid — enforced, finalValid false', async () => {
    const originalAnswer = 'The system processes 9999 requests/sec.';
    const retryAnswer    = 'The system processes 8888 requests/sec.';
    const sources        = ['handles 5000 requests/sec'];

    const regenerate = async (_hint) => retryAnswer;
    const result = await applyNumericEnforcement(originalAnswer, sources, regenerate);

    assert.strictEqual(result.initialValid,  false, 'initialValid must be false');
    assert.strictEqual(result.enforced,      true,  'enforced must be true');
    assert.strictEqual(result.retryCount,    1,     'retryCount must be 1');
    assert.strictEqual(result.finalValid,    false, 'finalValid must be false (retry still invalid)');
    assert.ok(result.finalUnsupported.includes('8888'),
      `finalUnsupported must include "8888"; got ${JSON.stringify(result.finalUnsupported)}`);
    assert.strictEqual(result.finalAnswer, retryAnswer, 'finalAnswer must be the retry answer');
  });

  // ── T5: CORRECTION_HINT passed to regenerate ────────────────────────────────

  test('T5: regenerate receives the correct CORRECTION_HINT string', async () => {
    const answer  = 'Uptime is 99 percent.';
    const sources = ['handles 5000 requests/sec'];

    let capturedHint = null;
    const regenerate = async (hint) => {
      capturedHint = hint;
      return 'The service is highly available.';
    };

    await applyNumericEnforcement(answer, sources, regenerate);

    assert.strictEqual(capturedHint, CORRECTION_HINT,
      `regenerate must receive CORRECTION_HINT exactly; got: ${JSON.stringify(capturedHint)}`);
  });

  // ── T6: regenerate called at most once ─────────────────────────────────────

  test('T6: regenerate is called at most once regardless of retry answer validity', async () => {
    const answer  = 'The system processes 9999 requests/sec.';
    const sources = ['handles 5000 requests/sec'];

    let callCount = 0;
    const regenerate = async (_hint) => {
      callCount++;
      return `Still bad: ${callCount * 1111} requests/sec.`;
    };

    const result = await applyNumericEnforcement(answer, sources, regenerate);

    assert.strictEqual(callCount, 1,
      `regenerate must be called exactly once; called ${callCount} times`);
    assert.strictEqual(result.retryCount, 1,
      `retryCount must be 1; got ${result.retryCount}`);
  });

  // ── T7: MAX_NUMERIC_RETRIES constant ────────────────────────────────────────

  test('T7: MAX_NUMERIC_RETRIES constant equals 1', () => {
    assert.strictEqual(MAX_NUMERIC_RETRIES, 1,
      `MAX_NUMERIC_RETRIES must be 1; got ${MAX_NUMERIC_RETRIES}`);
  });

  // ── T8: answer with no factual numbers → always valid, never enforced ───────

  test('T8: answer with no factual numbers — initialValid true, no enforcement', async () => {
    const answer  = 'Use a distributed cache to reduce latency.';
    const sources = [];

    let regenerateCalled = false;
    const regenerate = async (_hint) => {
      regenerateCalled = true;
      return 'retry';
    };

    const result = await applyNumericEnforcement(answer, sources, regenerate);

    assert.strictEqual(result.initialValid, true,  'initialValid must be true');
    assert.strictEqual(result.enforced,     false, 'enforced must be false');
    assert.strictEqual(result.retryCount,   0,     'retryCount must be 0');
    assert.strictEqual(result.finalValid,   true,  'finalValid must be true');
    assert.strictEqual(regenerateCalled,    false, 'regenerate must not be called for answer with no numbers');
    assert.strictEqual(result.finalAnswer, answer, 'finalAnswer must be original');
  });

});
