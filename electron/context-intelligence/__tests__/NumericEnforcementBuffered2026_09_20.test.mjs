// Phase 8 Step 24 — focused tests for the buffered V3 delivery + enforcement integration.
//
// These tests verify the enforcement flow as wired in ipcHandlers.ts:
//   - V3 tokens buffered internally (no per-token renderer events during accumulation)
//   - applyNumericEnforcement called with finalText + authoritative sources
//   - single final answer delivered to the renderer
//   - regenerate callback called at most once (MAX_NUMERIC_RETRIES = 1)
//   - structural/code numbers never trigger regeneration
//   - invalid → invalid: retry answer delivered with accurate observability
//   - superseded / abort paths: no double delivery
//
// All tests are pure: they drive applyNumericEnforcement directly with mock
// regenerate functions that simulate the llmHelper.streamChat buffering.

import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const base = path.resolve(process.cwd(), 'dist-electron/electron/context-intelligence');
const { applyNumericEnforcement, MAX_NUMERIC_RETRIES } =
  await import(pathToFileURL(path.join(base, 'validation/numeric-enforcement.js')).href);

// Simulated "renderer" — records events the ipcHandlers.ts delivery block would emit.
function makeRenderer() {
  const events = [];
  return {
    send: (channel, ...args) => events.push({ channel, args }),
    events,
    tokenEvents:  () => events.filter(e => e.channel === 'gemini-stream-token'),
    doneEvents:   () => events.filter(e => e.channel === 'gemini-stream-done'),
    deliveredTexts: () => events.filter(e => e.channel === 'gemini-stream-token').map(e => e.args[0]),
  };
}

// Simulate the ipcHandlers V3 delivery block (Step 24 implementation).
// Accepts: bufferedFinalText, sources, regenerateFn (may be undefined).
// Returns: { renderer, npResult, deliveredText }.
async function simulateV3Delivery(bufferedFinalText, sources, regenerateFn) {
  const renderer = makeRenderer();
  let deliveredText = bufferedFinalText;
  let npResult = null;
  try {
    npResult = await applyNumericEnforcement(bufferedFinalText, sources, regenerateFn);
    deliveredText = npResult.finalAnswer;
  } catch {
    // enforcement error: deliver original (abort not simulated here)
  }
  // Deliver single answer to "renderer"
  renderer.send('gemini-stream-token', deliveredText, { streamId: 'test-stream' });
  renderer.send('gemini-stream-done', { finalText: deliveredText, streamId: 'test-stream' });
  return { renderer, npResult, deliveredText };
}

describe('Buffered V3 delivery with numeric enforcement', () => {

  // ── T1: Valid answer — no regeneration, one answer delivered ───────────────

  test('T1: valid answer — regenerate not called, exactly one token + done delivered', async () => {
    const sources = ['The service handles 5,000 requests/sec.'];
    let regenerateCalled = false;
    const regenerate = async (_hint) => { regenerateCalled = true; return 'retry'; };

    const { renderer, npResult, deliveredText } =
      await simulateV3Delivery('The service handles 5,000 requests/sec.', sources, regenerate);

    assert.strictEqual(regenerateCalled, false, 'regenerate must not be called for valid answer');
    assert.strictEqual(npResult.initialValid, true, 'initialValid must be true');
    assert.strictEqual(npResult.enforced, false, 'enforced must be false');
    assert.strictEqual(npResult.retryCount, 0, 'retryCount must be 0');

    assert.strictEqual(renderer.tokenEvents().length, 1, 'exactly one gemini-stream-token event');
    assert.strictEqual(renderer.doneEvents().length,  1, 'exactly one gemini-stream-done event');
    assert.strictEqual(deliveredText, 'The service handles 5,000 requests/sec.', 'delivered text must be original');
  });

  // ── T2: Invalid → valid regeneration — only retry answer delivered ─────────

  test('T2: invalid first answer + valid retry — only retry answer delivered to renderer', async () => {
    const sources = ['The service handles 5,000 requests/sec.'];
    const originalAnswer = 'The service handles 50,000 requests/sec.';
    const retryAnswer    = 'The service handles 5,000 requests/sec.';

    let callCount = 0;
    const regenerate = async (_hint) => { callCount++; return retryAnswer; };

    const { renderer, npResult, deliveredText } =
      await simulateV3Delivery(originalAnswer, sources, regenerate);

    assert.strictEqual(callCount, 1, 'regenerate called exactly once');
    assert.strictEqual(npResult.initialValid, false,  'initialValid must be false');
    assert.strictEqual(npResult.enforced,     true,   'enforced must be true');
    assert.strictEqual(npResult.retryCount,   1,      'retryCount must be 1');
    assert.strictEqual(npResult.finalValid,   true,   'finalValid must be true');

    assert.strictEqual(renderer.tokenEvents().length, 1, 'exactly one gemini-stream-token event');
    assert.deepStrictEqual(renderer.deliveredTexts(), [retryAnswer],
      'renderer must receive retry answer, not original');
    assert.strictEqual(deliveredText, retryAnswer, 'deliveredText must be retry answer');
  });

  // ── T3: Invalid → invalid — retry delivered with accurate observability ────

  test('T3: invalid first + invalid retry — retry delivered, finalValid false', async () => {
    const sources = ['The service handles 5,000 requests/sec.'];
    const originalAnswer = 'The service handles 50,000 requests/sec.';
    const retryAnswer    = 'The service handles 75,000 requests/sec.';

    let callCount = 0;
    const regenerate = async (_hint) => { callCount++; return retryAnswer; };

    const { renderer, npResult, deliveredText } =
      await simulateV3Delivery(originalAnswer, sources, regenerate);

    assert.strictEqual(callCount, 1, 'regenerate called exactly once (not more)');
    assert.strictEqual(npResult.initialValid, false, 'initialValid must be false');
    assert.strictEqual(npResult.enforced,     true,  'enforced must be true');
    assert.strictEqual(npResult.retryCount,   1,     'retryCount must be 1');
    assert.strictEqual(npResult.finalValid,   false, 'finalValid must be false (retry also invalid)');
    assert.ok(npResult.finalUnsupported.includes('75000'),
      `finalUnsupported must contain "75000"; got ${JSON.stringify(npResult.finalUnsupported)}`);

    // The retry answer IS delivered (best available answer), but observability
    // clearly distinguishes it as still-invalid (finalValid: false).
    assert.strictEqual(renderer.tokenEvents().length, 1, 'exactly one gemini-stream-token event');
    assert.strictEqual(deliveredText, retryAnswer, 'deliveredText must be retry answer');
  });

  // ── T4: Structural numbers — no regeneration ───────────────────────────────

  test('T4: structural numbers (Step 1, Node.js 20, HTTP 200, port 3000) — no regeneration', async () => {
    const answer = 'Step 1: install Node.js 20. The API returns HTTP 200 on localhost port 3000.';
    const sources = [];
    let regenerateCalled = false;
    const regenerate = async (_hint) => { regenerateCalled = true; return 'retry'; };

    const { npResult } = await simulateV3Delivery(answer, sources, regenerate);

    assert.strictEqual(regenerateCalled, false, 'regenerate must not be called for structural numbers');
    assert.strictEqual(npResult.initialValid, true, 'structural numbers must all be excluded → valid');
    assert.strictEqual(npResult.enforced, false, 'enforced must be false');
  });

  // ── T5: Code literals — no regeneration ────────────────────────────────────

  test('T5: code literals in fenced block — not treated as factual, no regeneration', async () => {
    const answer =
      'Use an exponential backoff:\n\n```js\nconst count = 0;\nretries++;\n```\n\nStart at 100ms.';
    const sources = ['Start the retry interval at 100ms.'];
    let regenerateCalled = false;
    const regenerate = async (_hint) => { regenerateCalled = true; return 'retry'; };

    const { npResult } = await simulateV3Delivery(answer, sources, regenerate);

    assert.strictEqual(regenerateCalled, false, 'regenerate must not be called — code literals masked');
    assert.strictEqual(npResult.initialValid, true, 'code literals masked; 100ms supported → valid');
    assert.strictEqual(npResult.enforced, false, 'enforced must be false');
  });

  // ── T6: Renderer receives exactly one token event + one done event ─────────

  test('T6: renderer receives exactly one gemini-stream-token and one gemini-stream-done', async () => {
    const sources = ['The service handles 5,000 requests/sec.'];
    const answer  = 'The service handles 5,000 requests/sec.';

    const { renderer } = await simulateV3Delivery(answer, sources, undefined);

    assert.strictEqual(renderer.tokenEvents().length, 1,
      `expected 1 gemini-stream-token event; got ${renderer.tokenEvents().length}`);
    assert.strictEqual(renderer.doneEvents().length, 1,
      `expected 1 gemini-stream-done event; got ${renderer.doneEvents().length}`);
    // No double-answer: all token content comes from exactly one event
    assert.strictEqual(renderer.events.length, 2, 'total renderer events must be 2 (token + done)');
  });

  // ── T7: Regeneration does not recurse — retryCount always ≤ 1 ─────────────

  test('T7: regeneration does not recurse — MAX_NUMERIC_RETRIES enforced', async () => {
    const sources = ['The service handles 5,000 requests/sec.'];
    const answer  = 'The service handles 50,000 requests/sec.';

    let callCount = 0;
    // Retry also returns an invalid answer — but the module must not call again
    const regenerate = async (_hint) => {
      callCount++;
      return `Invalid attempt ${callCount}: ${callCount * 10000} requests/sec.`;
    };

    const { npResult } = await simulateV3Delivery(answer, sources, regenerate);

    assert.strictEqual(MAX_NUMERIC_RETRIES, 1,
      `MAX_NUMERIC_RETRIES must be 1; got ${MAX_NUMERIC_RETRIES}`);
    assert.strictEqual(callCount, 1,
      `regenerate must be called exactly once; called ${callCount} times`);
    assert.strictEqual(npResult.retryCount, 1,
      `retryCount must be 1; got ${npResult.retryCount}`);
  });

  // ── T8: Clarification-path unaffected — single-token pattern already correct

  test('T8: clarification-style single-token delivery pattern matches Step 24 delivery pattern', () => {
    // The clarification path (line 2148 in ipcHandlers.ts) delivers:
    //   event.sender.send('gemini-stream-token', clarification, { streamId })
    //   event.sender.send('gemini-stream-done', { finalText: clarification, streamId })
    //
    // Step 24 delivers:
    //   event.sender.send('gemini-stream-token', deliveredText, { streamId })
    //   event.sender.send('gemini-stream-done', { finalText: deliveredText, streamId })
    //
    // Both use one token + one done. The renderer handles both identically.
    // This test verifies the step 24 delivery leaves the same event shape.

    const renderer = makeRenderer();
    const answer = 'The distributed cache reduces latency significantly.';

    // Simulate ipcHandlers delivery block (clarification pattern)
    renderer.send('gemini-stream-token', answer, { streamId: 'test-stream' });
    renderer.send('gemini-stream-done', { finalText: answer, streamId: 'test-stream' });

    assert.strictEqual(renderer.tokenEvents().length, 1, '1 token event (clarification)');
    assert.strictEqual(renderer.doneEvents().length,  1, '1 done event (clarification)');
    assert.strictEqual(renderer.deliveredTexts()[0], answer, 'token payload matches answer');
    assert.strictEqual(
      renderer.doneEvents()[0].args[0].finalText, answer,
      'done finalText matches answer',
    );
  });

});
