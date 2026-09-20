// Phase 8 Step 19 — deterministic tests for the numeric provenance validator.
//
// Tests are split into two describe blocks:
//   extractFactualNumbers  — unit tests for the extraction + exclusion logic.
//   validateNumericProvenance — end-to-end provenance checks (the 15 required cases).
//
// No LLM calls. No V3 env flag needed. Pure function tests.

import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const base = path.resolve(process.cwd(), 'dist-electron/electron/context-intelligence');
const { extractFactualNumbers, validateNumericProvenance } =
  await import(pathToFileURL(path.join(base, 'validation/numeric-provenance.js')).href);

// ── extractFactualNumbers — unit tests for exclusion logic ───────────────────

describe('extractFactualNumbers: exclusion patterns', () => {
  test('T5-extract: Node.js 20 yields no factual numbers', () => {
    assert.deepStrictEqual(extractFactualNumbers('Node.js 20'), []);
  });

  test('T6-extract: Step 1 / Step 2 yield no factual numbers', () => {
    assert.deepStrictEqual(extractFactualNumbers('Step 1, then Step 2'), []);
  });

  test('T7-extract: O(n²) yields no factual numbers', () => {
    assert.deepStrictEqual(extractFactualNumbers('O(n²)'), []);
  });

  test('T8-extract: HTTP 404 yields no factual numbers', () => {
    assert.deepStrictEqual(extractFactualNumbers('HTTP 404'), []);
  });

  test('T9-extract: localhost:3000 yields no factual numbers', () => {
    assert.deepStrictEqual(extractFactualNumbers('localhost:3000'), []);
  });

  test('factual metric 5,000 is extracted', () => {
    assert.deepStrictEqual(extractFactualNumbers('5,000 requests/sec'), ['5000']);
  });

  test('factual percent 99.9% is extracted', () => {
    assert.deepStrictEqual(extractFactualNumbers('99.9%'), ['99.9']);
  });

  test('comma and non-comma forms normalize the same', () => {
    const a = extractFactualNumbers('16,000 users');
    const b = extractFactualNumbers('16000 users');
    assert.deepStrictEqual(a, ['16000']);
    assert.deepStrictEqual(b, ['16000']);
  });

  test('letter-before-digit identifier p99 is excluded', () => {
    assert.deepStrictEqual(extractFactualNumbers('p99 latency'), []);
  });

  test('unit-suffix digit 340ms: factual 340 is extracted (ms suffix not excluded)', () => {
    assert.deepStrictEqual(extractFactualNumbers('340ms latency'), ['340']);
  });
});

// ── validateNumericProvenance — the 15 required cases ───────────────────────

describe('validateNumericProvenance: 15 required cases', () => {
  // T1: PASS — exact metric
  test('T1: exact metric — PASS', () => {
    const r = validateNumericProvenance(
      '5,000 requests/sec',
      ['system handles 5,000 requests/sec'],
    );
    assert.strictEqual(r.valid, true, `unsupported: ${JSON.stringify(r.unsupported)}`);
    assert.deepStrictEqual(r.unsupported, []);
  });

  // T2: FAIL — changed metric
  test('T2: changed metric — FAIL', () => {
    const r = validateNumericProvenance(
      '10,000 requests/sec',
      ['5,000 requests/sec'],
    );
    assert.strictEqual(r.valid, false);
    assert.ok(r.unsupported.includes('10000'),
      `expected "10000" in unsupported, got ${JSON.stringify(r.unsupported)}`);
  });

  // T3: PASS — percent formatting normalization
  test('T3: percent formatting (99.9% ↔ 99.9 percent) — PASS', () => {
    const r = validateNumericProvenance(
      '99.9 percent',
      ['99.9%'],
    );
    assert.strictEqual(r.valid, true, `unsupported: ${JSON.stringify(r.unsupported)}`);
    assert.deepStrictEqual(r.unsupported, []);
  });

  // T4: FAIL — invented number not in any source
  test('T4: invented number (42%) — FAIL', () => {
    const r = validateNumericProvenance(
      '42%',
      ['the system should handle requests efficiently'],
    );
    assert.strictEqual(r.valid, false);
    assert.ok(r.unsupported.includes('42'),
      `expected "42" in unsupported, got ${JSON.stringify(r.unsupported)}`);
  });

  // T5: PASS — software version should not be flagged
  test('T5: Node.js 20 in answer — PASS (version excluded)', () => {
    const r = validateNumericProvenance('Using Node.js 20 for the runtime', []);
    assert.strictEqual(r.valid, true, `unsupported: ${JSON.stringify(r.unsupported)}`);
  });

  // T6: PASS — numbered list steps should not be flagged
  test('T6: numbered steps (Step 1, Step 2) in answer — PASS', () => {
    const r = validateNumericProvenance('Step 1, then Step 2', []);
    assert.strictEqual(r.valid, true, `unsupported: ${JSON.stringify(r.unsupported)}`);
  });

  // T7: PASS — complexity notation should not be flagged
  test('T7: O(n²) complexity in answer — PASS', () => {
    const r = validateNumericProvenance('This runs in O(n²) time', []);
    assert.strictEqual(r.valid, true, `unsupported: ${JSON.stringify(r.unsupported)}`);
  });

  // T8: PASS — HTTP status code should not be flagged
  test('T8: HTTP 404 in answer — PASS', () => {
    const r = validateNumericProvenance('Returns HTTP 404 when not found', []);
    assert.strictEqual(r.valid, true, `unsupported: ${JSON.stringify(r.unsupported)}`);
  });

  // T9: PASS — port number in URL should not be flagged
  test('T9: localhost:3000 in answer — PASS', () => {
    const r = validateNumericProvenance('Server runs on localhost:3000', []);
    assert.strictEqual(r.valid, true, `unsupported: ${JSON.stringify(r.unsupported)}`);
  });

  // T10: PASS — multiple supported numbers, all from source
  test('T10: multiple supported numbers — PASS', () => {
    const r = validateNumericProvenance(
      'handles 5,000 requests/sec with 99.9% uptime',
      ['5,000 requests/sec and 99.9% availability'],
    );
    assert.strictEqual(r.valid, true, `unsupported: ${JSON.stringify(r.unsupported)}`);
    assert.deepStrictEqual(r.unsupported, []);
  });

  // T11: FAIL — one supported, one unsupported; must list unsupported only
  test('T11: one supported + one unsupported (42%) — FAIL', () => {
    const r = validateNumericProvenance(
      '5,000 requests/sec with 42% failure rate',
      ['5,000 requests/sec'],
    );
    assert.strictEqual(r.valid, false);
    assert.ok(r.unsupported.includes('42'),
      `expected "42" in unsupported, got ${JSON.stringify(r.unsupported)}`);
    assert.ok(!r.unsupported.includes('5000'),
      `"5000" must not be unsupported, got ${JSON.stringify(r.unsupported)}`);
  });

  // T12: FAIL — unit conversion intentionally unsupported
  test('T12: unit conversion (1.2 seconds vs 1200ms) — FAIL (by design)', () => {
    // "1.2 seconds" normalizes to "1.2"; "1200ms" normalizes to "1200".
    // Without unit-aware normalization these are distinct values. This is the
    // known limitation documented in the Step 19 design.
    const r = validateNumericProvenance(
      '1200ms',
      ['1.2 seconds'],
    );
    assert.strictEqual(r.valid, false,
      'unit conversion must remain unsupported in this version');
    assert.ok(r.unsupported.includes('1200'),
      `expected "1200" in unsupported, got ${JSON.stringify(r.unsupported)}`);
  });

  // T13: PASS — answer has no numbers at all
  test('T13: answer with no numbers — PASS', () => {
    const r = validateNumericProvenance(
      'The system uses a distributed architecture',
      [],
    );
    assert.strictEqual(r.valid, true);
  });

  // T14: PASS — answer has only structural/technical numbers; empty sources
  test('T14: only structural/technical numbers in answer — PASS', () => {
    const r = validateNumericProvenance(
      'Step 1: install Node.js 20 and run on localhost:3000',
      [],
    );
    assert.strictEqual(r.valid, true, `unsupported: ${JSON.stringify(r.unsupported)}`);
  });

  // T15: PASS — formatting normalization (comma vs no-comma)
  test('T15: comma formatting (16,000 source vs 16000 answer) — PASS', () => {
    const r = validateNumericProvenance(
      '16000 users',
      ['16,000 users'],
    );
    assert.strictEqual(r.valid, true, `unsupported: ${JSON.stringify(r.unsupported)}`);
  });
});

// ── Phase 8 Step 22: FP hardening tests ─────────────────────────────────────
// T1–T12 exercise the two false-positive fixes:
//   FP-1: code regions (fenced + inline) are masked before extraction.
//   FP-2: bare HTTP status codes in context are excluded.
// T12 explicitly protects the decision NOT to implement arithmetic reasoning.

describe('FP hardening: code masking (FP-1) and bare HTTP status codes (FP-2)', () => {

  // ── FP-1: fenced code block ───────────────────────────────────────────────

  test('T1-fp1: fenced code block literals (0,1,2) yield no factual numbers', () => {
    const answer = '```js\nconst count = 0;\ncount++;\nreturn count;\n```';
    assert.deepStrictEqual(extractFactualNumbers(answer), []);
  });

  test('T2-fp1: inline code literal (3) yields no factual numbers', () => {
    const answer = 'Use `retryCount = 3` before returning.';
    assert.deepStrictEqual(extractFactualNumbers(answer), []);
  });

  test('T3-fp1: code block + factual prose number — only prose number extracted', () => {
    const answer =
      'The service handles 5,000 requests/sec.\n\n```js\nconst count = 0;\n```';
    const nums = extractFactualNumbers(answer);
    assert.ok(nums.includes('5000'), `expected "5000" in ${JSON.stringify(nums)}`);
    assert.ok(!nums.includes('0'), `"0" must not be extracted from code block; got ${JSON.stringify(nums)}`);
  });

  // ── FP-2: bare HTTP status codes ─────────────────────────────────────────

  test('T4-fp2: "301 redirect" yields no factual numbers', () => {
    assert.deepStrictEqual(extractFactualNumbers('The server returned a 301 redirect.'), []);
  });

  test('T5-fp2: "returned 404" yields no factual numbers', () => {
    assert.deepStrictEqual(extractFactualNumbers('The API returned 404 when the resource was missing.'), []);
  });

  test('T6-fp2: "500 error" yields no factual numbers', () => {
    assert.deepStrictEqual(extractFactualNumbers('The service returned a 500 error.'), []);
  });

  // ── Regression: real metrics must not be affected ─────────────────────────

  test('T7-reg: 4-digit performance metric (5000 rps) still extracted', () => {
    const nums = extractFactualNumbers('The service handles 5000 requests per second.');
    assert.deepStrictEqual(nums, ['5000']);
  });

  test('T8-reg: latency metric (340ms) still extracted', () => {
    const nums = extractFactualNumbers('The p99 latency is 340ms.');
    assert.ok(nums.includes('340'), `expected "340" in ${JSON.stringify(nums)}`);
    assert.ok(!nums.includes('99'), '"99" from p99 must remain excluded');
  });

  test('T9-reg: "returned 503" excluded; "5000 requests/sec" still extracted', () => {
    const nums = extractFactualNumbers(
      'The API returned 503, while the service handles 5000 requests/sec.',
    );
    assert.ok(nums.includes('5000'), `expected "5000" in ${JSON.stringify(nums)}`);
    assert.ok(!nums.includes('503'), '"503" must be excluded; got ' + JSON.stringify(nums));
  });

  // ── Full validateNumericProvenance with FP-1 fix ──────────────────────────

  test('T10-fp1: code block + factual prose number — provenance PASS', () => {
    const answer =
      'The service handles 5000 requests/sec.\n\n```js\nconst count = 0;\ncount++;\n```';
    const r = validateNumericProvenance(answer, ['The service handles 5000 requests/sec.']);
    assert.strictEqual(r.valid, true, `unsupported: ${JSON.stringify(r.unsupported)}`);
    assert.deepStrictEqual(r.unsupported, []);
  });

  test('T11-reg: invented metric (7500) still detected after FP hardening', () => {
    const r = validateNumericProvenance(
      'The service handles 7500 requests/sec.',
      ['The service handles 5000 requests/sec.'],
    );
    assert.strictEqual(r.valid, false);
    assert.ok(r.unsupported.includes('7500'),
      `expected "7500" in unsupported, got ${JSON.stringify(r.unsupported)}`);
  });

  // ── T12: arithmetic derivation remains unsupported (by design) ───────────

  test('T12-arith: arithmetic-derived value (0.1) remains unsupported — by design', () => {
    // Source: 99.9% availability.
    // Answer derives: 0.1% error budget (100% − 99.9%).
    // The validator does NOT perform arithmetic reasoning; 0.1 is not in source.
    // This test explicitly documents the known limitation and must remain failing.
    const r = validateNumericProvenance(
      'The remaining error budget is 0.1%.',
      ['Availability is 99.9%.'],
    );
    assert.strictEqual(r.valid, false,
      'arithmetic derivation must remain unsupported in this version');
    assert.ok(r.unsupported.includes('0.1'),
      `expected "0.1" in unsupported, got ${JSON.stringify(r.unsupported)}`);
  });
});
