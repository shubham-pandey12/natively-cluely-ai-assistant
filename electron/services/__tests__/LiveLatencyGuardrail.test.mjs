// LiveLatencyGuardrail.test.mjs
// Tests for the LLM first-token latency fixes:
//   1. totalHardTimeoutMs() returns per-type deadline (complex > standard > local)
//   2. Speculative validity: word-count ratio rejects if confirmed >50% longer
//   3. Jaccard acceptance still works for similar text
//   4. Timeout-path does not corrupt newer confirmed generation

import { test } from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const deadlinesPath = path.resolve(__dirname, '../../../dist-electron/electron/llm/liveDeadlines.js');
const enginePath = path.resolve(__dirname, '../../../dist-electron/electron/IntelligenceEngine.js');
const sessionPath = path.resolve(__dirname, '../../../dist-electron/electron/SessionTracker.js');
const require = createRequire(import.meta.url);

// ── Helpers ──────────────────────────────────────────────────────────────────

async function loadDeadlines() {
  return import(pathToFileURL(deadlinesPath).href);
}

function makeHelper() {
  return {
    setNegotiationCoachingHandler() {},
    isUsingOllama() { return false; },
  };
}

async function makeEngine() {
  const { IntelligenceEngine } = await import(pathToFileURL(enginePath).href);
  const { SessionTracker } = require(sessionPath);
  const session = new SessionTracker();
  const engine = new IntelligenceEngine(makeHelper(), session);
  return { engine, session };
}

function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// ── totalHardTimeoutMs per-type deadline ──────────────────────────────────────

test('totalHardTimeoutMs returns extended deadline for system_design_answer', async () => {
  const { totalHardTimeoutMs, LIVE_COMPLEX_TOTAL_HARD_TIMEOUT_MS, LIVE_TOTAL_HARD_TIMEOUT_MS } = await loadDeadlines();
  const systemDesignDeadline = totalHardTimeoutMs('system_design_answer', false);
  assert.ok(
    systemDesignDeadline >= LIVE_COMPLEX_TOTAL_HARD_TIMEOUT_MS,
    `system_design_answer should get ${LIVE_COMPLEX_TOTAL_HARD_TIMEOUT_MS}ms, got ${systemDesignDeadline}ms`,
  );
  assert.ok(
    systemDesignDeadline > LIVE_TOTAL_HARD_TIMEOUT_MS,
    `complex deadline must exceed standard deadline of ${LIVE_TOTAL_HARD_TIMEOUT_MS}ms`,
  );
});

test('totalHardTimeoutMs returns extended deadline for coding_question_answer', async () => {
  const { totalHardTimeoutMs, LIVE_COMPLEX_TOTAL_HARD_TIMEOUT_MS } = await loadDeadlines();
  const codingDeadline = totalHardTimeoutMs('coding_question_answer', false);
  assert.equal(codingDeadline, LIVE_COMPLEX_TOTAL_HARD_TIMEOUT_MS);
});

test('totalHardTimeoutMs returns standard deadline for general answers', async () => {
  const { totalHardTimeoutMs, LIVE_TOTAL_HARD_TIMEOUT_MS } = await loadDeadlines();
  for (const answerType of ['general_answer', 'behavioral_answer', 'general_meeting_answer']) {
    const d = totalHardTimeoutMs(answerType, false);
    assert.equal(d, LIVE_TOTAL_HARD_TIMEOUT_MS, `${answerType} should use standard timeout`);
  }
});

test('totalHardTimeoutMs returns local deadline regardless of type', async () => {
  const { totalHardTimeoutMs, LIVE_LOCAL_TOTAL_HARD_TIMEOUT_MS } = await loadDeadlines();
  assert.equal(totalHardTimeoutMs('system_design_answer', true), LIVE_LOCAL_TOTAL_HARD_TIMEOUT_MS);
  assert.equal(totalHardTimeoutMs('general_answer', true), LIVE_LOCAL_TOTAL_HARD_TIMEOUT_MS);
});

// ── Speculative validity: word-count ratio ────────────────────────────────────

test('speculative stream is rejected when confirmed question is >50% longer (word ratio)', async () => {
  const { engine, session } = await makeEngine();
  let generationCount = 0;
  engine.whatToAnswerLLM = {
    async *generateStream() {
      generationCount++;
      yield 'speculative answer for testing purposes here now done';
    },
  };

  // Short speculative question (5 words)
  const shortQuestion = 'How would you design this';
  session.handleTranscript({ speaker: 'interviewer', text: shortQuestion, timestamp: Date.now(), final: true, confidence: 0.9 });
  await engine.runWhatShouldISay(shortQuestion, 0.85, undefined, { speculative: true });

  // Confirmed question is much longer (>50% more words): should reject speculative
  const longQuestion = 'How would you design this URL shortener at massive global scale with high availability and fault tolerance';
  await engine.handleSuggestionTrigger({
    context: longQuestion,
    lastQuestion: longQuestion,
    confidence: 0.9,
    skipCooldown: true,
  });

  // Because word-count ratio rejected the speculative, confirmed should start a new generation
  assert.ok(generationCount >= 2, `Word-ratio rejection should have started a new confirmed generation (got ${generationCount} generations)`);
});

test('speculative stream is accepted when confirmed question is similarly short (same words)', async () => {
  const { engine, session } = await makeEngine();
  let generationCount = 0;
  engine.whatToAnswerLLM = {
    async *generateStream() {
      generationCount++;
      yield 'answer to question about design system';
    },
  };

  const question = 'How would you design a caching layer?';
  session.handleTranscript({ speaker: 'interviewer', text: question, timestamp: Date.now(), final: true, confidence: 0.9 });
  await engine.runWhatShouldISay(question, 0.85, undefined, { speculative: true });

  // Confirmed question nearly identical → Jaccard passes AND word ratio passes → reuse
  const confirmedQuestion = 'How would you design a caching layer?';
  await engine.handleSuggestionTrigger({
    context: confirmedQuestion,
    lastQuestion: confirmedQuestion,
    confidence: 0.9,
    skipCooldown: true,
  });

  assert.ok(generationCount <= 1, `Similar question should reuse speculative — no new generation (got ${generationCount})`);
});

// ── Timeout path does not corrupt newer generation (generationId guard) ───────

test('first-token timeout on a stale generation does not emit fallback over newer answer', async () => {
  const { engine, session } = await makeEngine();

  const emittedFallbacks = [];
  engine.on('suggested_answer_token', (chunk) => {
    if (chunk && chunk.includes('did not produce')) {
      emittedFallbacks.push(chunk);
    }
  });

  const question = 'Explain the CAP theorem in detail?';
  session.handleTranscript({ speaker: 'interviewer', text: question, timestamp: Date.now(), final: true, confidence: 0.9 });

  // Stub: first call hangs forever (simulates timeout), second succeeds quickly
  let callCount = 0;
  engine.whatToAnswerLLM = {
    async *generateStream() {
      callCount++;
      if (callCount === 1) {
        // Hang long enough for the generation to be superseded
        await delay(500);
        yield 'late answer that should be discarded';
      } else {
        yield 'fast answer from newer generation';
      }
    },
  };

  // Start first generation (will be slow)
  const firstGen = engine.runWhatShouldISay(question, 0.85);
  // Immediately start a second generation (supersedes the first)
  await delay(50);
  const secondGen = engine.runWhatShouldISay(question, 0.9);
  await Promise.all([firstGen, secondGen]);

  // The "did not produce" timeout message should NOT appear since the generation was superseded
  assert.equal(emittedFallbacks.length, 0, 'Timeout fallback must not be emitted when a newer generation is active');
});

// ── LIVE_COMPLEX_TOTAL_HARD_TIMEOUT_MS is meaningfully larger than standard ──

test('LIVE_COMPLEX_TOTAL_HARD_TIMEOUT_MS is at least 5s more than LIVE_TOTAL_HARD_TIMEOUT_MS', async () => {
  const { LIVE_COMPLEX_TOTAL_HARD_TIMEOUT_MS, LIVE_TOTAL_HARD_TIMEOUT_MS } = await loadDeadlines();
  assert.ok(
    LIVE_COMPLEX_TOTAL_HARD_TIMEOUT_MS >= LIVE_TOTAL_HARD_TIMEOUT_MS + 5000,
    `Complex timeout (${LIVE_COMPLEX_TOTAL_HARD_TIMEOUT_MS}ms) should be at least 5s > standard (${LIVE_TOTAL_HARD_TIMEOUT_MS}ms)`,
  );
});
