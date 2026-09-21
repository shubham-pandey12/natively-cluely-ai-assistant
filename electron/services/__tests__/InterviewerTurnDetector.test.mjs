// InterviewerTurnDetector.test.mjs
// Tests for the confirmed auto-trigger path (PATH 2):
//   final STT segment + silence debounce → INTERVIEWER_TURN_COMPLETED → handleSuggestionTrigger
//
// Also covers:
//   - duplicate trigger protection (same text)
//   - multi-segment accumulation
//   - speculative + confirmed dedup via Jaccard
//   - non-question text still triggers confirmed path
//   - skipCooldown threads through correctly

import { test } from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const enginePath = path.resolve(__dirname, '../../../dist-electron/electron/IntelligenceEngine.js');
const sessionPath = path.resolve(__dirname, '../../../dist-electron/electron/SessionTracker.js');
const require = createRequire(import.meta.url);

function makeHelper() {
  return { setNegotiationCoachingHandler() {} };
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

// --- Turn detection ---

test('final interviewer segment + silence fires INTERVIEWER_TURN_COMPLETED (handleSuggestionTrigger called)', async () => {
  const { engine, session } = await makeEngine();
  const triggers = [];
  const orig = engine.handleSuggestionTrigger.bind(engine);
  engine.handleSuggestionTrigger = async (trigger) => {
    triggers.push(trigger);
    return orig(trigger);
  };
  // Stub LLM to prevent actual generation
  engine.whatToAnswerLLM = { async *generateStream() { yield 'ok'; } };

  session.handleTranscript({ speaker: 'interviewer', text: 'How would you design a caching layer?', timestamp: Date.now(), final: true, confidence: 0.9 });
  engine.handleTranscript({ speaker: 'interviewer', text: 'How would you design a caching layer?', timestamp: Date.now(), final: true, confidence: 0.9, origin: 'stt' });

  // Wait longer than INTERVIEWER_TURN_SILENCE_MS (800ms)
  await delay(950);

  assert.equal(triggers.length, 1, 'exactly one trigger should fire');
  assert.ok(triggers[0].skipCooldown === true, 'confirmed trigger must set skipCooldown');
  assert.ok(triggers[0].confidence >= 0.8, 'confidence must be at least 0.8');
});

test('new interim speech before silence resets the timer — no premature completion', async () => {
  const { engine, session } = await makeEngine();
  const triggers = [];
  engine.handleSuggestionTrigger = async (trigger) => { triggers.push(trigger); };

  // Final segment arrives
  session.handleTranscript({ speaker: 'interviewer', text: 'How would you', timestamp: Date.now(), final: true, confidence: 0.9 });
  engine.handleTranscript({ speaker: 'interviewer', text: 'How would you', timestamp: Date.now(), final: true, confidence: 0.9, origin: 'stt' });

  // 400ms later: interim arrives (speaker still talking) — resets the timer
  await delay(400);
  session.handleTranscript({ speaker: 'interviewer', text: 'How would you design', timestamp: Date.now(), final: false, confidence: 0.85 });
  engine.handleTranscript({ speaker: 'interviewer', text: 'How would you design', timestamp: Date.now(), final: false, confidence: 0.85, origin: 'stt' });

  // Check immediately — timer was reset, nothing should have fired yet
  assert.equal(triggers.length, 0, 'no trigger before silence window completes after reset');

  // Wait for silence window from LAST speech
  await delay(950);
  // Now TURN_COMPLETED fires (from interim reset → no new final came → but timer resets, no final accumulated after interim)
  // Actually after the interim, since no new final arrives, the buffer contains only the first final's text.
  // The timer from the interim's clearTimeout means no completion fires unless a new final comes.
  // So triggers should still be 0 here since the last event was an interim (which only resets, doesn't rearm a new timer).
  // Let's just verify the behavior is defined: could be 0 or 1 depending on whether interim resets to no timer.
  // The implementation only arms the timer on FINAL segments, so after an interim the timer is cleared and not rearmed.
  // Therefore triggers.length should be 0 still.
  assert.equal(triggers.length, 0, 'interim does not complete a turn — timer stays cleared until next final');
});

test('multiple final segments accumulate into one turn, fire exactly once', async () => {
  const { engine, session } = await makeEngine();
  const triggers = [];
  engine.handleSuggestionTrigger = async (trigger) => { triggers.push(trigger); };

  // Two finals in quick succession (< SILENCE_MS apart)
  const t0 = Date.now();
  session.handleTranscript({ speaker: 'interviewer', text: 'How would you design', timestamp: t0, final: true, confidence: 0.9 });
  engine.handleTranscript({ speaker: 'interviewer', text: 'How would you design', timestamp: t0, final: true, confidence: 0.9, origin: 'stt' });

  await delay(200); // within silence window

  session.handleTranscript({ speaker: 'interviewer', text: 'a distributed cache?', timestamp: t0 + 200, final: true, confidence: 0.9 });
  engine.handleTranscript({ speaker: 'interviewer', text: 'a distributed cache?', timestamp: t0 + 200, final: true, confidence: 0.9, origin: 'stt' });

  // Now wait for silence
  await delay(1000);

  assert.equal(triggers.length, 1, 'multiple finals → exactly one trigger');
  // The trigger's lastQuestion should come from session (last interviewer turn)
  assert.ok(triggers[0].context.includes('How would you design') || triggers[0].context.includes('a distributed cache?'), 'context should contain interviewer text');
});

test('duplicate final transcript text does not fire a second trigger', async () => {
  const { engine, session } = await makeEngine();
  const triggers = [];
  engine.handleSuggestionTrigger = async (trigger) => { triggers.push(trigger); };

  const text = 'Tell me about your experience with distributed systems.';

  // First turn
  session.handleTranscript({ speaker: 'interviewer', text, timestamp: Date.now(), final: true, confidence: 0.9 });
  engine.handleTranscript({ speaker: 'interviewer', text, timestamp: Date.now(), final: true, confidence: 0.9, origin: 'stt' });
  await delay(950);
  assert.equal(triggers.length, 1, 'first turn fires once');

  // Duplicate: same text arrives again (STT provider resend)
  session.handleTranscript({ speaker: 'interviewer', text, timestamp: Date.now(), final: true, confidence: 0.9 });
  engine.handleTranscript({ speaker: 'interviewer', text, timestamp: Date.now(), final: true, confidence: 0.9, origin: 'stt' });
  await delay(950);
  assert.equal(triggers.length, 1, 'duplicate final text must not produce a second trigger');
});

// --- Confirmed path fires for non-question text ---

test('non-question interviewer turn still fires confirmed trigger (no hasQuestionSignal gate)', async () => {
  const { engine, session } = await makeEngine();
  const triggers = [];
  engine.handleSuggestionTrigger = async (trigger) => { triggers.push(trigger); };

  // "Okay let's move on." — no question signal, would have been skipped by old system
  const text = "Okay let's move on to the next topic.";
  session.handleTranscript({ speaker: 'interviewer', text, timestamp: Date.now(), final: true, confidence: 0.9 });
  engine.handleTranscript({ speaker: 'interviewer', text, timestamp: Date.now(), final: true, confidence: 0.9, origin: 'stt' });
  await delay(950);

  assert.equal(triggers.length, 1, 'confirmed path fires even without question keywords');
  assert.equal(triggers[0].skipCooldown, true, 'skipCooldown is set on confirmed triggers');
});

// --- Speculative generation improvements ---

test('speculative fires at 5 words (lowered threshold from 7)', async () => {
  const { engine, session } = await makeEngine();
  // Access the private constant via the compiled class
  // If SPECULATIVE_MIN_WORDS = 5, a 5-word partial with question signal should trigger
  let speculativeFired = false;
  engine.runWhatShouldISay = async (_question, _confidence, _images, opts) => {
    if (opts?.speculative) speculativeFired = true;
    return null;
  };

  // 5-word partial with question signal
  const segment = {
    speaker: 'interviewer',
    text: 'How would you design this',
    timestamp: Date.now(),
    final: false,
    confidence: 0.85,
    origin: 'stt',
  };
  session.handleTranscript({ ...segment });
  engine.handleTranscript({ ...segment });

  // Wait for speculative debounce (350ms)
  await delay(500);
  assert.ok(speculativeFired, 'speculative should fire at 5 words with question signal');
});

test('speculative does not fire at 4 words (below threshold)', async () => {
  const { engine, session } = await makeEngine();
  let speculativeFired = false;
  engine.runWhatShouldISay = async (_question, _confidence, _images, opts) => {
    if (opts?.speculative) speculativeFired = true;
    return null;
  };

  // Only 4 words
  const segment = {
    speaker: 'interviewer',
    text: 'How would you design',
    timestamp: Date.now(),
    final: false,
    confidence: 0.9,
    origin: 'stt',
  };
  session.handleTranscript({ ...segment });
  engine.handleTranscript({ ...segment });

  await delay(500);
  assert.equal(speculativeFired, false, 'speculative must not fire with only 4 words');
});

// --- Expanded question signals (PATH 1 speculative) ---

test('new question signals fire speculative: give me, show me, implement, build', async () => {
  const phrases = [
    'Give me an example of this pattern here',     // "give me" + 5+ words
    'Show me how you would implement this thing',  // "show me" + 5+ words
    'Implement a solution for this problem now',   // "implement" + 5+ words
    'Build a system that handles this workload',   // "build" + 5+ words
  ];

  for (const phrase of phrases) {
    const { engine, session } = await makeEngine();
    let fired = false;
    engine.runWhatShouldISay = async (_q, _c, _i, opts) => {
      if (opts?.speculative) fired = true;
      return null;
    };
    session.handleTranscript({ speaker: 'interviewer', text: phrase, timestamp: Date.now(), final: false, confidence: 0.85 });
    engine.handleTranscript({ speaker: 'interviewer', text: phrase, timestamp: Date.now(), final: false, confidence: 0.85, origin: 'stt' });
    await delay(500);
    assert.ok(fired, `speculative should fire for phrase: "${phrase}"`);
  }
});

// --- No duplicate generation from speculation + confirmation ---

test('speculative stream with high Jaccard match prevents duplicate confirmed generation', async () => {
  const { engine, session } = await makeEngine();
  let generationCount = 0;
  engine.whatToAnswerLLM = {
    async *generateStream() {
      generationCount++;
      yield 'speculative answer text here for testing';
    },
  };

  // Seed interviewer turn in session
  const question = 'How would you design a caching system for scale?';
  session.handleTranscript({ speaker: 'interviewer', text: question, timestamp: Date.now(), final: true, confidence: 0.9, origin: 'stt' });

  // Manually trigger speculative with same question (simulates maybeSpeculate)
  await engine.runWhatShouldISay(question, 0.85, undefined, { speculative: true });

  // Now confirmed trigger fires with same text → Jaccard accepts → no new generation
  await engine.handleSuggestionTrigger({
    context: question,
    lastQuestion: question,
    confidence: 0.9,
    skipCooldown: true,
  });

  assert.ok(generationCount <= 1, `speculative accepted by Jaccard — confirmed should not start new generation (got ${generationCount})`);
});

// --- skipCooldown threads through to runWhatShouldISay ---

test('skipCooldown on SuggestionTrigger bypasses shouldThrottleTrigger in runWhatShouldISay', async () => {
  const { engine, session } = await makeEngine();
  let generationCount = 0;
  engine.whatToAnswerLLM = {
    async *generateStream() {
      generationCount++;
      yield 'answer';
    },
  };

  const question = 'Explain the CAP theorem and when to apply it?';
  session.handleTranscript({ speaker: 'interviewer', text: question, timestamp: Date.now(), final: true, confidence: 0.9, origin: 'stt' });

  // Stamp lastTriggerTime to simulate recent trigger (would normally cause cooldown)
  engine.lastTriggerTime = Date.now();

  // Confirmed turn trigger with skipCooldown: true
  await engine.handleSuggestionTrigger({
    context: question,
    lastQuestion: question,
    confidence: 0.9,
    skipCooldown: true,
  });

  assert.ok(generationCount >= 1, 'confirmed trigger with skipCooldown must bypass cooldown and generate');
});
