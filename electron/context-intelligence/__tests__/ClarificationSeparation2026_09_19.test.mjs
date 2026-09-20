// Phase 8 Step 17 — Clarification + Answer Separation (by construction).
//
// Verifies that the CLARIFICATION short-circuit in engine-bridge.ts prevents
// composePrompt() from running when a bare follow-up has an unresolvable
// referent (Type B), while leaving Type A (IB_CLARIFICATION_RE) untouched.
//
// Type A: "What do you mean by that?" → IB_CLARIFICATION_RE fires; isBareFollowUp
//         returns false → referentUnresolved = false → no short-circuit.
// Type B: "Why?" with isFollowUp:true and no prior state → isBareFollowUp = true,
//         referentWasResolved = false → fallbackUsed = 'CLARIFICATION' → short-circuit.

import { test, describe, beforeEach, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const base = path.resolve(process.cwd(), 'dist-electron/electron/context-intelligence');
const { buildV3Prompt } =
  await import(pathToFileURL(path.join(base, 'orchestration/engine-bridge.js')).href);
const { CONTEXT_INTELLIGENCE_V3_ENV_KEY } =
  await import(pathToFileURL(path.join(base, 'contracts/flag.js')).href);

const ENV = CONTEXT_INTELLIGENCE_V3_ENV_KEY;
const enable = () => { process.env[ENV] = '1'; };
afterEach(() => { delete process.env[ENV]; });

describe('Phase 8 Step 17: Clarification + Answer Separation', () => {
  beforeEach(enable);

  let _n = 0;
  const sid = () => `cs-sep-${++_n}-${Date.now()}`;

  const mkReq = (over = {}) => ({
    surface: 'manual-chat',
    modeTemplateType: 'technical-interview',
    ...over,
  });

  // T1: CLARIFICATION short-circuit omits all normal compose sections
  test('T1: bare follow-up system omits #Rules #Mode #Grounding #Capabilities', async () => {
    const result = await buildV3Prompt(mkReq({
      question: 'Why?',
      isFollowUp: true,
      scope: { sessionId: sid() },
    }));
    assert.ok(result, 'buildV3Prompt returned null — is V3 enabled?');
    assert.ok(!result.system.includes('# Rules'),
      `CLARIFICATION system must not include # Rules:\n${result.system}`);
    assert.ok(!result.system.includes('# Mode'),
      `CLARIFICATION system must not include # Mode:\n${result.system}`);
    assert.ok(!result.system.includes('# Grounding'),
      `CLARIFICATION system must not include # Grounding:\n${result.system}`);
    assert.ok(!result.system.includes('# Capabilities'),
      `CLARIFICATION system must not include # Capabilities:\n${result.system}`);
  });

  // T2: CLARIFICATION system contains the explicit clarifying-question directive
  test('T2: bare follow-up system contains clarifying-question directive', async () => {
    const result = await buildV3Prompt(mkReq({
      question: 'Why?',
      isFollowUp: true,
      scope: { sessionId: sid() },
    }));
    assert.ok(result, 'buildV3Prompt returned null');
    assert.ok(/clarifying question/i.test(result.system),
      `missing "clarifying question" in system:\n${result.system}`);
    assert.ok(/best guess/i.test(result.system),
      `missing "best guess" in system:\n${result.system}`);
    assert.ok(/Do NOT answer/i.test(result.system),
      `missing "Do NOT answer" in system:\n${result.system}`);
  });

  // T3: CLARIFICATION evidenceCount is exactly 0
  test('T3: bare follow-up evidenceCount is 0', async () => {
    const result = await buildV3Prompt(mkReq({
      question: 'Why?',
      isFollowUp: true,
      scope: { sessionId: sid() },
    }));
    assert.ok(result, 'buildV3Prompt returned null');
    assert.strictEqual(result.evidenceCount, 0);
  });

  // T4: Normal answerable question is NOT short-circuited — full compose runs
  test('T4: normal answerable question still has #Rules and #Grounding', async () => {
    const result = await buildV3Prompt(mkReq({
      question: 'How does Redis work?',
      isFollowUp: false,
      scope: { sessionId: sid() },
    }));
    assert.ok(result, 'buildV3Prompt returned null');
    assert.ok(result.system.includes('# Rules'),
      `# Rules missing from normal system:\n${result.system.slice(0, 400)}`);
    assert.ok(result.system.includes('# Grounding'),
      `# Grounding missing from normal system:\n${result.system.slice(0, 400)}`);
  });

  // T5: Type A IB_CLARIFICATION_RE question is NOT short-circuited.
  // "What do you mean by that?" matches IB_CLARIFICATION_RE (/\bwhat do you mean by\b/i)
  // but isBareFollowUp returns false (FOLLOW_UP_RE requires the string to START with
  // "why|how|can you|..." — this starts with "what do you"). So referentUnresolved
  // stays false → fallbackUsed != 'CLARIFICATION' → short-circuit never fires.
  test('T5: Type A clarification "What do you mean by that?" is not short-circuited — still has #Mode', async () => {
    const result = await buildV3Prompt(mkReq({
      question: 'What do you mean by that?',
      isFollowUp: true,
      scope: { sessionId: sid() },
    }));
    assert.ok(result, 'buildV3Prompt returned null');
    assert.ok(result.system.includes('# Mode'),
      `# Mode missing — Type A clarification was incorrectly short-circuited:\n${result.system.slice(0, 400)}`);
  });
});
