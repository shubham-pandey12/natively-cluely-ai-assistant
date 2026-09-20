// electron/context-intelligence/evaluation/answer-quality.eval.mjs
//
// Offline LLM answer-quality evaluation runner.
//
// NOT a .test.mjs file — excluded from the node --test CI glob intentionally.
// Run manually:  npm run eval:interview
//
// Requirements:
//   GEMINI_API_KEY=<key>     (required — same key used by the rest of the project)
//   EVAL_MODEL=<model>       (optional, defaults to gemini-3.1-flash-lite)
//   EVAL_DELAY_MS=<number>   (optional, inter-fixture delay in ms, defaults to 5000)
//                            Increase to 8000+ when hitting Gemini free-tier rate limits
//                            (free tier: 15 req/min; each fixture uses 2 API calls)
//   npm run build:electron   (dist-electron must be up to date)

import { GoogleGenAI } from '@google/genai';
import path from 'node:path';
import fs from 'node:fs';
import { pathToFileURL, fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ── Guard: require API key before doing anything else ────────────────────────

const API_KEY = process.env.GEMINI_API_KEY ?? '';
if (!API_KEY) {
  console.error(
    '\n[eval:interview] GEMINI_API_KEY is not set.\n' +
    'Export the key before running:\n' +
    '  export GEMINI_API_KEY=AIza...\n' +
    '  npm run eval:interview\n' +
    '\nThe runner requires a valid Gemini key to generate answers and run the judge.\n' +
    'No files have been modified. Exiting without error.\n',
  );
  process.exit(0); // graceful — no CI failure
}

// ── Load dist-electron modules ───────────────────────────────────────────────

const base = path.resolve(process.cwd(), 'dist-electron/electron/context-intelligence');

let decide, composePrompt, MODE_POLICIES, adaptLegacyChunks;
let advanceConversationState, clearConversationState, getConversationState, recordAnswerSummary;
try {
  ({ decide }             = await import(pathToFileURL(path.join(base, 'orchestration/orchestrator.js')).href));
  ({ composePrompt }      = await import(pathToFileURL(path.join(base, 'generation/prompt-composer.js')).href));
  ({ MODE_POLICIES }      = await import(pathToFileURL(path.join(base, 'policies/mode-policy-registry.js')).href));
  ({ adaptLegacyChunks }  = await import(pathToFileURL(path.join(base, 'retrieval/legacy-adapter.js')).href));
  ({ advanceConversationState, clearConversationState, getConversationState, recordAnswerSummary } =
    await import(pathToFileURL(path.join(base, 'question/conversation-state-store.js')).href));
} catch (err) {
  console.error(
    '\n[eval:interview] Failed to load dist-electron modules.\n' +
    'Run  npm run build:electron  first.\n\n' + err.message + '\n',
  );
  process.exit(1);
}

// ── Numeric provenance validator (Phase 8 Step 21) ───────────────────────────
// Loaded separately so a missing build artifact degrades gracefully.
let validateNumericProvenance;
try {
  ({ validateNumericProvenance } = await import(pathToFileURL(path.join(base, 'validation/numeric-provenance.js')).href));
} catch {
  console.warn('[eval:interview] numeric-provenance validator not available — np checks disabled');
}

const POLICY = MODE_POLICIES['technical-interview'];

// ── Shared evaluation scope ───────────────────────────────────────────────────

const EVAL_SCOPE = { userId: 'eval', modeId: 'technical-interview' };

// ── Mock evidence infrastructure (Phase 8 Step 9, 2026-09-19) ────────────────
//
// Fixtures that test stories=true intents (behavioral, introduction,
// experience_question, project_context, project_deep_dive, technology_decision)
// must supply fictional evidence so composePrompt grounds the answer instead of
// firing noEvidenceNotice.
//
// mockEvidence fixture field: Array of MockEvidenceDescriptor
//   { sourceId: string, sourceType: SourceType, text: string,
//     chunkIndex?: number, score?: number }
//
// All content must be entirely fictional — invented company names, invented
// metrics, invented tech stack. Never real user profile data.
//
// When the field is absent or an empty array, evidence = [] exactly as before.
// No existing fixture has this field, so existing evaluator behavior is unchanged.

function buildMockEvidence(descriptors) {
  if (!descriptors || descriptors.length === 0) return [];
  const MOCK_VERSION   = 'mock-v1';
  const sourceTypes    = new Map(descriptors.map((d) => [d.sourceId, d.sourceType]));
  const activeVersions = new Map(descriptors.map((d) => [d.sourceId, MOCK_VERSION]));
  const chunkVersions  = new Map(descriptors.map((d) => [d.sourceId, MOCK_VERSION]));
  const chunks = descriptors.map((d, i) => ({
    sourceId:   d.sourceId,
    text:       d.text,
    chunkIndex: d.chunkIndex ?? i,
    score:      d.score ?? 0.9,
  }));
  const { evidence } = adaptLegacyChunks(chunks, {
    scope: EVAL_SCOPE,
    sourceTypes,
    activeVersions,
    chunkVersions,
    assumeInScopeWhenUnknown: true,
  });
  return evidence;
}

// ── Multi-turn helpers ────────────────────────────────────────────────────────

// formatChainForEval: mirrors engine-bridge.ts formatTopicChain() which is private
// (not exported from engine-bridge.js). Produces the same string the live production
// path passes as conversationSummary to composePrompt.
function formatChainForEval(chain) {
  if (!chain || chain.length === 0) return '';
  const last = chain[chain.length - 1];
  if (chain.length === 1) {
    let s = `Previous question: ${last.question}`;
    if (last.answerSummary) s += `\nPrevious answer (referent only, NOT evidence): ${last.answerSummary}`;
    return s;
  }
  const lines = chain.map((t, i) => `[${i + 1}] ${t.question}`);
  let s = `Conversation chain (${chain.length} turns):\n${lines.join('\n')}`;
  if (last.answerSummary) s += `\nPrevious answer (referent only, NOT evidence): ${last.answerSummary}`;
  return s;
}

// setupPriorTurns: establish prior turns in the production conversation-state store
// before evaluating the final fixture question. Uses the same functions that
// orchestrate() calls after each turn — advanceConversationState + recordAnswerSummary.
// No Gemini call is made for prior turns; syntheticAnswer is used directly.
function setupPriorTurns(fixture, sessionId) {
  clearConversationState(sessionId);
  for (const pt of fixture.priorTurns) {
    const d = decide({
      requestId: `${fixture.id}-prior`, requestSequence: 0,
      surface: 'manual_chat', modeId: 'technical-interview',
      scope: EVAL_SCOPE, sessionId,
      manualQuestion: pt.question,
    });
    advanceConversationState({
      sessionId,
      scope: EVAL_SCOPE,
      question: pt.question,
      interviewIntent: d.interviewIntent,
    });
    recordAnswerSummary(sessionId, pt.syntheticAnswer);
  }
  const state = getConversationState(sessionId);
  const chain = state?.topicChain ?? [];
  if (chain.length !== fixture.priorTurns.length) {
    console.error(
      `  [${fixture.id}] WARN: expected chain length ${fixture.priorTurns.length}, got ${chain.length}`,
    );
  }
  return chain;
}

// ── Models ───────────────────────────────────────────────────────────────────

// gemini-3.1-flash-lite is the project's benchmark model (see package.json benchmark:* scripts).
const GENERATION_MODEL = process.env.EVAL_MODEL       ?? 'gemini-3.1-flash-lite';
const JUDGE_MODEL      = process.env.EVAL_JUDGE_MODEL ?? 'gemini-3.1-flash-lite';
// Inter-fixture delay to stay within Gemini free-tier rate limit (15 req/min).
// Each fixture uses 2 API calls (generate + judge). At 5 s between fixtures the
// effective rate is ~20 calls/min on fast hardware; increase EVAL_DELAY_MS to
// 8000 when hitting 429 errors (24 calls across ~100 s ≈ 14.4 calls/min).
const EVAL_DELAY_MS    = Number(process.env.EVAL_DELAY_MS ?? '5000');

// ── Fixtures ─────────────────────────────────────────────────────────────────

const FIXTURES = [
  {
    id: 'aq_001',
    question: 'What is a closure?',
    expectedIntent: 'concept_explanation',
    expectedFollowUpLikelihood: 'high',
    expectedDepth: 'standard',
    requiredDimensions: ['voice', 'strategy_adherence', 'depth', 'pacing'],
  },
  {
    id: 'aq_002',
    question: 'How does garbage collection work in JavaScript?',
    expectedIntent: 'mechanism_explanation',
    expectedFollowUpLikelihood: 'high',
    expectedDepth: 'standard',
    requiredDimensions: ['voice', 'strategy_adherence', 'depth', 'pacing'],
  },
  {
    id: 'aq_003',
    question: 'Implement a function to detect a cycle in a linked list.',
    expectedIntent: 'coding_task',
    expectedFollowUpLikelihood: 'high',
    expectedDepth: 'standard',
    requiredDimensions: ['voice', 'strategy_adherence', 'depth', 'pacing'],
  },
  {
    id: 'aq_004',
    question: 'Design a URL shortener at scale.',
    expectedIntent: 'system_design',
    expectedFollowUpLikelihood: 'high',
    expectedDepth: 'deep',
    requiredDimensions: ['voice', 'strategy_adherence', 'depth', 'pacing'],
  },
  {
    id: 'aq_005',
    // Replaced "Tell me about a time you had to meet a tight deadline."
    // (behavioral, requires personal evidence → noEvidenceNotice fires correctly but
    // makes the fixture ungradeable without mock evidence). Replaced with a mechanism
    // question that exercises explain_mechanism and needs no personal evidence.
    question: 'How does the JavaScript event loop work?',
    expectedIntent: 'mechanism_explanation',
    expectedFollowUpLikelihood: 'high',
    expectedDepth: 'standard',
    requiredDimensions: ['voice', 'strategy_adherence', 'depth', 'pacing'],
  },
  {
    id: 'aq_006',
    // Replaced "Are you familiar with Kubernetes?" (knowledge_check with
    // personal familiarity claim → noEvidenceNotice fires, ungradeable without evidence).
    // Replaced with a pure knowledge question that routes to concept_explanation with
    // no personal-evidence requirement and exercises the define_concept strategy.
    question: 'What is Kubernetes?',
    expectedIntent: 'concept_explanation',
    expectedFollowUpLikelihood: 'high',
    expectedDepth: 'standard',
    requiredDimensions: ['voice', 'strategy_adherence', 'depth', 'pacing'],
  },
  {
    id: 'aq_007',
    // Phase 5 Step 4A: corrected from 'comparison' to 'tradeoff'.
    // "tradeoffs" in the question matches /\b(?:tradeoffs?|…)\b/i at
    // turn-classifier.ts:1354 before the comparison branch at :1352.
    question: 'What are the tradeoffs between SQL and NoSQL databases?',
    expectedIntent: 'tradeoff',
    expectedFollowUpLikelihood: 'high',
    expectedDepth: 'standard',
    requiredDimensions: ['voice', 'strategy_adherence', 'depth', 'pacing'],
  },
  {
    id: 'aq_008',
    // Original "Why is my React component re-rendering too many times?"
    // triggered AI-assistant response mode (voice=1). Rephrased to interviewer framing.
    // "What are common causes..." (second version) got strat=3 because TRACE_BUG is
    // procedural (symptom → hypotheses → diagnose → fix) but the question asked for
    // general knowledge. Replaced with a scenario-based question that naturally exercises
    // the full TRACE_BUG diagnostic protocol — specific symptom, interviewer framing.
    question: 'A React component re-renders on every keystroke even though its props have not changed. Walk me through how you would debug this.',
    expectedIntent: 'debugging',
    expectedFollowUpLikelihood: 'medium',
    expectedDepth: 'standard',
    requiredDimensions: ['voice', 'strategy_adherence', 'depth'],
    notes: 'Medium likelihood — pacing not checked',
  },
  {
    id: 'aq_009',
    // Phase 5 GAP-2: expectedIntent corrected from 'scalability' to 'system_design'.
    // "How would you scale X to N users" matches SYSTEM_DESIGN_RE's `scale (a|the|to)`
    // pattern before the scalability branch is reached (turn-classifier.ts:377 vs :1358).
    // The classifier correctly returns 'system_design' for this phrasing; the fixture was wrong.
    question: 'How would you scale a social media feed to 100 million daily active users?',
    expectedIntent: 'system_design',
    expectedFollowUpLikelihood: 'high',
    expectedDepth: 'deep',
    requiredDimensions: ['voice', 'strategy_adherence', 'depth', 'pacing'],
  },
  {
    id: 'aq_010',
    // Replaced "Tell me about yourself." (introduction → requires resume
    // evidence for a meaningful self-intro; noEvidenceNotice fires correctly but makes
    // the fixture ungradeable). Replaced with a tradeoff question that exercises
    // analyze_options with no personal evidence requirement.
    question: 'What are the tradeoffs of using TypeScript over JavaScript?',
    expectedIntent: 'tradeoff',
    expectedFollowUpLikelihood: 'high',
    expectedDepth: 'standard',
    requiredDimensions: ['voice', 'strategy_adherence', 'depth', 'pacing'],
  },
  {
    id: 'aq_011',
    // Replaced "Walk me through a challenging technical project you have
    // worked on." (experience_question → requires personal evidence; actual classifier
    // routing was experience_question via EXPERIENCE_CHALLENGE_RE, and the fixture's
    // expectedIntent 'project_context' was also wrong). Replaced with an LLD question
    // that exercises design_classes with no personal evidence requirement.
    question: 'Design a parking lot with object-oriented classes.',
    expectedIntent: 'lld',
    expectedFollowUpLikelihood: 'high',
    expectedDepth: 'deep',
    requiredDimensions: ['voice', 'strategy_adherence', 'depth', 'pacing'],
  },
  {
    id: 'aq_012',
    question: 'How would you optimize a slow database query?',
    expectedIntent: 'optimization',
    expectedFollowUpLikelihood: 'medium',
    expectedDepth: 'standard',
    requiredDimensions: ['voice', 'strategy_adherence', 'depth'],
    notes: 'Optimization intent — medium likelihood',
  },

  // ── Phase 5 GAP-3: evidence-free intent coverage (aq_013–aq_016) ─────────────
  // NOTE: aq_007 is labeled expectedIntent:'comparison' but actually routes to
  // 'tradeoff' ("tradeoffs" matches the tradeoff branch before comparison).
  // That mislabel is a discovered problem reported separately; do not fix here.
  // aq_013 adds genuine comparison coverage.
  {
    id: 'aq_013',
    question: 'What is the difference between REST and GraphQL?',
    expectedIntent: 'comparison',
    expectedFollowUpLikelihood: 'high',
    expectedDepth: 'standard',
    requiredDimensions: ['voice', 'strategy_adherence', 'depth', 'pacing'],
  },
  {
    id: 'aq_014',
    // knowledge_check: depth=brief, followUpLikelihood=low, stories=false.
    // "Are you comfortable with X?" triggers the knowledge_check branch (line 1369
    // of turn-classifier.ts). No personal evidence required — the model produces a
    // "yes + brief explanation" pattern without resume claims.
    question: 'Are you comfortable with async/await in JavaScript?',
    expectedIntent: 'knowledge_check',
    expectedFollowUpLikelihood: 'low',
    expectedDepth: 'brief',
    requiredDimensions: ['voice', 'strategy_adherence', 'depth', 'pacing'],
  },
  // aq_015 removed: technology_decision has contextRequirements.stories=true.
  // With evidence:[], the prompt fires noEvidenceNotice — a hard production refusal
  // directing the user to add their profile. This is correct production behavior and
  // is not a gradeable answer-quality failure. A technology_decision fixture requires
  // StoryBank evidence (mock or real) to produce a gradeable answer; adding fabricated
  // personal history would violate PERMANENT_RULES. Removed rather than silenced.
  {
    id: 'aq_016',
    // scalability: depth=deep, followUpLikelihood=high. Uses "traffic grows" phrasing
    // which routes to scalability (turn-classifier.ts:1358). Avoids "scale (a|the|to)"
    // which would route to system_design via SYSTEM_DESIGN_RE instead.
    // Question rephrased (Phase 5 Step 6): original "What strategies would you apply when
    // traffic grows by 10x?" gave no baseline numbers, making analyze_scale step 1
    // ("Restate the scale target — DAU, QPS, storage") semantically unfulfillable.
    // The new phrasing gives a concrete starting point (5K RPS, monolithic + single DB)
    // so the model can anchor step 1, identify bottlenecks (step 2), and quantify
    // headroom (step 3) without fabricating numbers.
    question: 'Your service handles around 5,000 requests per second on a monolithic backend with a single database. Traffic is expected to grow 10x over the next quarter. What would you do?',
    expectedIntent: 'scalability',
    expectedFollowUpLikelihood: 'high',
    expectedDepth: 'deep',
    requiredDimensions: ['voice', 'strategy_adherence', 'depth', 'pacing'],
  },

  // ── Phase 8 Step 10: evidence-backed fixtures for stories=true intents ────────
  // All six intents below require personal evidence (contextRequirements.stories=true).
  // Without mockEvidence the prompt fires noEvidenceNotice — the correct production
  // behavior when no StoryBank is present, but ungradeable. Each fixture carries
  // entirely fictional evidence (Helix Commerce / Orion project — invented company,
  // invented project, invented metrics) sufficient for the strategy to answer from.
  //
  // Source-type note:
  //   RESUME     → authoritative for USER_EMPLOYMENT, USER_PROJECT, USER_SKILL, etc.
  //   PROFILE_FACT → additionally authoritative for USER_MOTIVATION (prohibited on RESUME).
  //   technology_decision uses PROFILE_FACT because it noteClaims USER_MOTIVATION.
  {
    id: 'aq_018',
    // behavioral: BEHAVIORAL_FRAMING_RE ("tell me about a time") fires at
    // turn-classifier.ts:1358 → intent=behavioral, strategy=tell_behavioral_story.
    // followUpLikelihood=medium (not in HIGH or LOW literal arrays).
    // Pacing not required: medium likelihood makes pass/fail ambiguous.
    question: 'Tell me about a time you had to convince your team to adopt a different technical approach.',
    expectedIntent: 'behavioral',
    expectedStrategy: 'tell_behavioral_story',
    expectedFollowUpLikelihood: 'medium',
    expectedDepth: 'standard',
    requiredDimensions: ['voice', 'strategy_adherence', 'depth'],
    notes: 'Medium follow-up likelihood — pacing not checked',
    mockEvidence: [
      {
        sourceId:   'mock-resume-behavioral-1',
        sourceType: 'RESUME',
        text:
          'At Helix Commerce (fictional B2B SaaS), I proposed replacing our polling-based ' +
          'inventory sync with Kafka for the Orion order-processing service. The team was ' +
          'sceptical — none of us had operated Kafka in production, and the on-call overhead ' +
          'worried two senior engineers. I volunteered to build a proof of concept in one week, ' +
          'wrote a short operations runbook covering common failure modes, and ran a load-test ' +
          'demo showing a 4× reduction in inventory desync events under simulated peak traffic. ' +
          'After reviewing the runbook and demo results, the team agreed to proceed. We shipped ' +
          'Kafka-based async inventory sync six weeks later; desync incidents dropped from ' +
          'roughly three per week to fewer than one per month.',
        chunkIndex: 0,
        score: 0.92,
      },
    ],
  },
  {
    id: 'aq_019',
    // introduction: INTRODUCTION_RE ("tell me about yourself") fires at
    // turn-classifier.ts:1360 → intent=introduction, strategy=introduce_self.
    // followUpLikelihood=low (in LOW_LIKELIHOOD literal array).
    question: 'Tell me about yourself.',
    expectedIntent: 'introduction',
    expectedStrategy: 'introduce_self',
    expectedFollowUpLikelihood: 'low',
    expectedDepth: 'standard',
    requiredDimensions: ['voice', 'strategy_adherence', 'depth', 'pacing'],
    mockEvidence: [
      {
        sourceId:   'mock-resume-intro-1',
        sourceType: 'RESUME',
        text:
          'Senior Backend Engineer at Helix Commerce (fictional B2B SaaS) for two years. ' +
          'Prior to that, three years as a full-stack engineer at two early-stage startups ' +
          'building customer-facing APIs and learning to tune PostgreSQL under concurrent write loads. ' +
          'At Helix Commerce I moved fully into backend infrastructure. Most significant project: ' +
          'Orion — an order-processing microservice I led from architecture to production. Orion ' +
          'reduced checkout p99 latency from 1.2 s to 340 ms and now handles roughly 2,000 orders ' +
          'per day. Currently looking for a role focused on distributed systems at larger scale ' +
          'where I can deepen this experience.',
        chunkIndex: 0,
        score: 0.93,
      },
    ],
  },
  {
    id: 'aq_020',
    // experience_question: EXPERIENCE_CHALLENGE_RE ("tell me about a challenging...")
    // fires at turn-classifier.ts:1364 → intent=experience_question,
    // strategy=narrate_experience. followUpLikelihood=medium.
    question: 'Tell me about a challenging production incident you had to resolve.',
    expectedIntent: 'experience_question',
    expectedStrategy: 'narrate_experience',
    expectedFollowUpLikelihood: 'medium',
    expectedDepth: 'standard',
    requiredDimensions: ['voice', 'strategy_adherence', 'depth'],
    notes: 'Medium follow-up likelihood — pacing not checked',
    mockEvidence: [
      {
        sourceId:   'mock-resume-experience-1',
        sourceType: 'RESUME',
        text:
          'During the Black Friday load peak at Helix Commerce (fictional), the Orion checkout ' +
          'service began timing out under concurrent order submissions. I pulled pg_stat_activity ' +
          'and found hundreds of transactions waiting on row-level locks in the inventory ' +
          'reservation step — a deadlock cycle between two queries updating the same inventory ' +
          'row in different lock orders. Root cause: the reservation query did not enforce a ' +
          'consistent row-ID ordering across concurrent sessions. I deployed a fix that imposed ' +
          'ascending row-ID lock order across all callers. Within ten minutes of the deploy, ' +
          'checkout timeout errors dropped from roughly 800 per minute to near zero. ' +
          'p99 latency recovered from 4.1 s to under 400 ms. I followed up by adding a ' +
          'pg_stat_activity-based deadlock alert so we would catch recurrences before users did.',
        chunkIndex: 0,
        score: 0.91,
      },
    ],
  },
  {
    id: 'aq_021',
    // project_context: "your project" triggers PERSONAL_RE + PROJECT_RE →
    // PERSONAL_PROJECT at turn-classifier.ts:665. No PROJECT_DEEP_RE or "why" →
    // falls through to intent=project_context at line 1372,
    // strategy=describe_project. followUpLikelihood=medium.
    question: 'Walk me through your most recent project and the role you played in it.',
    expectedIntent: 'project_context',
    expectedStrategy: 'describe_project',
    expectedFollowUpLikelihood: 'medium',
    expectedDepth: 'standard',
    requiredDimensions: ['voice', 'strategy_adherence', 'depth'],
    notes: 'Medium follow-up likelihood — pacing not checked',
    mockEvidence: [
      {
        sourceId:   'mock-resume-project-1',
        sourceType: 'RESUME',
        text:
          'Most recent project: Orion — order and inventory management microservice at Helix ' +
          'Commerce (fictional). Tech lead and primary implementer, working with one junior ' +
          'engineer. Orion replaced a monolithic checkout module that had become the primary ' +
          'reliability bottleneck. Built in Node.js with TypeScript, backed by PostgreSQL for ' +
          'order state and Redis for inventory read cache. Async inventory reconciliation runs ' +
          'over Kafka. My role covered architecture design, database schema, API contract, Kafka ' +
          'topic design, and all production hardening. Core challenge: maintaining consistency ' +
          'across the synchronous order placement path and the asynchronous inventory update path. ' +
          'Resolved using a reservation-then-confirm pattern with an outbox table. Orion handles ' +
          'roughly 2,000 orders per day with a p99 checkout latency of 340 ms.',
        chunkIndex: 0,
        score: 0.93,
      },
    ],
  },
  {
    id: 'aq_022',
    // project_deep_dive: PROJECT_DEEP_RE ("how did you handle") AND PERSONAL_PROJECT
    // ("did you" via PERSONAL_RE, "built" via PROJECT_RE) →
    // intent=project_deep_dive at turn-classifier.ts:1370, strategy=describe_project.
    // followUpLikelihood=medium. Depth=deep: deep-dive expects comprehensive detail.
    question: 'How did you handle database reliability in the service you built?',
    expectedIntent: 'project_deep_dive',
    expectedStrategy: 'describe_project',
    expectedFollowUpLikelihood: 'medium',
    expectedDepth: 'deep',
    requiredDimensions: ['voice', 'strategy_adherence', 'depth'],
    notes: 'Medium follow-up likelihood — pacing not checked. Depth=deep: deep-dive warrants comprehensive detail.',
    mockEvidence: [
      {
        sourceId:   'mock-resume-deepdive-1',
        sourceType: 'RESUME',
        text:
          'In the Orion service at Helix Commerce (fictional), database reliability used three layers. ' +
          'First, all order-placement writes ran inside a serializable transaction covering both the ' +
          'inventory reservation and the order record insert, preventing partial writes under any ' +
          'crash scenario. Second, Kafka-consumer inventory updates used optimistic locking: each ' +
          'inventory row carried a version counter, and a conflicting update retried up to five times ' +
          'with exponential back-off before failing with a structured error rather than silently ' +
          'dropping the update. Third, an outbox table decoupled order confirmation from Kafka ' +
          'publish: if the broker was unreachable the order still committed atomically, and a ' +
          'background reconciler replayed the outbox row when the broker recovered. During a ' +
          'four-hour Kafka outage in staging, zero order records were lost. We also ran weekly ' +
          'chaos drills that killed the PostgreSQL primary under active load to verify automatic ' +
          'failover behaviour.',
        chunkIndex: 0,
        score: 0.91,
      },
    ],
  },
  {
    id: 'aq_023',
    // technology_decision: TECH_DECISION_FRAMING_RE ("why did you choose") AND
    // TECH_DECISION_SUBJECT_RE ("PostgreSQL", "MongoDB") both match →
    // intent=technology_decision at turn-classifier.ts:1366, strategy=justify_decision.
    // followUpLikelihood=high (in HIGH_LIKELIHOOD literal array).
    // Source type = PROFILE_FACT (not RESUME): technology_decision noteClaims
    // USER_MOTIVATION, which prohibits RESUME as an authoritative source
    // (source-authority-policy.ts:55). PROFILE_FACT covers both USER_MOTIVATION
    // and USER_PROJECT.
    question: 'Why did you choose PostgreSQL over MongoDB for your project?',
    expectedIntent: 'technology_decision',
    expectedStrategy: 'justify_decision',
    expectedFollowUpLikelihood: 'high',
    expectedDepth: 'standard',
    requiredDimensions: ['voice', 'strategy_adherence', 'depth', 'pacing'],
    mockEvidence: [
      {
        sourceId:   'mock-profile-techd-1',
        sourceType: 'PROFILE_FACT',
        text:
          'For the Orion order-processing service at Helix Commerce (fictional), chose PostgreSQL ' +
          'over MongoDB. Deciding factor: checkout required atomic writes across two tables — the ' +
          'orders table and the inventory reservation table — in a single ACID transaction. At the ' +
          'version of MongoDB available to the team, multi-document cross-collection transactions ' +
          'were either unavailable or insufficiently mature for production use. PostgreSQL row-level ' +
          'locking also fit the reservation-then-confirm pattern directly: we could lock specific ' +
          'inventory rows during checkout without blocking the entire collection. Main tradeoff: ' +
          'schema changes required coordinated migrations, whereas MongoDB would have allowed ' +
          'field additions without schema overhead. Given the stable domain model and the hard ' +
          'consistency requirement on order placement, PostgreSQL was the correct call for this context.',
        chunkIndex: 0,
        score: 0.94,
      },
    ],
  },

  // ── Phase 5 Step 5: follow_up_generic multi-turn fixture (aq_017) ─────────────
  // Uses priorTurns to establish conversation state via the production state-store
  // functions (advanceConversationState + recordAnswerSummary) before evaluating
  // the DEEPENING turn. No Gemini call is made for the prior turn — syntheticAnswer
  // is used directly. Pacing is not a required dimension: low followUpLikelihood
  // ("complete and direct") conflicts with DEEPENING elaboration behavior; the judge
  // would produce unreliable pacing verdicts. strategy_adherence covers the
  // deepening_elaboration structure instead.
  {
    id: 'aq_017',
    // "elaborate" triggers IB_DEEPENING_RE → DEEPENING → isOverrideBehavior=true
    // → intent=follow_up_generic (turn-classifier.ts:1318-1319).
    question: 'Can you elaborate on the memory cost tradeoff?',
    expectedIntent: 'follow_up_generic',
    expectedFollowUpLikelihood: 'low',
    expectedDepth: 'standard',
    requiredDimensions: ['voice', 'strategy_adherence', 'depth'],
    notes: 'Multi-turn: prior turn established via priorTurns. Pacing omitted — low followUpLikelihood conflicts with DEEPENING elaboration.',
    priorTurns: [
      {
        question: 'Why would you choose Redis for the URL shortener?',
        syntheticAnswer:
          "I'd use Redis mainly for the hot read path because URL lookups are frequent " +
          "and predictable. The main tradeoff is memory cost, so I'd keep the durable " +
          'mapping in a database.',
      },
    ],
  },

  // ── Phase 8 Step 21: Numeric Provenance evaluation fixtures ──────────────────
  // These five fixtures specifically exercise the numeric provenance validator.
  // Authoritative sources = [fixture.question, composed.user] — mirrors Step 20 wiring.
  // numericProvenanceExpected: true  = expect valid===true
  //                            null  = observation only (no assertion)
  {
    id: 'np_01',
    // NP-01: question contains "5,000" → question is an authoritative source.
    // Gemini repeating this number should produce valid===true.
    question: 'The service handles 5,000 requests per second. How would you scale it if traffic increased by 10x?',
    expectedIntent: 'scalability',
    expectedFollowUpLikelihood: 'high',
    expectedDepth: 'deep',
    requiredDimensions: ['voice', 'strategy_adherence', 'depth', 'pacing'],
    numericProvenanceExpected: true,
    notes: 'NP-01: 5,000 in question (authoritative source) — Gemini repeating it should be valid',
  },
  {
    id: 'np_02',
    // NP-02: question contains 99.9% and 340ms.
    // p99 is excluded by the letter-before-digit pattern.
    // "99.9" and "340" are factual numbers; both come from the question.
    // Gemini repeating them should be valid.
    question: 'The service has 99.9% availability and p99 latency of 340ms. How would you improve its reliability?',
    expectedIntent: 'optimization',
    expectedFollowUpLikelihood: 'medium',
    expectedDepth: 'standard',
    requiredDimensions: ['voice', 'strategy_adherence', 'depth'],
    numericProvenanceExpected: true,
    notes: 'NP-02: 99.9 and 340 in question; p99 excluded by pattern — repeating them should be valid',
  },
  {
    id: 'np_03',
    // NP-03: structural/technical numbers in likely answer: Node.js 20, port 3000,
    // Step 1, HTTP 200, O(n log n). All excluded by EXCLUSION_PATTERNS.
    // Expected: valid===true (no factual claims to validate).
    question: 'Walk me through how you would set up a Node.js 20 API server with HTTP rate limiting.',
    expectedIntent: 'system_design',
    expectedFollowUpLikelihood: 'medium',
    expectedDepth: 'standard',
    requiredDimensions: ['voice', 'strategy_adherence', 'depth'],
    numericProvenanceExpected: true,
    notes: 'NP-03: structural numbers (Node.js 20, port 3000, Step 1, HTTP 200) all excluded → valid',
  },
  {
    id: 'np_04',
    // NP-04: question contains "5,000" as the only authoritative metric.
    // Gemini may introduce other factual numbers (e.g., cache hit ratios, TTLs,
    // memory sizes) not grounded in the source. Purpose: observation only.
    question: 'The service currently handles 5,000 requests per second. How would you design a caching layer to reduce database load?',
    expectedIntent: 'system_design',
    expectedFollowUpLikelihood: 'high',
    expectedDepth: 'deep',
    requiredDimensions: ['voice', 'strategy_adherence', 'depth', 'pacing'],
    numericProvenanceExpected: null,
    notes: 'NP-04: observation — Gemini may introduce factual numbers not in the source (e.g., TTL values, hit rates)',
  },
  {
    id: 'np_05',
    // NP-05: conceptual comparison; answer should not require factual numeric claims.
    // Expected: valid===true (no factual numbers extracted, or only structural ones).
    question: 'What is the difference between a thread and a process?',
    expectedIntent: 'comparison',
    expectedFollowUpLikelihood: 'high',
    expectedDepth: 'standard',
    requiredDimensions: ['voice', 'strategy_adherence', 'depth', 'pacing'],
    numericProvenanceExpected: true,
    notes: 'NP-05: no factual numeric claims expected in a conceptual comparison answer → valid',
  },
];

// ── Gemini helpers ────────────────────────────────────────────────────────────

function extractText(response) {
  // Method 1: direct response.text (string property on the SDK result object)
  if (typeof response.text === 'string' && response.text.length > 0) return response.text;
  // Method 2: SDK accessor (some SDK versions expose response.text as a function)
  if (typeof response.text === 'function') {
    try { const t = response.text(); if (t) return t; } catch {}
  }
  // Method 3: candidates array
  const candidate = response.candidates?.[0];
  if (!candidate) return '';
  const parts = candidate.content?.parts;
  if (Array.isArray(parts)) return parts.map((p) => p?.text ?? '').join('');
  if (typeof candidate.content === 'string') return candidate.content;
  return '';
}

async function geminiGenerate(client, model, systemInstruction, userMessage, maxOutputTokens = 512) {
  const response = await client.models.generateContent({
    model,
    contents: [{ role: 'user', parts: [{ text: userMessage }] }],
    config: {
      systemInstruction: { parts: [{ text: systemInstruction }] },
      maxOutputTokens,
      temperature: 0.4,
    },
  });
  return extractText(response);
}

// ── Deterministic pre-filter ─────────────────────────────────────────────────

const WRONG_VOICE_RE   = /\bas an ai\b|\bi'?d be happy to\b|\bcertainly!|\bgreat question!|\bof course!|\babsolutely!|\bi cannot provide\b|\bi'?m unable to\b/i;
const TEMPLATE_LEAK_RE = /^#{2,3}\s/m;
const FABRICATION_RE   = /\bin my (?:previous|current|last|former) (?:role|job|company|team)\b|\bwhen i (?:built|created|developed|implemented|worked on)\b|\bat my (?:previous|current|last) (?:company|employer|job)\b/i;

const GENERAL_KNOWLEDGE_INTENTS = new Set([
  'concept_explanation', 'mechanism_explanation', 'coding_task',
  'comparison', 'tradeoff', 'knowledge_check', 'debugging', 'optimization',
]);

function deterministicPreFilter(answer, fixture, actualIntent) {
  const flags = [];

  if (WRONG_VOICE_RE.test(answer)) flags.push('wrong_voice');

  if (fixture.expectedIntent !== 'coding_task' && TEMPLATE_LEAK_RE.test(answer)) {
    flags.push('template_leak');
  }

  const wordCount = answer.trim().split(/\s+/).length;
  if (fixture.expectedDepth === 'deep' && wordCount < 30) flags.push('too_brief');

  const intentForCheck = actualIntent || fixture.expectedIntent;
  if (GENERAL_KNOWLEDGE_INTENTS.has(intentForCheck) && FABRICATION_RE.test(answer)) {
    flags.push('fabricated_claim');
  }

  return flags;
}

// ── LLM judge ────────────────────────────────────────────────────────────────

const JUDGE_SYSTEM =
  'You are an expert technical interview answer evaluator. ' +
  'Your job is to grade an AI-generated answer against a rubric. ' +
  'Respond with a JSON object only — no prose, no markdown code fences.';

function buildJudgeUserMessage(fixture, actualIntent, steps, answer) {
  const stepList = Array.isArray(steps) && steps.length
    ? steps.map((s, i) => `  ${i + 1}. ${s}`).join('\n')
    : '  (strategy steps not available)';

  return (
    `Intent: ${actualIntent || fixture.expectedIntent}\n` +
    `Expected depth: ${fixture.expectedDepth} (brief=1-2 sentences, standard=moderate, deep=comprehensive)\n` +
    `Follow-up likelihood: ${fixture.expectedFollowUpLikelihood} ` +
    `(high=leave clear entry points, low=answer directly and completely, medium=neutral)\n\n` +
    `Strategy steps the answer should follow:\n${stepList}\n\n` +
    `Rate the answer on these dimensions:\n\n` +
    `voice (1-5):\n` +
    `  5 = sounds like a person thinking out loud mid-conversation\n` +
    `  4 = natural, occasional written-text feel\n` +
    `  3 = borderline — some natural flow, some formal writing\n` +
    `  2 = mostly reads like documentation or an AI assistant\n` +
    `  1 = clearly AI: "Certainly!", meta-commentary, or over-formal tone\n\n` +
    `strategy_adherence (1-5):\n` +
    `  5 = follows the strategy steps in the correct order with appropriate detail\n` +
    `  4 = mostly follows, minor omissions\n` +
    `  3 = partial — some steps missing or out of order\n` +
    `  2 = little resemblance to the strategy steps\n` +
    `  1 = ignores the strategy completely\n\n` +
    `depth: "pass" or "fail"\n` +
    `  pass = depth matches expected level\n` +
    `  fail = too long for brief, too short for deep, or otherwise wrong depth\n\n` +
    `pacing: "pass" or "fail"\n` +
    `  pass = pacing matches follow-up likelihood\n` +
    `    (high: thorough but leaves entry points; low: complete and direct; medium: neutral)\n` +
    `  fail = wrong pacing for the follow-up likelihood\n\n` +
    `Respond with ONLY this JSON (no markdown, no prose):\n` +
    `{"voice":<1-5>,"strategy_adherence":<1-5>,"depth":"pass"|"fail","pacing":"pass"|"fail","explanation":"<what failed, or all pass>"}\n\n` +
    `Question: ${fixture.question}\n\n` +
    `Answer to evaluate:\n${answer}`
  );
}

async function runJudge(client, fixture, actualIntent, steps, answer) {
  const raw = await geminiGenerate(
    client, JUDGE_MODEL, JUDGE_SYSTEM,
    buildJudgeUserMessage(fixture, actualIntent, steps, answer),
    256,
  );
  // Strip any accidental markdown fence
  const jsonText = raw.replace(/^```(?:json)?\n?/, '').replace(/\n?```$/, '').trim();
  const parsed = JSON.parse(jsonText);
  return {
    voice:              Number(parsed.voice),
    strategy_adherence: Number(parsed.strategy_adherence),
    depth:              parsed.depth === 'pass' ? 'pass' : 'fail',
    pacing:             parsed.pacing === 'pass' ? 'pass' : 'fail',
    explanation:        String(parsed.explanation ?? ''),
  };
}

// ── Grading logic ─────────────────────────────────────────────────────────────

function grade(fixture, preFilterFlags, judgeScores) {
  if (
    preFilterFlags.includes('wrong_voice') ||
    preFilterFlags.includes('template_leak') ||
    preFilterFlags.includes('fabricated_claim')
  ) {
    return 'FAIL';
  }

  if (!judgeScores) return preFilterFlags.length === 0 ? 'PASS' : 'FAIL';

  const required = new Set(fixture.requiredDimensions);
  if (required.has('voice')              && judgeScores.voice              < 4)       return 'FAIL';
  if (required.has('strategy_adherence') && judgeScores.strategy_adherence < 4)       return 'FAIL';
  if (required.has('depth')             && judgeScores.depth              !== 'pass') return 'FAIL';
  if (required.has('pacing')            && judgeScores.pacing             !== 'pass') return 'FAIL';
  if (required.has('grounding')         && judgeScores.grounding          !== 'pass') return 'FAIL';
  return 'PASS';
}

// ── Runner ───────────────────────────────────────────────────────────────────

async function run() {
  const client = new GoogleGenAI({ apiKey: API_KEY });

  console.log(`\n[eval:interview] LLM Answer Quality Evaluation`);
  console.log(`Generation model: ${GENERATION_MODEL}`);
  console.log(`Judge model:      ${JUDGE_MODEL}`);
  console.log(`Fixtures:         ${FIXTURES.length}`);
  console.log(`Inter-fixture delay: ${EVAL_DELAY_MS} ms  (set EVAL_DELAY_MS to adjust)\n`);

  const results = [];

  for (const fixture of FIXTURES) {
    const depthTag = fixture.expectedDepth === 'deep' ? '[deep/1024t] ' : '[std/512t]  ';
    process.stdout.write(`  ${fixture.id}  ${depthTag}${fixture.question.slice(0, 48).padEnd(48)} `);

    // 1. Classify and compose prompt.
    // Multi-turn fixtures (priorTurns present) use a unique sessionId and establish
    // prior turns in the production state store before classifying the final question,
    // replicating what orchestrate() does via advanceConversationState + recordAnswerSummary.
    const isMultiTurn = Array.isArray(fixture.priorTurns) && fixture.priorTurns.length > 0;
    const sessionId   = isMultiTurn ? fixture.id : 'eval-session';

    let conversationSummary;
    if (isMultiTurn) {
      const chain = setupPriorTurns(fixture, sessionId);
      conversationSummary = formatChainForEval(chain);
    }

    const d = decide({
      requestId: fixture.id, requestSequence: 1,
      surface: 'manual_chat', modeId: 'technical-interview',
      scope: EVAL_SCOPE,
      sessionId,
      manualQuestion: fixture.question,
    });

    const actualIntent             = d.interviewIntent?.intent ?? '';
    const actualFollowUpLikelihood = d.interviewIntent?.followUpLikelihood ?? '';
    const steps                    = d.answerStrategy?.steps ?? [];

    const evidence = Array.isArray(fixture.mockEvidence) && fixture.mockEvidence.length > 0
      ? buildMockEvidence(fixture.mockEvidence)
      : [];

    const composed = composePrompt({
      decision: d, policy: POLICY, evidence,
      ...(conversationSummary ? { conversationSummary } : {}),
    });

    // Deterministic checks for multi-turn fixtures.
    if (isMultiTurn) {
      if (actualIntent !== fixture.expectedIntent) {
        console.error(`  [${fixture.id}] WARN: expected intent ${fixture.expectedIntent}, got ${actualIntent}`);
      }
      if (!composed.user.includes('Conversation so far')) {
        console.error(`  [${fixture.id}] WARN: conversation section absent from final prompt — context injection may have failed`);
      }
      const priorQ = fixture.priorTurns[0]?.question ?? '';
      if (priorQ && !composed.user.includes(priorQ.slice(0, 40))) {
        console.error(`  [${fixture.id}] WARN: prior question not visible in composed prompt`);
      }
    }

    // 2. Generate answer — deep fixtures get a larger token budget so 8-step
    // strategies (ANALYZE_SCALE, DESIGN_SYSTEM) have room to cover every step.
    const genTokens = fixture.expectedDepth === 'deep' ? 1024 : 512;
    let answer = '';
    try {
      answer = await geminiGenerate(client, GENERATION_MODEL, composed.system, composed.user, genTokens);
      answer = answer.trim();
    } catch (err) {
      console.error(`\n  [${fixture.id}] Generation failed: ${err.message}`);
      results.push({
        caseId: fixture.id, question: fixture.question,
        actualIntent, actualFollowUpLikelihood, answer: '',
        scores: { voice: null, strategy_adherence: null, depth: null, pacing: null, grounding: null },
        flags: [], grade: 'FAIL', gradedBy: 'skipped',
        judgeExplanation: `Generation error: ${err.message}`,
      });
      continue;
    }

    // 2b. Numeric provenance (observability only — does NOT affect grading)
    let numericProvenance = null;
    if (validateNumericProvenance && answer) {
      try {
        numericProvenance = validateNumericProvenance(answer, [fixture.question, composed.user]);
      } catch { /* observability only */ }
    }

    // 3. Deterministic pre-filter
    const preFilterFlags = deterministicPreFilter(answer, fixture, actualIntent);

    // 4. LLM judge (skipped if deterministic failure already found)
    let judgeScores = null;
    let gradedBy = 'deterministic';

    const autoFail =
      preFilterFlags.includes('wrong_voice') ||
      preFilterFlags.includes('template_leak') ||
      preFilterFlags.includes('fabricated_claim');

    if (!autoFail) {
      try {
        judgeScores = await runJudge(client, fixture, actualIntent, steps, answer);
        gradedBy = 'deterministic+llm_judge';
      } catch (err) {
        console.error(`\n  [${fixture.id}] Judge failed: ${err.message}`);
        gradedBy = 'deterministic';
      }
    }

    const finalGrade  = grade(fixture, preFilterFlags, judgeScores);
    const gradeSymbol = finalGrade === 'PASS' ? '✓' : '✗';
    const npSuffix = numericProvenance
      ? (numericProvenance.valid ? '  np=✓' : `  np=✗(${numericProvenance.unsupported.join(',')})`)
      : '';
    process.stdout.write(
      `${gradeSymbol}  ` +
      (judgeScores
        ? `voice=${judgeScores.voice} strat=${judgeScores.strategy_adherence} depth=${judgeScores.depth} pacing=${judgeScores.pacing}`
        : preFilterFlags.length > 0 ? `flags: ${preFilterFlags.join(',')}` : 'deterministic only'
      ) + npSuffix + '\n',
    );

    results.push({
      caseId: fixture.id,
      question: fixture.question,
      actualIntent,
      actualFollowUpLikelihood,
      answer,
      scores: judgeScores
        ? { voice: judgeScores.voice, strategy_adherence: judgeScores.strategy_adherence,
            depth: judgeScores.depth, pacing: judgeScores.pacing, grounding: null }
        : { voice: null, strategy_adherence: null, depth: null, pacing: null, grounding: null },
      flags: preFilterFlags,
      grade: finalGrade,
      gradedBy,
      judgeExplanation: judgeScores?.explanation,
      numericProvenance,
    });

    // Rate-limit pacing: wait between fixtures so we don't exhaust the Gemini
    // free-tier quota (15 req/min). Skipped after the final fixture.
    const isLast = fixture === FIXTURES[FIXTURES.length - 1];
    if (EVAL_DELAY_MS > 0 && !isLast) {
      await new Promise((r) => setTimeout(r, EVAL_DELAY_MS));
    }
  }

  // 5. Aggregate
  const nonSkipped = results.filter((r) => r.gradedBy !== 'skipped');
  const passed     = nonSkipped.filter((r) => r.grade === 'PASS').length;
  const failed     = nonSkipped.filter((r) => r.grade === 'FAIL').length;
  const skipped    = results.filter((r)  => r.gradedBy === 'skipped').length;
  const passRate   = nonSkipped.length > 0 ? passed / nonSkipped.length : 0;

  const report = {
    date:       new Date().toISOString(),
    model:      GENERATION_MODEL,
    judgeModel: JUDGE_MODEL,
    totalCases: results.length,
    passed,
    failed,
    skipped,
    passRate,
    results,
  };

  console.log('\n── Results ─────────────────────────────────────────────────');
  console.log(`Total:   ${results.length}`);
  console.log(`Passed:  ${passed}`);
  console.log(`Failed:  ${failed}`);
  console.log(`Skipped: ${skipped}`);
  console.log(`Pass rate (non-skipped): ${(passRate * 100).toFixed(1)}%`);
  console.log(passRate >= 0.8 ? '\n✓ PASS — ≥ 80% threshold met' : '\n✗ FAIL — below 80% threshold');

  // 5b. Numeric provenance summary table (NP fixtures only)
  const npResults = results.filter((r) => r.caseId.startsWith('np_'));
  if (npResults.length > 0) {
    console.log('\n── Numeric Provenance Results ──────────────────────────────');
    console.log('Fixture  NP valid   Unsupported claims         Answer-quality');
    console.log('───────  ─────────  ─────────────────────────  ──────────────');
    for (const r of npResults) {
      const np = r.numericProvenance;
      const npValid   = np ? (np.valid ? 'true ' : 'false') : 'n/a  ';
      const npUnsup   = np && !np.valid ? np.unsupported.join(', ') : '—';
      const npUnsupPad = npUnsup.slice(0, 25).padEnd(25);
      console.log(`${r.caseId.padEnd(8)} ${npValid}      ${npUnsupPad}  ${r.grade}`);
    }
  }

  // 6. Write JSON report
  const resultsDir = path.join(__dirname, 'results');
  fs.mkdirSync(resultsDir, { recursive: true });
  const datePart   = new Date().toISOString().slice(0, 10);
  const reportPath = path.join(resultsDir, `${datePart}.json`);
  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2), 'utf8');
  console.log(`\nReport written to: ${reportPath}\n`);
}

run().catch((err) => {
  console.error('\n[eval:interview] Unexpected error:', err);
  process.exit(1);
});
