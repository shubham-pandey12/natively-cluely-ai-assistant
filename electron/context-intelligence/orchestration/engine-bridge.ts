// electron/context-intelligence/orchestration/engine-bridge.ts
//
// One adoption point for the IntelligenceEngine surfaces.
//
// WHY A SHARED BRIDGE
// Five engine surfaces — assist, clarify, brainstorm, code-hint and manual
// answer — construct NO source authority whatsoever today. Each passes a raw
// `session.getFormattedContext(N)` blob straight to the model. Wiring them one
// at a time would recreate the very thing F2 describes: several near-identical
// decision sites drifting apart. There is exactly one here.
//
// Returns null whenever the flag is off or anything goes wrong, so a caller's
// integration is a two-line change and the failure mode is "legacy behaviour",
// never "no answer".

import { isContextIntelligenceV3Enabled } from '../contracts/flag';
import { orchestrate, type AnswerRequest, type RetrievalPort, type OrchestratorResult } from './orchestrator';
import { composePrompt } from '../generation/prompt-composer';
import { resolveModePolicy, isModeId, type ModeId } from '../policies/mode-policy-registry';
import { recordLegacyTurn } from '../observability/legacy-trace';
import {
  readProviderScopePolicy,
  filterEvidenceByProviderScopes,
  dataScopesForEvidence,
  isScopeDenied,
} from '../policies/provider-scope-policy';
import { DESIGN_INTENTS } from '../contracts/types';
import type { AnswerSurface, EvidenceScope, EvidenceItem, InterviewIntent } from '../contracts/types';
import type { ProviderDataScope } from '../../llm/ProviderRouter';

export interface BridgeInput {
  surface: AnswerSurface;
  question: string;
  /** Raw templateType from ModesManager; unknown ids fall back rather than throw. */
  modeTemplateType?: string | null;
  /** The mode's UNIQUE id (mode_<uuid>) when one exists. Keys the per-mode
   *  Answer policy choice: two custom modes share a templateType, never an id. */
  modeUniqueId?: string | null;
  /** How many reference files the active mode has. Lets the composer say "no
   *  document is attached" instead of "the document does not mention it". */
  attachedSourceCount?: number;
  /** Attached file NAMES — deterministic filename-role routing (glossary /
   *  formula sheet, deep-run 2 issue 9). Always populated by call sites;
   *  never gated on debug level (routing must not depend on logging). */
  attachedFileNames?: readonly string[];
  /** How many Profile Intelligence sources hydrated this turn's retrieval
   *  (active résumé / target JD). Composer wording + telemetry — a zero-
   *  attachment turn with a live profile must NOT claim nothing was searched. */
  profileSourceCount?: number;
  /** Identity of the profile sources resolved into the turn ({role, id} only,
   *  never content) — the [V3] line's answer to "which source pools existed",
   *  as distinct from `sources` = what retrieval actually accepted. */
  resolvedProfileSources?: Array<{ role: string; id: string }>;
  /** Human-readable mode name for the [V3] observability line. */
  modeName?: string | null;
  /** Distinguishes call sites that share an AnswerSurface (AnswerSurface has
   *  no 'clarify'/'brainstorm' members, and the engine's manual-answer path
   *  shares 'manual-chat' with the IPC surface). Appended to legacyPath and
   *  the [V3] line so traces from different call sites stay separable. */
  pathTag?: string;
  scope?: Partial<EvidenceScope>;
  requestId?: string;
  requestSequence?: number;
  isFollowUp?: boolean;
  hasScreenContext?: boolean;
  /**
   * Where `question` came from. Defaults to 'manual' — correct for the manual
   * chat and typed-question call sites, which is what every caller was before
   * this field existed.
   *
   * 'transcript' means the string was chosen from live speech by
   * extractLatestQuestion, NOT typed by the user. That matters downstream:
   * resolveQuestion (orchestrator.ts) stamps manual input `confidence: 1`, so
   * a low-confidence extraction — a fragment the extractor itself scored 0.3
   * or 0.4 — was arriving indistinguishable from a deliberate typed question,
   * with no answerability signal for the decision layer to defend against.
   */
  questionSource?: 'manual' | 'transcript';
  /** Extractor confidence (0..1) when questionSource is 'transcript'. */
  questionConfidence?: number;
  /**
   * Attachment-derived source types for custom/general modes (deep-test D10) —
   * computed at the call site that holds the files, via
   * attachmentSourceTypeExtensions. Additive only.
   */
  extraAllowedSourceTypes?: import('../contracts/types').SourceType[];
  /**
   * Context-debug (2026-08-01): identity of the sources this turn COULD read
   * — id/role/name/status only, built at the call site that holds the files.
   * Never content.
   */
  debugSources?: import('../debug/debug-types').ContextDebugSource[];
  /**
   * TRUE when the calling transport will record generation timing + the final
   * answer and call the collector's complete() itself (manual-chat). Surfaces
   * that cannot correlate the final answer leave this unset and the bridge
   * finalizes a decision-only record immediately.
   */
  deferDebugCompletion?: boolean;
  /** Tone/length only — cannot widen authorization (§19.2). */
  realtimeInstruction?: string;
  conversationSummary?: string;
  retrieval?: RetrievalPort;
  /**
   * Surface persona/voice contract factory (2026-08-02). Called AFTER the
   * decision exists so the caller can compose against what the turn actually
   * is (`codingTask` = the router classified it a coding problem — the same
   * semantic-activation contract Prompt System v2 uses everywhere else).
   * A callback rather than a string because the caller cannot know the
   * decision, and the bridge must not import the caller's prompt system —
   * the caller (which already holds both worlds) supplies the bridge a
   * closure instead. Returning null/throwing ⇒ no persona, composition
   * byte-identical to before this field existed.
   */
  personaBase?: (ctx: { codingTask: boolean }) => string | null;
}

export interface BridgeResult {
  system: string;
  user: string;
  answerability: string;
  fallbackUsed: string;
  evidenceCount: number;
  /** True when the mode authorizes no source for this question — the caller
   *  should NOT quietly answer from model knowledge. */
  unsupportedInMode: boolean;
  modeId: ModeId;
  /**
   * The provider data scopes this prompt ACTUALLY carries, derived from the
   * source types of the evidence that survived filtering AND packing.
   *
   * The transport must be TOLD what it is sending. Before this existed, the V3
   * call site passed `[]` and LLMHelper regex-sniffed the payload for legacy
   * tag names that a V3 prompt never contains — so it inferred nothing and
   * enforced nothing. Pass this straight into streamChat's `extraDataScopes`.
   */
  packedDataScopes: ProviderDataScope[];
  /** Scopes whose evidence a privacy setting withheld from this turn. Empty on
   *  a normal turn. Present so callers can log/surface the withholding. */
  withheldDataScopes: ProviderDataScope[];
  /** Set when a context-debug collector is open for this turn (deferred
   *  completion) — the transport looks it up by this id to record the final
   *  answer and timings. */
  debugRequestId?: string;
}

/**
 * Format a topic chain for prompt injection (Phase 4).
 * Chain depth 1 matches the existing flat format for backward compatibility.
 * Chain depth 2+ summarises all turns with numbered lines.
 */
function formatTopicChain(chain: readonly import('../contracts/types').ChainTurn[]): string {
  if (chain.length === 0) return '';
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

/**
 * Apply the outbound provider-data-scope filter over evidence and conversation
 * summary in one consistent pass. Called after retrieval, before prompt packing.
 *
 * Why here — not inside the composer or after packing: the composer writes its
 * grounding contracts (checked-absence, no-evidence notice) against the evidence
 * it receives. Filtering downstream would leave a prompt whose contracts describe
 * material the model can no longer see, which is a fabrication path.
 *
 * Policy is read live on every call — never cached — because esbuild inlines this
 * module into every entry bundle and a cached copy would go stale across bundles.
 *
 * convoSummary maps to the transcript scope: it reaches the prompt as prose and is
 * invisible to the evidence filter, so it is suppressed here when that scope is
 * denied.
 */
function applyOutboundScopeFilter(
  evidence: readonly EvidenceItem[],
  convoSummary: string | undefined,
) {
  const scopePolicy    = readProviderScopePolicy();
  const scopeFilter    = filterEvidenceByProviderScopes(evidence, scopePolicy);
  const withheldScopes = new Set<ProviderDataScope>(scopeFilter.withheldScopes);

  let filteredSummary = convoSummary;
  if (filteredSummary && isScopeDenied('transcript', scopePolicy)) {
    filteredSummary = undefined;
    withheldScopes.add('transcript');
  }

  return { scopeFilter, withheldScopes, convoSummary: filteredSummary };
}

/**
 * Safely invoke the caller-supplied persona factory.
 * Returning null or throwing is treated as "no persona" — composition is
 * byte-identical to a missing factory.
 */
function resolvePersona(
  factory: BridgeInput['personaBase'],
  isCodingTask: boolean,
): string | undefined {
  try {
    return factory?.({ codingTask: isCodingTask }) ?? undefined;
  } catch {
    return undefined;
  }
}

/**
 * Evaluate the conversation-chain gate for the current turn and return the
 * formatted topic-chain string when the gate is open and the chain is non-empty.
 *
 * Gate opens when contextRequirements.conversation is true, OR when the intent is
 * one of the long-form design intents (DESIGN_INTENTS) and the behavior is not
 * HINT. Returns undefined when the gate is closed or the chain is empty — callers
 * treat undefined as complete exclusion from the prompt.
 *
 * Phase 4/5: design-oriented QUESTION turns whose interviewerBehavior is QUESTION
 * (so contextRequirements.conversation=false) but whose intent is one of the
 * long-form design intents open the gate so established interview context (e.g.
 * "we have 100M users") is visible to subsequent design sub-questions.
 * Phase 5 GAP-1: extended from 3 to 6 intents — optimization, tradeoff, and
 * comparison commonly appear mid-design-thread and need the same chain access.
 *
 * Does not mutate preOrchState.
 */
function evaluateConversationGate(
  interviewIntent: InterviewIntent | undefined,
  preOrchState: import('../question/conversation-state').ConversationState | null,
): string | undefined {
  const intentIsDesign = !!interviewIntent && DESIGN_INTENTS.has(interviewIntent.intent);
  const behaviorSuppresses = interviewIntent?.interviewerBehavior === 'HINT';
  const conversationGateOpen =
    (interviewIntent?.contextRequirements.conversation ?? false) ||
    (intentIsDesign && !behaviorSuppresses);
  if (conversationGateOpen && preOrchState) {
    const chain = preOrchState.topicChain ?? [];
    if (chain.length > 0) {
      return formatTopicChain(chain);
    }
  }
  return undefined;
}

/**
 * Fire the Context Intelligence debug collector for the current turn.
 *
 * Observes the SAME objects the [V3] line reads — the production trace and
 * frozen decision — never recomputing any of it. Level 'off' costs one
 * function call.
 *
 * Returns the requestId when deferDebugCompletion=true so the transport can
 * complete the record after generation; returns undefined otherwise.
 * Any failure degrades to undefined — debug logging must never break an answer.
 */
function recordDebugTurn(
  req: AnswerRequest,
  input: BridgeInput,
  result: OrchestratorResult,
  modeId: string,
  raw: string,
  policyVersion: string,
  scopeFilterEvidence: readonly EvidenceItem[],
): string | undefined {
  try {
    const { getContextDebugLevel, getContentInclusionEnabled } = require('../debug/debug-config');
    const level = getContextDebugLevel();
    if (level === 'off') return undefined;

    const { beginTurnCollector } = require('../debug/turn-collector');
    const collector = beginTurnCollector({
      sessionId: req.sessionId,
      ...(req.scope.meetingId ? { meetingId: req.scope.meetingId } : {}),
      turnId: `turn_${req.requestId}`,
      requestId: req.requestId,
      conversationGeneration: req.requestSequence,
      modeId,
      ...(input.modeUniqueId ? { modeUniqueId: input.modeUniqueId } : {}),
      surface: `${input.surface}${input.pathTag ? `:${input.pathTag}` : ''}`,
    }, { level, includeContent: getContentInclusionEnabled(level) });

    collector.recordDecisionTrace({
      trace: result.trace,
      decision: result.decision,
      modeName: input.modeName,
      modeType: raw === 'general' && (input.modeName ?? 'General') !== 'General' ? 'custom' : 'default',
      policyVersion,
      extraAllowedSourceTypes: input.extraAllowedSourceTypes as readonly string[] | undefined,
      documentSpecific: result.decision.claimRequirements
        .some((c) => c.authority === 'PRIVATE_SOURCE_REQUIRED'),
      propertyMatched: (result.trace.claimPlan ?? [])
        .some((c) => c.support === 'DIRECT_EVIDENCE'),
    });
    const resolvedProfiles = input.resolvedProfileSources ?? [];
    collector.recordAvailableSources({
      modeAttachmentCount: input.attachedSourceCount ?? 0,
      profileResumeCount: resolvedProfiles.filter((s) => s.role === 'profile_resume').length,
      profileJobDescriptionCount: resolvedProfiles.filter((s) => s.role === 'profile_job_description').length,
      profileFactCount: resolvedProfiles.filter((s) => s.role === 'profile_fact').length,
    });
    // Authorized sources = mode attachments PLUS resolved Profile
    // Intelligence pools (deep-run 2, issue 14: profile turns logged
    // authorizedSources: [] while the résumé/JD pools answered the turn).
    const authorized = [
      ...(input.debugSources ?? []),
      ...resolvedProfiles.map((s) => ({ id: s.id, role: s.role })),
    ];
    if (authorized.length) collector.recordAuthorizedSources(authorized);
    const rr = result.trace.referentResolution;
    if (rr) {
      collector.recordConversationState({
        activePerson: rr.activePerson ?? null,
        activeTopic: rr.activeTopic ?? null,
        previousQuestion: rr.previousQuestion ?? null,
        referentApplied: rr.applied,
        referentReason: rr.reason ?? null,
        referent: rr.referent ?? null,
      });
    }
    // The FILTERED set: the debug record must show what the model was
    // actually sent, and must not persist content the user's privacy
    // setting withheld from a cloud provider.
    collector.recordEvidenceItems(scopeFilterEvidence);

    if (input.deferDebugCompletion) {
      return req.requestId;   // the transport completes it
    }
    collector.recordAnswer('', false, 'answer_not_correlated_on_this_surface');
    collector.complete();
    return undefined;
  } catch {
    return undefined;
  }
}

/**
 * Build a V3 prompt for an engine surface, or null to keep legacy behaviour.
 *
 * Never throws. A defect in the new path must degrade to legacy, never break a
 * live answer — the same contract the wired manual-chat surface uses.
 */
export async function buildV3Prompt(input: BridgeInput): Promise<BridgeResult | null> {
  try {
    if (!isContextIntelligenceV3Enabled()) return null;
    const question = String(input.question || '').trim();
    if (!question) return null;

    const raw = input.modeTemplateType ?? 'technical-interview';
    const modeId: ModeId = isModeId(raw) ? raw : 'technical-interview';

    // The user's per-mode grounding choice (§6). Read per turn — not cached on
    // the bridge — so a change in Settings applies to the very next answer.
    let userAnswerPolicy: import('../policies/answer-policy').AnswerPolicy | null = null;
    try {
      const { getStoredAnswerPolicy } = require('../policies/answer-policy-store');
      userAnswerPolicy = getStoredAnswerPolicy(input.modeUniqueId ?? modeId);
    } catch { /* store unavailable — mode default applies */ }
    const policy = resolveModePolicy(modeId);

    const req: AnswerRequest = {
      requestId: input.requestId ?? `v3-${input.surface}-${Date.now()}`,
      requestSequence: input.requestSequence ?? 0,
      surface: input.surface,
      modeId,
      scope: { userId: 'local', ...input.scope },
      sessionId: input.scope?.sessionId ?? 'engine',
      // Route the question to the field that matches its actual provenance so
      // resolveQuestion assigns real confidence (manual → 1, transcript → 0.7)
      // instead of stamping everything `manual/1.0`.
      ...(input.questionSource === 'transcript'
        ? { transcriptQuestion: question }
        : { manualQuestion: question }),
      questionConfidence: input.questionConfidence,
      userAnswerPolicy,
      isFollowUp: input.isFollowUp,
      hasScreenContext: input.hasScreenContext,
      // Definite value lookups ground only where documents exist (deep-test D2).
      hasAttachedDocuments: (input.attachedSourceCount ?? 0) > 0
        || (input.profileSourceCount ?? 0) > 0,
      attachedFileNames: input.attachedFileNames,
      extraAllowedSourceTypes: input.extraAllowedSourceTypes,
    };

    // Phase 4: capture PRE-ORCHESTRATION state before orchestrate() advances it.
    // The conversation chain for THIS turn's prompt is built from what existed
    // BEFORE the current question was processed — the post-advance state has the
    // current turn already appended and must never be used as prior context.
    let preOrchState: import('../question/conversation-state').ConversationState | null = null;
    try {
      const { getConversationState } = require('../question/conversation-state-store');
      preOrchState = getConversationState(req.sessionId);
    } catch { /* continuity must never break a turn */ }

    const result = await orchestrate(req, input.retrieval, preOrchState);

    // CLARIFICATION short-circuit (Phase 8 Step 17, DoD).
    // When a bare follow-up has an unresolvable referent the orchestrator sets
    // trace.fallbackUsed = 'CLARIFICATION'. Calling composePrompt() after that
    // produces a system prompt that contains both the clarifying-question
    // directive AND the full answer-generation framework (strategy, depth,
    // capabilities, grounding), making "clarifying question + answer in same
    // turn" structurally possible. Returning here before composePrompt()
    // removes that possibility by construction.
    if (result.trace.fallbackUsed === 'CLARIFICATION') {
      return {
        system:
          '# Task\n'
          + 'The follow-up question could not be resolved — you do not know its subject.\n'
          + 'Ask exactly ONE brief clarifying question. Name your best guess at the subject. '
          + 'Do NOT answer as though the subject were known.',
        user: `# Question\n${result.decision.resolvedQuestion}`,
        answerability: result.answerability,
        fallbackUsed: 'CLARIFICATION',
        evidenceCount: 0,
        packedDataScopes: [],
        withheldDataScopes: [],
        unsupportedInMode: false,
        modeId,
      };
    }

    // Build conversation summary AFTER orchestrate() so we have the current turn's
    // interviewIntent to consult the gate — but USING the pre-orchestration chain.
    // Caller-supplied conversationSummary bypasses the gate (backward compat).
    let convoSummary: string | undefined = input.conversationSummary;
    if (!convoSummary) {
      // conversation=false and non-design intent → evaluateConversationGate returns
      // undefined → convoSummary stays undefined → complete exclusion from prompt.
      convoSummary = evaluateConversationGate(result.decision.interviewIntent, preOrchState);
    }

    // ── Outbound provider-data-scope filter ─────────────────────────────────
    const { scopeFilter, withheldScopes, convoSummary: filteredConvoSummary } =
      applyOutboundScopeFilter(result.evidence, convoSummary);
    convoSummary = filteredConvoSummary;

    // Phase 15 (D-03): derive codingTask from the authoritative semantic intent,
    // not from raw questionTypes — questionTypes still carries CODING_TASK for
    // DSA concept questions because CODING_TASK_RE matches DSA nouns, but Phase 13
    // resolved those to concept_explanation in interviewIntent.intent.
    const personaBase = resolvePersona(
      input.personaBase,
      result.decision.interviewIntent?.intent === 'coding_task',
    );

    const composed = composePrompt({
      decision: result.decision,
      policy,
      personaBase,
      evidence: scopeFilter.evidence,
      withheldScopes: [...withheldScopes],
      realtimeInstruction: input.realtimeInstruction,
      conversationSummary: convoSummary,
      attachedSourceCount: input.attachedSourceCount,
      profileSourceCount: input.profileSourceCount,
      // Defect D (2026-08-01): the CLARIFICATION verdict was computed and then
      // dropped here — the composer is the only place it can act.
      fallbackUsed: result.trace.fallbackUsed,
    });

    // Phase 11: record prompt section names for observability (names only, never content).
    result.trace.promptSectionsRendered = composed.sections;

    // What this prompt actually carries: the scopes of the evidence that
    // survived BOTH the privacy filter and the packing budget, plus the
    // transcript scope when a conversation summary rides along. This is the
    // truth handed to the transport instead of it guessing from the bytes.
    const includedIds = new Set(composed.packed.includedEvidenceIds);
    const packedDataScopes = new Set<ProviderDataScope>(
      dataScopesForEvidence(scopeFilter.evidence.filter((e) => includedIds.has(e.evidenceId))),
    );
    if (convoSummary) packedDataScopes.add('transcript');

    // ── Per-turn source line ────────────────────────────────────────────────
    // The one thing production could not answer about itself. A cross-mode
    // contamination report arrived with a full terminal log that contained no
    // record of which mode, which files, or which evidence any turn used, so
    // the defect had to be reconstructed by comparing answer prose against the
    // reference files in SQLite. Identity only — never content (12 §4).
    try {
      const acc = result.trace.acceptedEvidence ?? [];
      const srcIds = [...new Set(acc.map((e) => e.sourceId))];
      // {role, id} pairs of what retrieval actually ACCEPTED — the counterpart
      // of resolvedSources (what pools existed). Identity only, never content.
      const retrievedSources = [...new Map(
        acc.map((e) => [`${e.sourceType}:${e.sourceId}`, { role: e.sourceType, id: e.sourceId }]),
      ).values()];
      console.log('[V3]', JSON.stringify({
        surface: input.surface,
        ...(input.pathTag ? { tag: input.pathTag } : {}),
        mode: modeId,
        modeUniqueId: input.modeUniqueId ?? null,
        modeName: input.modeName ?? null,
        // `attachedFiles` kept for rerun-protocol parsers; the split fields are
        // the honest representation — attachedFiles alone reported the whole
        // source state as 0 while a profile résumé/JD pool existed.
        attachedFiles: input.attachedSourceCount ?? null,
        modeAttachedFiles: input.attachedSourceCount ?? null,
        profileSources: input.profileSourceCount ?? 0,
        profileResumeSources: (input.resolvedProfileSources ?? []).filter((s) => s.role === 'profile_resume').length,
        profileJobDescriptionSources: (input.resolvedProfileSources ?? []).filter((s) => s.role === 'profile_job_description').length,
        profileFactSources: (input.resolvedProfileSources ?? []).filter((s) => s.role === 'profile_fact').length,
        resolvedSources: input.resolvedProfileSources ?? [],
        // Defect D/F telemetry (2026-08-01): the resolver's decision and the
        // retrieval funnel were both invisible — a stale referent rewrite and a
        // "0 raw candidates" vs "all candidates filtered" distinction each had
        // to be reconstructed from answer prose.
        originalQuestion: result.trace.originalQuestion,
        resolvedQuestion: result.trace.resolvedQuestion,
        intent: result.trace.questionTypes,
        knowledgePolicy: policy.groundingPolicy,
        path: result.decision.retrievalPlan.path,
        planned: result.decision.retrievalPlan.sourceTypes,
        evidence: acc.length,
        sources: srcIds,
        retrievedSources,
        retrieval: (result.trace.retrievalAttempts ?? []).map((a) => ({
          candidates: a.candidateCount,
          admitted: a.admittedAfterScopeFilter,
          rejected: a.rejectedByScopeFilter,
          ...(a.failed ? { failed: true } : {}),
        })),
        answerability: result.trace.answerability,
        fallback: result.trace.fallbackUsed,
        // Privacy withholding (2026-08-01). Identity/counts only. A turn that
        // answered thinly because the user switched a data scope off was
        // previously indistinguishable in the logs from a retrieval miss.
        privacyWithheldCount: scopeFilter.withheldCount,
        privacyWithheldScopes: [...withheldScopes],
        outboundScopes: [...packedDataScopes],
        // Deep-test D5/D6 (2026-08-01): the two signals that turn a masked
        // failure into a diagnosable one — was this a document-specific
        // question, and did any claim get property-level (not merely topical)
        // support.
        documentSpecific: result.decision.claimRequirements
          .some((c) => c.authority === 'PRIVATE_SOURCE_REQUIRED'),
        propertyMatched: (result.trace.claimPlan ?? [])
          .some((c) => c.support === 'DIRECT_EVIDENCE'),
      }));
    } catch { /* observability must never break an answer */ }

    // ── Context Intelligence debug collector (2026-08-01) ───────────────────
    const debugRequestId = recordDebugTurn(req, input, result, modeId, raw, policy.version, scopeFilter.evidence);

    try {
      recordLegacyTurn({
        ...(result.trace as unknown as Record<string, unknown>),
        legacyPath: `v3-${input.surface}${input.pathTag ? `-${input.pathTag}` : ''}`,
      } as never);
    } catch { /* observability must never break an answer */ }

    return {
      system: composed.system,
      user: composed.user,
      answerability: result.answerability,
      fallbackUsed: result.trace.fallbackUsed,
      // Post-filter: the count callers branch on must describe the evidence the
      // model was actually given, not the evidence retrieval found.
      evidenceCount: scopeFilter.evidence.length,
      packedDataScopes: [...packedDataScopes],
      withheldDataScopes: [...withheldScopes],
      // GROUNDED with nothing to retrieve means the mode authorizes no source
      // for this question. Distinct from FAST, where none was needed.
      unsupportedInMode: result.decision.retrievalPlan.path !== 'FAST'
        && result.decision.retrievalPlan.shouldRetrieve === false,
      modeId,
      ...(debugRequestId ? { debugRequestId } : {}),
    };
  } catch (e) {
    // Flag-off returns null EARLY above; reaching here means the V3 path
    // FAILED and the caller will silently revert to legacy. That reversion
    // must be observable (§22.1) — count it and log one structured line.
    try {
      require('../observability/rollout-metrics').recordV3Fallback(
        `${input.surface}${input.pathTag ? `-${input.pathTag}` : ''}`, e,
      );
    } catch { /* observability only */ }
    return null;
  }
}
