// electron/context-intelligence/validation/numeric-enforcement.ts
//
// Phase 8 Step 23 — pure numeric provenance enforcement logic.
// No external imports. No IO. No Gemini calls. No streaming.
//
// This module encapsulates the enforcement decision only. Answer delivery
// and retry generation are provided by the caller as an optional callback.
//
// Production streaming integration note
// ──────────────────────────────────────
// The V3 answer path in ipcHandlers.ts sends each token to the renderer
// in real time (`gemini-stream-token` inside the streaming loop, line 1294).
// By the time `finalText` is assembled the user has already seen the full
// answer. Enforcement with stream replacement therefore requires either:
//
//   A. Buffered delivery — suppress `gemini-stream-token` events until
//      validation passes; send the complete validated answer as a single
//      synthetic token + `gemini-stream-done`. Loses the streaming typing UX.
//
//   B. Renderer replace protocol — add `gemini-stream-replace` IPC event
//      that the renderer handles by clearing and re-rendering content.
//      Requires renderer changes not in scope for this step.
//
// Until one of those is implemented, the production call passes
// `regenerate = undefined` and enforcement is observability-only.
// See Step 23 report for the minimal next step.

import { validateNumericProvenance } from './numeric-provenance';

/** Hard limit on retry attempts. Never exceeded by `applyNumericEnforcement`. */
export const MAX_NUMERIC_RETRIES = 1;

/**
 * The correction hint injected into the regeneration prompt when the first
 * answer contains unsupported numeric claims.
 */
export const CORRECTION_HINT =
  'The previous answer contained numeric claims that were not supported by the ' +
  'supplied sources. Rewrite the answer without introducing unsupported factual ' +
  'numbers. Use only numeric values explicitly supported by the interviewer ' +
  'question or supplied evidence. Do not invent, estimate, calculate, or derive ' +
  'new factual metrics.';

/** Result of an enforcement evaluation. */
export interface NumericEnforcementResult {
  /** Whether the first generated answer passed numeric provenance. */
  initialValid: boolean;
  /** Whether a retry was attempted (requires `regenerate` callback). */
  enforced: boolean;
  /** Number of regeneration attempts made. Always 0 or 1. */
  retryCount: number;
  /** Whether the final answer (post-retry if enforced) passed provenance. */
  finalValid: boolean;
  /** Unsupported numbers found in the first answer. */
  initialUnsupported: string[];
  /** Unsupported numbers found in the final answer ([] when valid). */
  finalUnsupported: string[];
  /** The answer text to deliver (original if valid or no-retry; retry if enforced+valid; original if fallback). */
  finalAnswer: string;
}

/**
 * Apply numeric provenance enforcement to a generated answer.
 *
 * Flow:
 *   1. Validate `answer` against `sources`.
 *   2. If valid → return immediately, `enforced=false`.
 *   3. If invalid and `regenerate` is present → call it exactly ONCE with
 *      `CORRECTION_HINT`; validate the retry answer; return result.
 *   4. If invalid and no `regenerate` → return `enforced=false, finalValid=false`
 *      (caller must handle safe fallback; answer is unchanged).
 *
 * The `regenerate` callback receives the correction hint and must return
 * the full text of the regenerated answer. It is called at most once.
 *
 * @param answer   The generated answer to validate.
 * @param sources  Authoritative sources (question + evidence text).
 * @param regenerate  Optional async callback for a single controlled retry.
 */
export async function applyNumericEnforcement(
  answer: string,
  sources: string[],
  regenerate?: (correctionHint: string) => Promise<string>,
): Promise<NumericEnforcementResult> {
  const initial = validateNumericProvenance(answer, sources);

  if (initial.valid) {
    return {
      initialValid:      true,
      enforced:          false,
      retryCount:        0,
      finalValid:        true,
      initialUnsupported: [],
      finalUnsupported:  [],
      finalAnswer:       answer,
    };
  }

  // Invalid — check whether regeneration is available.
  if (!regenerate) {
    // No regeneration available. Return the original answer unchanged.
    // The caller is responsible for logging and observability.
    return {
      initialValid:      false,
      enforced:          false,
      retryCount:        0,
      finalValid:        false,
      initialUnsupported: initial.unsupported,
      finalUnsupported:  initial.unsupported,
      finalAnswer:       answer,
    };
  }

  // Perform exactly ONE regeneration (MAX_NUMERIC_RETRIES = 1; loop is structurally impossible).
  const retryAnswer = await regenerate(CORRECTION_HINT);
  const retry = validateNumericProvenance(retryAnswer, sources);

  return {
    initialValid:      false,
    enforced:          true,
    retryCount:        1,
    finalValid:        retry.valid,
    initialUnsupported: initial.unsupported,
    finalUnsupported:  retry.unsupported,
    finalAnswer:       retryAnswer,
  };
}
