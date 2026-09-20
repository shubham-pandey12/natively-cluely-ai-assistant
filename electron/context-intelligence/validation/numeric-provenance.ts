// electron/context-intelligence/validation/numeric-provenance.ts
//
// Phase 8 Step 19 — pure deterministic numeric provenance validator.
// No external imports. No IO. No LLM calls. No network access.
//
// Two exports:
//   extractFactualNumbers(text)          → string[]
//   validateNumericProvenance(ans, srcs) → { valid, unsupported }
//
// Intentional limitation: unit conversion is NOT supported.
// "1.2 seconds" and "1200ms" normalize to "1.2" and "1200" respectively
// and will NOT match each other. Unit-aware normalization is deferred.

/**
 * Patterns whose matched spans are replaced with equal-length whitespace
 * before factual-number extraction. Applied in order so later patterns see
 * already-blanked input.
 *
 * Numbers excluded here are NOT factual claims — they are structural labels,
 * technical identifiers, or version/status codes.
 */
const EXCLUSION_PATTERNS: readonly RegExp[] = [
  // 1. Asymptotic / complexity notation: O(n²), Θ(n log n), Ω(1)
  /\b[OΘΩ]\s*\([^)]*\)/g,

  // 2. HTTP status codes and protocol versions: HTTP 404, HTTP/2, HTTPS/1.1
  //    One pattern covers both — HTTP + separator (space or slash) + digits.
  /\bHTTPS?[\s/]\d[\d.]*/gi,

  // 3. Port and URL numbers: port 8080, localhost:3000
  /\bport\s+\d+/gi,
  /\blocalhost:\d+/gi,

  // 4. Software version contexts: Node.js 20, React 19, Python 3.11
  //    The \s*\.?\s* between name and version handles "Node.js 20" and "ES 2026".
  /\b(?:node(?:\.js)?|react|angular|vue|python|ruby|java|swift|kotlin|php|rust|go|typescript|javascript|css|html|ecmascript)\s*\.?\s*\d[\d.]*/gi,

  // 5. Generic version prefix: v3.1, v20, v1.2.3
  /\bv\d[\d.]*/g,

  // 6. Structural number references: Step 1, item 2, Question 3, Chapter 4
  /\b(?:step|item|question|chapter|part|section|phase|rule|example|figure|table|appendix|lesson)\s+\d+/gi,

  // 7. Ordered list markers at the start of a line: "1. " or "1) "
  /^\s*\d+[.)]\s/gm,

  // 8. Bit-width descriptors: 32-bit, 64-bit, 128-bit
  /\b\d+-bit\b/gi,

  // 9. Letter-before-digit technical identifiers: p99, L4, R7, IPv4.
  //    The leading \b ensures this does NOT fire on unit suffixes that
  //    directly follow digits — e.g. in "340ms" the 'm' is NOT preceded
  //    by a word boundary (digit '0' → letter 'm' carries no \b), so "ms"
  //    is not matched and the factual "340" is preserved.
  /\b[a-zA-Z]+\d+[a-zA-Z]*\b/g,

  // 10. Bare HTTP status codes preceded by common HTTP-action verbs:
  //     "returned 200", "status 503", "got 404", "received 201".
  //     Matches 3-digit numbers in the HTTP range (100-599) immediately
  //     following a verb. Does NOT match 4-digit performance metrics (5000)
  //     because [12345]\d{2} is exactly 3 digits with a trailing word boundary.
  /\b(?:return(?:ed|s)?|status|received?|got)\s+(?:(?:HTTP|HTTPS)\s+)?[12345]\d{2}\b/gi,

  // 11. Bare HTTP status codes followed by common HTTP-descriptor nouns:
  //     "301 redirect", "404 response", "500 error".
  //     Same 3-digit HTTP range constraint as pattern 10.
  /\b[12345]\d{2}\s+(?:redirect|response|error)\b/gi,
];

/**
 * Replace fenced code blocks (```...```) and inline code (`...`) with
 * equal-length whitespace so numeric literals inside code examples are not
 * treated as factual claims.
 *
 * Applied before applyExclusions() and NUMERIC_RE scanning.
 * Does not attempt to parse the code; this is purely text-level masking.
 *
 * Known limitation: code blocks with unbalanced backticks or non-standard
 * fencing (~~~~) are not masked. These are uncommon in LLM answers.
 */
function maskCodeRegions(text: string): string {
  // Fenced blocks: ```...``` with any optional language hint on the opening line.
  // Non-greedy [\s\S]*? handles multi-line blocks and stops at the first ```.
  let t = text.replace(/```[\s\S]*?```/g, (m) => ' '.repeat(m.length));
  // Inline code: `...` — single backtick pair, no newlines inside.
  t = t.replace(/`[^`\n]+`/g, (m) => ' '.repeat(m.length));
  return t;
}

/** Replace all spans matched by re with spaces of the same length. */
function applyExclusions(text: string): string {
  let t = text;
  for (const re of EXCLUSION_PATTERNS) {
    t = t.replace(re, (m) => ' '.repeat(m.length));
  }
  return t;
}

/**
 * Normalize a raw numeric match to a canonical string for comparison:
 *   - strip leading currency symbol ($, ₹, £, €)
 *   - remove comma separators (5,000 → 5000)
 *
 * Percent signs and unit suffixes (ms, s, x, K …) are NOT part of the
 * NUMERIC_RE match and need no stripping here.
 */
function normalize(raw: string): string {
  return raw
    .replace(/^[$₹£€]\s*/, '') // strip leading currency
    .replace(/,/g, '');                         // remove comma separators
}

/**
 * Extract numeric strings representing factual claims from `text`.
 *
 * The following are excluded (not treated as factual):
 *   - Code regions:         fenced blocks ```...``` and inline `...`
 *   - Structural labels:    Step 1, item 2, Question 3, Chapter 4
 *   - Software versions:    Node.js 20, React 19, Python 3.11, v3.1
 *   - HTTP codes/versions:  HTTP 404, HTTP/2; bare status codes in context
 *                           ("returned 200", "404 response", "status 503")
 *   - Port numbers:         port 8080, localhost:3000
 *   - Complexity notation:  O(n²), Θ(n log n)
 *   - Bit-widths:           32-bit, 64-bit
 *   - Technical identifiers: p99, L4, IPv4
 *
 * Unit conversion is NOT performed. "1.2 seconds" → "1.2" and
 * "1200ms" → "1200" are distinct values that will not match.
 */
export function extractFactualNumbers(text: string): string[] {
  const noCode  = maskCodeRegions(String(text));
  const cleaned = applyExclusions(noCode);
  const results: string[] = [];
  // Match: optional currency prefix, integer with optional commas, optional decimal.
  // No trailing word-boundary required — unit suffixes (ms, s, x, K) are not
  // captured by this pattern; the regex stops naturally at the first non-digit,
  // non-comma, non-period character.
  const re = /[$₹£€]?\d[\d,]*(?:\.\d+)?/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(cleaned)) !== null) {
    const n = normalize(m[0]);
    if (n) results.push(n);
  }
  return results;
}

/**
 * Validate that every factual numeric claim in `answer` appears in at least
 * one of the provided `sources` (typically: resolvedQuestion + joined evidence
 * content).
 *
 * Every extracted number in the answer is checked — the function never stops
 * early on the first failure.
 *
 * Returns `{ valid: true, unsupported: [] }` when no factual numbers are
 * found in the answer (nothing to validate).
 *
 * Unit conversion is NOT supported. A source "1.2 seconds" does NOT validate
 * an answer "1200ms" — they normalize to distinct strings "1.2" and "1200".
 * This is intentional for this version; unit-aware normalization is deferred.
 */
export function validateNumericProvenance(
  answer: string,
  sources: string[],
): { valid: boolean; unsupported: string[] } {
  const answerNums = extractFactualNumbers(answer);
  if (answerNums.length === 0) {
    return { valid: true, unsupported: [] };
  }

  const sourceNums = new Set<string>();
  for (const src of sources) {
    for (const n of extractFactualNumbers(src)) {
      sourceNums.add(n);
    }
  }

  const unsupported: string[] = [];
  for (const n of answerNums) {
    if (!sourceNums.has(n)) {
      unsupported.push(n);
    }
  }

  return { valid: unsupported.length === 0, unsupported };
}
