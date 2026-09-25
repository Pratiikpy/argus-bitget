/**
 * Run TradePilot-AI's own self-evolution review step, from an unmodified local clone, over a
 * stream of decisions ARGUS supplies.
 *
 *   node tradepilot_review_runner.mjs <clone> <typescript-module-dir> < in.json > out.jsonl
 *
 * TradePilot-AI (Azedfx/TradePilot-AI, an S2 entry naming "Review & Self-Evolution") carries no
 * licence, so nothing of it is copied here. This file reads `server/review/review.service.ts` from
 * the clone the caller names, transpiles it in memory with the TypeScript compiler (CommonJS,
 * `experimentalDecorators`, exactly the transform its own `nest build` applies), and runs the class
 * body as written. Three imports are satisfied by shims, because they are framework and database
 * plumbing rather than review logic:
 *
 *   @nestjs/common               `Injectable()` is DI metadata (a no-op decorator here) and
 *                                `NotFoundException` an Error subclass;
 *   ../db/research.repository    a type-only import the compiler elides;
 *   ../generated/prisma/client   the Prisma-generated `ThesisBias` enum, re-declared with the three
 *                                members its schema defines (prisma/schema.prisma).
 *
 * The repository object handed to the constructor answers `findRecentReviewFindings(excludeId,
 * limit)` the way `server/db/research.repository.ts:375-386` does — the most recent `limit` review
 * findings of *other* sessions, newest first, each carrying `data.patterns` — from the decisions
 * this stream has already reviewed. Nothing else in the service is stubbed.
 *
 * Input (stdin, JSON): {"decisions": [{"id": "<seq>", "patterns": ["<rule>", ...]}, ...]} in
 * chronological order. Output (stdout, one JSON object per decision): the `recurring` patterns
 * `findRecurring` returned and the length of the checklist `buildChecklist` emitted.
 */

import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { join } from "node:path";

const [clone, tsDir] = process.argv.slice(2);
if (!clone || !tsDir) {
  console.error("usage: node tradepilot_review_runner.mjs <clone> <typescript-module-dir>");
  process.exit(2);
}

const require = createRequire(import.meta.url);
const ts = require(tsDir);
const sourcePath = join(clone, "server", "review", "review.service.ts");
const source = readFileSync(sourcePath, "utf8");
const { outputText } = ts.transpileModule(source, {
  compilerOptions: {
    module: ts.ModuleKind.CommonJS,
    target: ts.ScriptTarget.ES2020,
    experimentalDecorators: true,
    emitDecoratorMetadata: false,
  },
  fileName: "review.service.ts",
});

class NotFoundException extends Error {}
const shims = {
  "@nestjs/common": { Injectable: () => (target) => target, NotFoundException },
  "../db/research.repository": {},
  "../generated/prisma/client": {
    ThesisBias: { BULLISH: "BULLISH", BEARISH: "BEARISH", NEUTRAL: "NEUTRAL" },
  },
};
const moduleShim = { exports: {} };
const localRequire = (name) => {
  if (name in shims) return shims[name];
  throw new Error(`review.service.ts imported ${name}, which this runner does not provide`);
};
new Function("require", "module", "exports", outputText)(
  localRequire, moduleShim, moduleShim.exports,
);
const { ReviewService } = moduleShim.exports;

const input = JSON.parse(readFileSync(0, "utf8"));
const findings = []; // newest last; each is one saved "review" finding
const repo = {
  async findRecentReviewFindings(excludeSessionId, limit = 20) {
    return findings
      .filter((f) => f.sessionId !== excludeSessionId)
      .slice(-limit)
      .reverse();
  },
};
const service = new ReviewService(repo);

for (const decision of input.decisions) {
  const patterns = decision.patterns.map((id) => ({ id, message: id }));
  // The order `review()` itself uses (review.service.ts:85-103): detect, find recurring against
  // the past, build the checklist, then save this session's finding for the next one to read.
  const recurring = await service.findRecurring(decision.id, patterns);
  const checklist = service.buildChecklist(patterns);
  findings.push({ sessionId: decision.id, category: "review", data: { patterns } });
  process.stdout.write(
    JSON.stringify({
      id: decision.id,
      recurring: recurring.map((r) => ({ id: r.id, occurrences: r.occurrences })),
      checklist: checklist.length,
    }) + "\n",
  );
}
