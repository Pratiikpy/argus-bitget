/**
 * Call OpenAlice's real yfinance income-statement fetcher, live, in its own process.
 *
 * Not imported by ARGUS. OpenAlice (TraderAlice/OpenAlice, AGPL-3.0) is run from its clone; this
 * file only calls it and is not a copy of it. The fetcher is `YFinanceIncomeStatementFetcher` in
 * `packages/opentypebb/src/providers/yfinance/models/income-statement.ts`, the one its agent's
 * `equity` tool uses for an income statement (`src/tool/equity.ts`, provider "yfinance"). It has
 * no date or as-of parameter; it returns what Yahoo reports now. The response is written to disk
 * the moment it arrives and every score is computed from that file.
 *
 * Usage: npx tsx openalice_runner.mts --clone DIR --tickers NVDA,AAPL --out out.json
 */
import { writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

const arg = (name: string): string => {
  const i = process.argv.indexOf(`--${name}`);
  if (i < 0 || i + 1 >= process.argv.length) throw new Error(`--${name} is required`);
  return process.argv[i + 1];
};

const clone = resolve(arg("clone"));
const tickers = arg("tickers").split(",").map((t) => t.trim().toUpperCase()).filter(Boolean);
const outPath = arg("out");
const modulePath = resolve(clone, "packages/opentypebb/src/providers/yfinance/models/income-statement.ts");
const { YFinanceIncomeStatementFetcher: F } = await import(pathToFileURL(modulePath).href);

const out: Record<string, unknown> = {
  runner: "openalice_runner", clone: "OpenAlice", fetched_at: new Date().toISOString(), tickers: {},
};
for (const ticker of tickers) {
  const started = performance.now();
  let response: unknown;
  try {
    const query = F.transformQuery({ symbol: ticker, period: "quarter", limit: 5 });
    response = { rows: F.transformData(query, await F.extractData(query, null)) };
  } catch (error) {
    response = { error: error instanceof Error ? `${error.name}: ${error.message}` : String(error) };
  }
  (out.tickers as Record<string, unknown>)[ticker] = {
    response, elapsed_s: (performance.now() - started) / 1000,
  };
  // Write-through: a live answer is saved before the next request is made.
  writeFileSync(outPath, JSON.stringify(out, null, 1), "utf-8");
  await new Promise((r) => setTimeout(r, 500));
}
