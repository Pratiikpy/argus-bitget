/**
 * Export AnalogDesk's own validation grid and predictions, run from an unmodified local clone.
 *
 *   node analogdesk_export.mjs <clone> <out.json> [horizon]
 *
 * AnalogDesk (lixinde586-afk/analogdesk, an S2 entry in Decision Stress Testing) carries no licence,
 * so nothing of it is copied here: this file imports its engine from the clone the caller names and
 * writes, for every query of its pre-registered grid, the realised outcome and each of its five
 * predictors' centre and unscaled half-width, exactly as its own `probe()` returns them, plus the
 * adjusted closes its forward returns are computed from. `eval/analogstress_comparison.py` scores
 * ARGUS on the same queries from the same closes.
 */

import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const [clone, out, horizonArg] = process.argv.slice(2);
if (!clone || !out) {
  console.error("usage: node analogdesk_export.mjs <clone> <out.json> [horizon]");
  process.exit(2);
}
const load = (rel) => import(pathToFileURL(join(clone, rel)).href);
const { createEngine } = await load("src/engine/analog.mjs");
const { queryGrid, probe, pooledUncondBand, VALIDATION_DEFAULTS, ERAS } =
  await load("src/engine/validation.mjs");

const ds = JSON.parse(readFileSync(join(clone, "data-cache", "dataset.json"), "utf8"));
const engine = createEngine(ds);
const C = { ...VALIDATION_DEFAULTS, horizon: Number(horizonArg || VALIDATION_DEFAULTS.horizon) };
engine.C.horizon = C.horizon;
const { mx } = engine;

const rowsFor = (era, stride) => {
  const rows = [];
  for (const { sq, q } of queryGrid(engine, { era, stride })) {
    const r = probe(engine, sq, q, C);
    if (!r) continue;
    rows.push({
      era, sym: r.sym, sq, q, date: r.date, y: r.y, pit: r.pit, vol20: r.vol20,
      analogConformal: r.analogConformal ? { centre: r.analogConformal.centre, hw: r.analogConformal.hw } : null,
      analogRaw: r.analogRaw, uncondNamePIT: r.uncondNamePIT, volHarness: r.volHarness,
    });
  }
  return rows;
};

const calib = rowsFor("calibration", C.strideCalib);
const test = rowsFor("test", C.strideTest);
let calibEnd = 0;
for (let i = 0; i < mx.nDates; i++) if (mx.dates[i] <= ERAS.calibration.to) calibEnd = i;
const pooled = pooledUncondBand(mx, C.horizon, calibEnd - C.horizon, C.coverage);
for (const r of [...calib, ...test]) {
  r.pooledUncond = pooled ? { centre: pooled.centre, hw: pooled.half } : null;
}

const adjusted = {};
for (let s = 0; s < mx.nSym; s++) {
  const series = [];
  for (let i = 0; i < mx.nDates; i++) {
    const v = mx.priceA[s * mx.nDates + i];
    series.push(Number.isFinite(v) ? v : null);
  }
  adjusted[mx.syms[s]] = series;
}

writeFileSync(out, JSON.stringify({
  horizon: C.horizon, k: C.k, coverage: C.coverage, eras: ERAS,
  strides: { calibration: C.strideCalib, test: C.strideTest },
  dates: mx.dates, rows: [...calib, ...test], adjusted,
}));
console.log(`exported ${calib.length} calibration and ${test.length} test queries (H=${C.horizon})`);
