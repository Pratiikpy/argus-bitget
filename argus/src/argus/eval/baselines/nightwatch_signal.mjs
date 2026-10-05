// NIGHTWATCH AI's directional signal, run unmodified from its clone on saved inputs (build-list 5.4).
//
// drained69/nightwatchai (MIT, a Season 2 AI Trading Desk entry) produces its signal with
// LocalNightwatchEngine.research (src/domain.js:1996-2005): runSkillPack over five deterministic
// skill readers, then synthesizeSignal. Its live server builds the market row from Bitget's R-pair
// spot candles (server/providers/bitget.mjs computeIndicators, mergeMarketRow) and a Yahoo daily
// macro snapshot (server/providers/macro.mjs). This script builds the same objects from inputs
// saved by eval/nightwatch_comparison.py and calls the same functions; the three classifiers that
// server/market-context.mjs:44-56 keeps private are restated as written there.
//
// Not reproduced: its news store and the live overlay (server/live-enhance.mjs: spot book depth
// and a market-intel snapshot), which exist only at the moment of a live call.
//
//   NIGHTWATCH_DIR=<clone> node nightwatch_signal.mjs <inputs.json> <out.json>
import { readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { pathToFileURL } from 'node:url'

const DIR = process.env.NIGHTWATCH_DIR
if (!DIR) throw new Error('set NIGHTWATCH_DIR to the NIGHTWATCH AI clone')
const domain = await import(pathToFileURL(join(DIR, 'src/domain.js')).href)
const bitget = await import(pathToFileURL(join(DIR, 'server/providers/bitget.mjs')).href)

// server/market-context.mjs:44-56, private there, restated as written
const classifyVolatility = (atrPct) => (atrPct >= 4 ? 'HIGH' : atrPct >= 2 ? 'MED' : 'LOW')
function classifyMomentum(change24h, volumeZ) {
  const move = Math.abs(change24h ?? 0)
  if (move >= 3 || (volumeZ != null && volumeZ >= 1.5)) return 'HIGH'
  if (move >= 1 || (volumeZ != null && volumeZ >= 0.75)) return 'MED'
  return 'LOW'
}
function classifyLiquidity(spreadBps, volumeUsd24h) {
  const spreadOk = spreadBps == null || spreadBps <= 10
  if (spreadOk && (volumeUsd24h ?? 0) >= 50_000_000) return 'HIGH'
  if (spreadOk && (volumeUsd24h ?? 0) >= 5_000_000) return 'MED'
  return 'LOW'
}

// server/providers/macro.mjs:84-93, the risk regime from the daily snapshot
function macroAt(daily, atMs) {
  const one = (name) => {
    const rows = (daily[name] || []).filter(([ts]) => ts <= atMs)
    if (rows.length < 2) return null
    const [, last] = rows[rows.length - 1]
    const [, prev] = rows[rows.length - 2]
    return { last, changePct: Number((((last - prev) / prev) * 100).toFixed(2)) }
  }
  const [dxy, spx, ndx, vix, ust10y] = ['dxy', 'spx', 'ndx', 'vix', 'ust10y'].map(one)
  if (!(dxy || spx || ndx || vix || ust10y)) return null
  const vixLvl = vix?.last ?? null
  const vixJump = vix?.changePct ?? null
  const eqChg = spx?.changePct ?? ndx?.changePct ?? null
  const riskRegime =
      (vixLvl != null && vixLvl >= 25) ? 'RISK_OFF'
    : (vixJump != null && vixJump >= 10) ? 'RISK_OFF'
    : (eqChg != null && eqChg <= -0.75) ? 'RISK_OFF'
    : (vixLvl != null && vixLvl >= 20 && eqChg != null && eqChg < 0) ? 'RISK_OFF'
    : (eqChg != null && eqChg >= 0.4 && (vixLvl == null || vixLvl < 20)) ? 'RISK_ON'
    : 'NEUTRAL'
  return { dxy, spx, ndx, vix, ust10y, riskRegime, live: true, stale: false, source: 'yahoo-daily-saved' }
}

const closed = (candles, atMs) => candles.filter((c) => c.ts + 3_600_000 <= atMs).slice(-200)

async function rowFor(symbol, candles, atMs) {
  const bars = closed(candles, atMs)
  if (bars.length < 50) return null
  const ind = await bitget.computeIndicators(symbol, bars)
  if (!ind) return null
  const volumeUsd24h = bars.slice(-24).reduce((s, c) => s + (c.volume || 0), 0)
  const ticker = { last: ind.last, changePct24h: ind.change24h, volumeUsd24h, spreadBps: null }
  const base = domain.DEMO_UNIVERSE.find((u) => u.symbol === symbol)
  if (!base) return null
  const row = bitget.mergeMarketRow(base, ticker, ind)
  row.volatility = classifyVolatility(row.atrPct)
  row.momentum = classifyMomentum(row.change24h, ind.volumeZ)
  row.liquidity = classifyLiquidity(row.spreadBps, volumeUsd24h)
  row.indicators = ind
  row.volumeUsd24h = volumeUsd24h
  return row
}

const [inPath, outPath] = process.argv.slice(2)
const inputs = JSON.parse(readFileSync(inPath, 'utf8'))
const engine = new domain.LocalNightwatchEngine()
const out = []
for (const inst of inputs.instants) {
  const row = await rowFor(inst.symbol, inputs.candles[inst.symbol] || [], inst.at_ms)
  if (!row) {
    const listed = domain.DEMO_UNIVERSE.some((u) => u.symbol === inst.symbol)
    out.push({ key: inst.key, skipped: listed ? 'fewer than 50 closed bars' : 'not in its universe' })
    continue
  }
  const btc = await rowFor('BTC', inputs.candles.BTC || [], inst.at_ms)
  const context = {
    universe: [row, ...(btc ? [btc] : [])],
    macro: macroAt(inputs.macro_daily, inst.at_ms),
    fearGreed: null, btcChange24h: btc?.change24h ?? null, news: [], newsBySymbol: {},
    memory: domain.initialSession().memory,  // what a fresh user is given (src/domain.js:227)
  }
  const { report } = engine.research({ intent: 'research', asset: inst.symbol, question: `${inst.symbol}?`, context })
  out.push({ key: inst.key, direction: report.signal.direction, status: report.signal.status,
             confidence: report.signal.confidence, netEdge: report.signal.netEdge })
}
writeFileSync(outPath, JSON.stringify(out) + '\n')
console.log(`${out.length} instants, ${out.filter((r) => r.status === 'SIGNAL').length} signals`)
