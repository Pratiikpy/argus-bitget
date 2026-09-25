'use strict';
// HedgeAgents (JansenAnalytics/hedge-agents @8b87aaf, a replication of arXiv 2502.13165) in the
// XA arena — the path it runs without an LLM.
//
// Run as `node hedgeagents_runner.cjs [equal|optimizer]` with the clone as working directory.
// Without a model every LLM call in the replication fails over to its own coded fallback:
//   * Budget Allocation Conference (src/conferences/bac.cjs:147-160): Otto's decision fails and
//     `_fallbackOttoDecision` returns equal weights; the optimiser output is computed but unused.
//   * Extreme Market Conference (src/conferences/emc.cjs): the trigger is pure arithmetic
//     (`checkTrigger`, 5% one-day / 10% three-day) and is run here unmodified; every step after
//     it fails over to "Hold".
// Mode `equal` applies what the replication does keyless. Mode `optimizer` applies the weights of
// its own deterministic optimiser (`domain.runPortfolioOptimizer` -> math.cjs `optimizePortfolio`,
// lambda1 0.5, lambda2 0.3) with the neutral 5% forecast its BAC uses when a report fails
// (bac.cjs:74) — the paper's allocation machinery with the LLM forecasts removed.
// Sleeves: Dave = the crypto perpetuals, Bob = the rToken basket (the "equity" analyst); the
// arena book's 10% cash stays outside the conference. BAC every 30 days (config/schedule.json).
const path = require('path');
const readline = require('readline');

const out = process.stdout;
process.env.LOG_LEVEL = 'error';
console.log = (...a) => process.stderr.write(a.join(' ') + '\n');
console.info = console.log;

const repo = process.cwd();
const MODE = process.argv[2] || 'equal';
const domain = require(path.join(repo, 'src/tools/domain.cjs'));
const { ExtremeMarketConference } = require(path.join(repo, 'src/conferences/emc.cjs'));
const schedule = require(path.join(repo, 'config/schedule.json'));
const portfolio = require(path.join(repo, 'config/portfolio.json'));

const EQ = ['spot:RNVDAUSDT', 'spot:RAAPLUSDT', 'spot:RTSLAUSDT'];
const hist = { btc: [], eq: [] }; // daily closes (00:00 UTC)
const last = {};
let startTs = null;
let lastBac = null;
const counts = { bac: 0, emc_triggers: 0, emc_actions: 0, days: 0 };
const allocations = [];

function push(msg) {
  for (const [k, b] of Object.entries(msg.bars)) last[k] = b[3];
  if ((msg.ts / 3600000) % 24 === 0) {
    if (last['spot:BTCUSDT']) hist.btc.push(last['spot:BTCUSDT']);
    const eq = EQ.map((k) => last[k]).filter((v) => v > 0);
    if (eq.length === EQ.length) hist.eq.push(eq.reduce((a, b) => a + b, 0));
  }
}

function change(series, n) {
  if (series.length <= n) return 0;
  return series[series.length - 1] / series[series.length - 1 - n] - 1;
}

async function step(msg) {
  const ts = msg.ts;
  if (startTs === null) startTs = ts;
  if ((ts / 3600000) % 24 !== 0) return { orders: [], action: 'HOLD' };
  counts.days += 1;
  const trig = ExtremeMarketConference.checkTrigger({
    'BTC-USD': { day1: change(hist.btc, 1), day3cumulative: change(hist.btc, 3) },
    '^DJI': { day1: change(hist.eq, 1), day3cumulative: change(hist.eq, 3) },
  }, schedule);
  if (trig.triggered) counts.emc_triggers += 1; // keyless EMC final decision: Hold
  const days = Math.round((ts - startTs) / 86400000);
  if (lastBac === null) lastBac = 0;
  if (days - lastBac < (schedule.bac?.intervalDays || 30)) {
    return { orders: [], action: trig.triggered ? 'EMC_HOLD' : 'HOLD',
             note: trig.triggered ? trig.reason : '' };
  }
  lastBac = days;
  counts.bac += 1;
  let w = { Dave: 0.5, Bob: 0.5 };
  if (MODE === 'optimizer') {
    const n = Math.min(hist.btc.length, hist.eq.length, 120);
    const res = await domain.runPortfolioOptimizer({
      analystForecasts: { Dave: { projected_return_pct: 5 }, Bob: { projected_return_pct: 5 } },
      allAssetData: { Dave: { symbol: 'BTC-USD', closes: hist.btc.slice(-n) },
                      Bob: { symbol: '^DJI', closes: hist.eq.slice(-n) } },
      config: portfolio,
    });
    if (!res.error && res.optimalWeights) {
      const s = (res.optimalWeights.Dave || 0) + (res.optimalWeights.Bob || 0);
      if (s > 0) w = { Dave: res.optimalWeights.Dave / s, Bob: res.optimalWeights.Bob / s };
    }
  }
  allocations.push({ day: days, ...w });
  const book = msg.book;
  const nav = book.nav;
  const risky = nav * 0.9;
  const orders = [];
  const cryptoNow = {};
  let cryptoTot = 0;
  for (const sym of ['BTCUSDT', 'ETHUSDT']) {
    const v = (book.perp[sym] || 0) * (msg.bars['perp:' + sym]?.[3] || 0);
    cryptoNow[sym] = v; cryptoTot += v;
  }
  for (const sym of ['BTCUSDT', 'ETHUSDT']) {
    const share = cryptoTot > 0 ? cryptoNow[sym] / cryptoTot : (sym === 'BTCUSDT' ? 0.625 : 0.375);
    const d = risky * w.Dave * share - cryptoNow[sym];
    if (Math.abs(d) >= 50) orders.push({ kind: 'perp', symbol: sym, usd: d });
  }
  const eqNow = {};
  let eqTot = 0;
  for (const k of EQ) {
    const sym = k.split(':')[1];
    const v = (book.spot[sym] || 0) * (last[k] || 0);
    eqNow[sym] = v; eqTot += v;
  }
  const sells = [], buys = [];
  for (const [sym, v] of Object.entries(eqNow)) {
    const share = eqTot > 0 ? v / eqTot : 1 / EQ.length;
    const d = risky * w.Bob * share - v;
    if (Math.abs(d) >= 50) (d < 0 ? sells : buys).push({ kind: 'spot', symbol: sym, usd: d });
  }
  orders.push(...sells, ...buys);
  return { orders, action: 'BAC_REBALANCE',
           note: `BAC (${MODE}) Dave ${(w.Dave * 100).toFixed(1)}% / Bob ${(w.Bob * 100).toFixed(1)}% of the risky sleeve` };
}

const rl = readline.createInterface({ input: process.stdin });
(async () => {
  for await (const line of rl) {
    if (!line.trim()) continue;
    const msg = JSON.parse(line);
    let reply = null;
    try {
      if (msg.type === 'init') reply = { ok: true };
      else if (msg.type === 'bar') push(msg);
      else if (msg.type === 'step') { push(msg); reply = await step(msg); }
      else if (msg.type === 'finish') reply = { counts, allocations, mode: MODE };
    } catch (e) {
      reply = { error: String((e && e.stack) || e).slice(0, 2000) };
    }
    if (reply) out.write(JSON.stringify(reply) + '\n');
    if (msg.type === 'finish') break;
  }
})();
