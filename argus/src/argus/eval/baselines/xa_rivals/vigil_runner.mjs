// VIGIL (norbert351/vigil @3bfad2d) in the XA arena — its own replay decision path.
//
// Run as `node vigil_runner.mjs` with the VIGIL clone as working directory. Loads VIGIL's own,
// unmodified `src/risk.js` (planOrders) and `src/regime.js` (crossAssetRegime) and applies them
// exactly as its own backtest does (`src/backtest.js:128-160`): once per UTC day on daily closes,
// with the day's historical Fear & Greed, plus the backtest's risk-off "slice" sells
// (backtest.js:154-160, re-stated here because that glue lives inside runBacktest's loop).
// Differences from its replay, all stated: the book is the arena's (targets are passed as the
// arena weights — `targets` is a user input to planOrders), the drawdown passed is the book's
// real drawdown (its live agent does this; its backtest passes 0), and crypto keys map to the
// arena's perpetual sleeve. VIGIL never shorts; SELL on a crypto key reduces the long.
import path from "node:path";
import readline from "node:readline";
import { pathToFileURL } from "node:url";

const out = process.stdout;
console.log = (...a) => process.stderr.write(a.join(" ") + "\n");
console.info = console.log;

const repo = process.cwd();
const { planOrders } = await import(pathToFileURL(path.join(repo, "src/risk.js")).href);
const { crossAssetRegime } = await import(pathToFileURL(path.join(repo, "src/regime.js")).href);

const KEY = {
  rnvda: "spot:RNVDAUSDT", raapl: "spot:RAAPLUSDT", rtsla: "spot:RTSLAUSDT",
  rspy: "spot:RSPYUSDT", rqqq: "spot:RQQQUSDT", btc: "spot:BTCUSDT", eth: "spot:ETHUSDT",
};
const ASSET = { rnvda: "equity", raapl: "equity", rtsla: "equity", rspy: "index", rqqq: "index",
                btc: "crypto", eth: "crypto" };
const PERP = { btc: "BTCUSDT", eth: "ETHUSDT" };
const SPOT = { rnvda: "RNVDAUSDT", raapl: "RAAPLUSDT", rtsla: "RTSLAUSDT", rspy: "RSPYUSDT",
               rqqq: "RQQQUSDT" };

const hist = {};          // key -> [[ts, close]]
let fng = [];             // [ts, value]
let targets = null;
const counts = { days: 0, orders: 0, riskOff: 0, riskOn: 0, neutral: 0, breaker: 0 };

function push(msg) {
  for (const [k, sym] of Object.entries(KEY)) {
    const b = msg.bars[sym];
    if (b) (hist[k] ||= []).push([msg.ts, b[3]]);
  }
  for (const p of msg.fng || []) fng.push(p);
}

function dailyClose(k, ts) {
  const h = hist[k] || [];
  for (let j = h.length - 1; j >= 0; j--) if (h[j][0] <= ts) return h[j][1];
  return null;
}

function step(msg) {
  const ts = msg.ts;
  if ((ts / 3600000) % 24 !== 0) return { orders: [], action: "HOLD" };
  counts.days += 1;
  const book = msg.book;
  const nav = book.nav;
  const prices = {};
  const priceNum = {};
  for (const k of Object.keys(KEY)) {
    const cur = dailyClose(k, ts), prev = dailyClose(k, ts - 86400000);
    if (cur && isFinite(cur)) {
      prices[k] = { lastMicro: Math.round(cur * 1e6), chg24: prev ? cur / prev - 1 : 0, asset: ASSET[k] };
      priceNum[k] = cur;
    }
  }
  const f = fng.filter((p) => p[0] <= ts - 3600000);
  const fear = f.length ? f[f.length - 1][1] : null;
  const regime = crossAssetRegime(prices, fear);
  counts[regime.regime === "risk-off" ? "riskOff" : regime.regime === "risk-on" ? "riskOn" : "neutral"] += 1;

  const posVal = {};
  const valueOf = {};
  for (const k of Object.keys(KEY)) {
    let v = 0;
    if (PERP[k]) v = (book.perp[PERP[k]] || 0) * (msg.bars["perp:" + PERP[k]]?.[3] || 0);
    else if (SPOT[k]) v = (book.spot[SPOT[k]] || 0) * (priceNum[k] || 0);
    valueOf[k] = v;
    if (v > 0) posVal[k] = { qty: 0, value: BigInt(Math.round(v * 1e6)),
                             price: BigInt(Math.round((priceNum[k] || 0) * 1e6)),
                             chg24: prices[k]?.chg24 ?? 0, weight: 0 };
  }
  if (!targets) {
    targets = {};
    for (const [k, v] of Object.entries(valueOf)) if (v > 0) targets[k] = Math.round((v / nav) * 1e6);
  }
  const drawdown = book.peak_nav > 0 ? Math.max(0, (book.peak_nav - nav) / book.peak_nav) : 0;
  const risk = planOrders({ nav: nav * 1e6, drawdown, positions: posVal, prices, targets,
                            night: false, fearGreed: fear });
  if (risk.breaker) counts.breaker += 1;
  let orders = risk.orders;
  if (regime.regime === "risk-off") {
    const slice = ["btc", "eth", "rtsla", "rnvda"]
      .filter((k) => posVal[k] && Number(posVal[k].value) > 0)
      .map((k) => ({ action: "SELL", key: k, usdMicro: Math.round(Math.min(Number(posVal[k].value), nav * 0.06 * 1e6)), reason: "regime-fear-slice" }));
    orders = [...slice, ...orders];
  }
  const perpGross = Object.entries(book.perp).reduce((s, [sym, q]) => s + Math.abs(q) * (msg.bars["perp:" + sym]?.[3] || 0), 0);
  let free = Math.max(0, book.cash - perpGross);
  const held = { ...valueOf };
  const res = [];
  for (const o of orders) {
    const usd = Number(o.usdMicro) / 1e6;
    if (!(usd >= 0.1)) continue;
    const buy = o.action === "BUY" || o.action === "HEDGE";
    if (!buy) {
      const amt = Math.min(usd, held[o.key] || 0);
      if (amt <= 0) continue;
      held[o.key] -= amt; free += amt;
      if (PERP[o.key]) res.push({ kind: "perp", symbol: PERP[o.key], usd: -amt });
      else if (SPOT[o.key]) res.push({ kind: "spot", symbol: SPOT[o.key], usd: -amt });
    } else {
      const amt = Math.min(usd, free / 1.001);
      if (amt <= 0) continue;
      free -= amt; held[o.key] = (held[o.key] || 0) + amt;
      if (PERP[o.key]) res.push({ kind: "perp", symbol: PERP[o.key], usd: amt });
      else if (SPOT[o.key]) res.push({ kind: "spot", symbol: SPOT[o.key], usd: amt });
    }
  }
  counts.orders += res.length;
  return { orders: res, action: res.length ? (regime.regime === "risk-off" ? "RISK_OFF" : "REBALANCE") : "HOLD",
           note: res.length ? `${regime.note} | ${risk.rationale}`.slice(0, 240) : "" };
}

const rl = readline.createInterface({ input: process.stdin });
for await (const line of rl) {
  if (!line.trim()) continue;
  const msg = JSON.parse(line);
  let reply = null;
  try {
    if (msg.type === "init") reply = { ok: true };
    else if (msg.type === "bar") push(msg);
    else if (msg.type === "step") { push(msg); reply = step(msg); }
    else if (msg.type === "finish") reply = { counts, targets };
  } catch (e) {
    reply = { error: String(e && e.stack || e).slice(0, 2000) };
  }
  if (reply) out.write(JSON.stringify(reply) + "\n");
  if (msg.type === "finish") break;
}
