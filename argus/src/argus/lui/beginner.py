"""A first-time user's questions, read before any engine can mistake them for something else.

Round 42's newcomer audit (Activity/audits/round42_newcomer.md) found 17 of 40 beginner questions
refused and 9 of 14 follow-ups answered without their topic. Most had a vetted answer already in
`lui/newcomer.py`; they never reached it, because routes earlier in the chain caught a word first:
"scared" went to the loss-sizing reader, "lost" and "hold" to the decision record, "money" to the
record's refusal, "SIP" to the contract list, "price moves 2%" to a quote.

This layer runs first and is deliberately narrow. It answers only:

- **Safety-critical basics** (seed phrase, scams, KYC, leverage a friend uses, a loss that hurts),
  where a wrong route is worse than no answer;
- **Beginner questions that had no answer** (emergency fund, gas fees, tokenised stocks, buying
  with $100, monthly buying, telling a fake exchange from a real one);
- **Short follow-ups**, read with the turn before them: "and for futures?" after a fee answer,
  "so what if price moves 2% against me" after 50x, "is it safe to give it" after a passport.

Everything else falls through untouched. Where `newcomer.py` already has the vetted words, they
are reached by their own canonical question rather than copied here, so one answer stays one
answer. A figure stated here carries its source.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Final

Lines = list[str]

_TYPOS: Final = ((r"\bhw\b", "how"), (r"\bwat\b|\bwut\b|\bwht\b", "what"), (r"\bwit\b", "with"),
                 (r"\bdollrs?\b", "dollars"),
                 (r"\bwall?it\b|\bwalet\b|\bwallat\b", "wallet"), (r"\bn\b", "and"),
                 (r"\bu\b", "you"), (r"\bur\b", "your"), (r"\bsayin\b", "saying"),
                 (r"\bmsg\b", "message"), (r"\bsumthin\b", "something"), (r"\br\b", "are"))


def _plain(text: str) -> str:
    out = text
    for typo, word in _TYPOS:
        out = re.sub(typo, word, out, flags=re.I)
    return out


def _first(canonical: str) -> Lines | None:
    from argus.lui.newcomer import first_reply, reply

    said = first_reply(canonical)
    if said:
        return said
    found = reply(canonical, named=False)
    return list(found.lines) if found is not None and found.lines else None


# --- first questions --------------------------------------------------------------------------

_SEED: Final = re.compile(r"\bseed\s*phrase\b|\brecovery\s+phrase\b|\b(?:12|twelve|24|twenty[\s-]?"
                          r"four)\s+(?:secret\s+)?words\b", re.I)


def _seed(text: str) -> Lines | None:
    if not _SEED.search(text):
        return None
    if re.search(r"\b(?:forgot|forgotten|lost|lose)\b", text, re.I):
        return _first("i forgot my seed phrase is my money gone forever")
    said = _first("what is a seed phrase")
    if said is None:
        return None
    return [*said, "Keep it safe like this: write the words on paper (or stamp them in metal), "
                   "keep it somewhere private, and never type it into a website, a chat, an "
                   "email or a photo — no real support team will ever ask for it."]


_LOSS_FELT: Final = re.compile(
    r"\b(?:lost|down)\s+(?:like\s+|about\s+|almost\s+|over\s+)?(?:\$?\d[\d,.]*\s*%?|half|most|"
    r"everything|a\s+lot)[^?]{0,60}\b(?:feel\w*|sick|scared|panic\w*|stress\w*|tension|cry\w*|"
    r"can'?t\s+sleep|depress\w*|awful|terrible)\b|\bpaisa\s+doob\s+gaya\b|\bdoob\s+gay[ae]\b|"
    r"\bbahut\s+tension\b", re.I)


def _loss_felt(text: str) -> Lines | None:
    if not _LOSS_FELT.search(text):
        return None
    return ["Bottom line: a loss like that is a real shock, and feeling sick about it is normal — "
            "the most useful thing today is to stop: no new trades, no more money in, no "
            "leverage.",
            "The urge right now is to win it back quickly; that is the move that most often turns "
            "a bad loss into a far worse one.",
            "When you are calmer, look at what is left with numbers: what you still hold, whether "
            "any of it is leveraged (a leveraged position can still be liquidated), and how much "
            "more it could fall in a bad week — ask \"how much could I lose on BTC in a bad week "
            "with $500\" with your own coin and amount.",
            "Talk to someone you trust. If you cannot stop thinking about trading, free "
            "confidential help exists: in the US the National Problem Gambling Helpline, "
            "1-800-GAMBLER; in the UK, GamCare, 0808 8020 133."]


_WIN_BACK: Final = re.compile(r"\b(?:win|make|get|earn)\s+(?:it|my\s+money|that|the\s+money|it\s+"
                              r"all)\s+back\b|\brecover\s+(?:my\s+)?loss\w*\s+(?:fast|quick\w*)\b|"
                              r"\bchase\s+(?:my\s+|the\s+)?loss\w*\b", re.I)


def _win_back(text: str) -> Lines | None:
    if not _WIN_BACK.search(text):
        return None
    return ["Bottom line: no — putting everything back in to win it back fast is the single most "
            "common way a bad loss becomes a much bigger one.",
            "A loss already taken does not change what the next trade is worth; trading bigger or "
            "with leverage to recover it only raises the odds of a second, larger loss.",
            "Take a break of at least a few days, decide how much you could lose without it "
            "hurting, and only then trade again — small, on spot, no leverage."]


_FRIEND_LEVERAGE: Final = re.compile(
    r"\b(?P<x>\d{1,3})\s*x\b[^?]{0,60}\b(?:ok|okay|safe|good|fine|for\s+me|should\s+i)\b|"
    r"\b(?:friend|everyone|people|he|she|they)\b[^?]{0,30}\b(?:uses?|trades?|does|using)\s+"
    r"(?P<x2>\d{1,3})\s*x\b", re.I)


def _friend_leverage(text: str) -> Lines | None:
    m = _FRIEND_LEVERAGE.search(text)
    if m is None:
        return None
    times = int(m.group("x") or m.group("x2"))
    if times <= 1:
        return None
    move = 100 / times
    return [f"Bottom line: for a beginner, no — at {times}x a move of about {move:.1f}% against "
            "the position wipes out the margin and the exchange closes it (liquidation), and "
            f"crypto moves {move:.0f}% in a day often.",
            f"That {move:.1f}% is before fees and the exchange's maintenance margin, which close "
            "you a little earlier still. Someone else surviving it says how their trades went, "
            "not what the leverage is.",
            "Start on spot with no leverage, where the most you can lose is what you paid; if you "
            "ever use leverage, keep it low (2x needs about a 50% move to liquidate) and set a "
            "stop-loss first."]


_SCAM_DOUBLE: Final = re.compile(
    r"\b(?:send|deposit|transfer)\b[^?]{0,40}\b(?:get|receive|gets?)\b[^?]{0,30}\b(?:doubl\w*|"
    r"back\s+doubl\w*|2x|twice|triple\w*)\b|\b(?:doubl\w*|2x|triple\w*)\s+(?:your|my)\s+(?:usdt|"
    r"money|crypto|btc|eth|coins?)\b", re.I)


def _scam_double(text: str) -> Lines | None:
    if not _SCAM_DOUBLE.search(text):
        return None
    return ["Bottom line: it is a scam — do not send anything. No real person, company or "
            "exchange doubles money you send them; once crypto is sent it cannot be pulled back.",
            "The pattern is always the same: a message from a stranger (WhatsApp, Telegram, a "
            "\"support\" account), a promise of a fixed big return, and urgency. Some send a "
            "small \"payout\" first so you send more.",
            "Block the sender, do not click their links, and if you already sent something, "
            "report it to the exchange you sent from and to your local police cyber-crime unit."]


_EMERGENCY: Final = re.compile(r"\bemergency\s+(?:fund|money|savings|cash)\b|\brainy[\s-]day\b",
                               re.I)


def _emergency(text: str) -> Lines | None:
    if not _EMERGENCY.search(text):
        return None
    return ["Bottom line: a common rule of thumb is three to six months of your essential "
            "expenses (rent, food, bills, debt payments) in cash or an insured bank account "
            "before any money goes into crypto.",
            "If your income is unsteady, or others depend on you, lean to the higher end; pay "
            "off high-interest debt first, since a credit-card rate is a sure loss a trade "
            "rarely beats.",
            "Only money beyond that, and that you could lose most of without changing your "
            "plans, belongs in something as volatile as crypto."]


_GAS: Final = re.compile(r"\bgas\s+(?:fees?|prices?)\b[^?]{0,40}\b(?:what|why|mean|high|"
                         r"expensive)\b|\bwhat(?:'?s|\s+is|\s+are)\s+(?:a\s+)?gas(?:\s+fees?)?\b",
                         re.I)


def _gas(text: str) -> Lines | None:
    if not _GAS.search(text):
        return None
    lines = ["Bottom line: a gas fee is what you pay a blockchain's validators to process your "
             "transaction — on Ethereum it is charged in ETH, and it rises when many people want "
             "the same limited block space at once.",
             "It is high in busy moments (a popular token launch, a market crash) and low when "
             "the network is quiet; a simple transfer costs less gas than a complex swap.",
             "Buying and selling inside Bitget pays Bitget's trading fee, not gas; you meet a "
             "network fee only when you withdraw coins to a wallet, and the withdrawal screen "
             "shows it before you confirm."]
    try:
        from argus.lui import onchain

        now, _sources, _data = onchain.gas("eth gas now")
        if now:
            lines.append("Right now on Ethereum: " + now[0].removeprefix("Bottom line: "))
    except Exception:
        pass
    return lines


_TOKENISED: Final = re.compile(r"\btokeni[sz]ed\s+stocks?\b|\brtokens?\b|\bstock\s+tokens?\b",
                               re.I)


def _tokenised(text: str) -> Lines | None:
    if not (_TOKENISED.search(text) and re.search(r"\breal\b|\bactual\b|\bown\b|\bshares?\b|"
                                                  r"\bsomething\s+else\b|\bsame\s+as\b", text,
                                                  re.I)):
        return None
    return ["Bottom line: no — a tokenised stock on Bitget (an rToken, such as rNVDA) is a token "
            "that tracks the share's price, not the share itself: you are not a registered "
            "shareholder and have no vote.",
            "It trades around the clock, so outside US market hours its price can drift from "
            "the stock's last close and jump back at the open.",
            "If owning the actual share matters to you (voting, holding it at a broker under your "
            "name), buy it through a regular stock broker."]


_FIRST_BUY: Final = re.compile(
    r"\bhow\s+(?:do\s+i|to|can\s+i|should\s+i)\s+(?:start\s+(?:to\s+)?)?(?:buy|get)\s+(?:some\s+)?"
    r"(?:btc|bitcoin|eth|ether|crypto)\b[^?]{0,40}?(?:\$\s?(?P<a>\d{1,4})\b|(?P<b>\d{1,4})\s*"
    r"(?:dollars?|usd|bucks)\b)?", re.I)


def _first_buy(text: str) -> Lines | None:
    m = _FIRST_BUY.search(text)
    if m is None:
        return None
    amount = m.group("a") or m.group("b")
    nervous = re.search(r"\bnervous|\bscared|\bfirst\s+time|\bbeginner|\bnew\s+to\b|\bnever\b",
                        text, re.I)
    if not (nervous or (amount and int(amount) <= 1000)):
        return None
    said = _first("how do i buy bitcoin first time")
    if said is None:
        return None
    if amount:
        fee = int(amount) * 0.001
        said = [*said, f"With ${int(amount)}, Bitget's standard spot fee of 0.10% is about "
                       f"${fee:.2f}; feeling nervous is a good reason to start this small."]
    return said


_SIP: Final = re.compile(r"\bsip\b|\bsystematic\s+investment\b|\bhar\s+mahine\b|\bevery\s+month"
                         r"\b[^?]{0,30}\b(?:buy|invest|put)\b|\b(?:monthly|recurring|regular)\s+"
                         r"(?:buy\w*|invest\w*|purchase\w*)\b", re.I)


def _sip(text: str) -> Lines | None:
    if not (_SIP.search(text) and re.search(r"\bcrypto|\bbitcoin|\bbtc\b|\beth\b|\bcoin", text,
                                            re.I)):
        return None
    return ["Bottom line: yes — buying a fixed amount every month (what a SIP does with mutual "
            "funds) works with crypto too; it is called dollar-cost averaging, and it spreads "
            "your buying over many prices instead of betting on one day's.",
            "You can do it by hand — the same amount of BTC on spot on the same day each month — "
            "or with a recurring-buy feature if your app offers one; check its fee per purchase.",
            "It does not make crypto safe: a SIP into something that falls 70% still falls 70%. "
            "Keep the monthly amount to what you could lose without it hurting."]


_EXCHANGE_SCAM: Final = re.compile(
    r"\b(?:know|tell|check|see|find\s+out|spot)\b[^?]{0,30}\b(?:if|whether)\b[^?]{0,20}\b"
    r"(?:an?|the|this)?\s*(?:exchange|platform|app|site|website|broker)\b[^?]{0,20}\b(?:is\s+)?"
    r"(?:a\s+)?(?:scam|fake|legit|real|safe|fraud\w*)\b|\b(?:exchange|platform)\s+is\s+(?:a\s+)?"
    r"(?:scam|fake|legit)\b", re.I)


def _exchange_scam(text: str) -> Lines | None:
    if not _EXCHANGE_SCAM.search(text):
        return None
    return ["Bottom line: check five things before you send an exchange any money — and if any "
            "fails, do not use it.",
            "1. Is it registered with the financial regulator where you live? Regulators publish "
            "lists of registered firms; a name missing from it is a warning.",
            "2. Did you reach it yourself, from the official app store or a URL you typed — not "
            "from a link in a message or an ad? Fake copies of real exchanges are common.",
            "3. Does it promise fixed or guaranteed returns? Real exchanges do not.",
            "4. Can you withdraw? Deposit a small amount, then withdraw it, before anything more. "
            "Scam platforms let you deposit and find reasons you cannot take money out.",
            "5. Does it publish proof-of-reserves and have a long public record? Large exchanges, "
            "Bitget among them, publish proof-of-reserves reports on their own websites."]


_KYC_WHY: Final = re.compile(r"\bwhy\b[^?]{0,40}\b(?:want|need|ask\w*|require\w*)\b[^?]{0,20}\b"
                             r"(?:my\s+)?(?:passport|id\b|identity|kyc|selfie|aadhaar)\b|\bwhy\s+"
                             r"(?:is\s+)?kyc\b", re.I)


def _kyc_why(text: str) -> Lines | None:
    if not _KYC_WHY.search(text):
        return None
    return ["Bottom line: because the law requires it — exchanges must verify who their users "
            "are under anti-money-laundering rules (\"know your customer\", KYC), the same "
            "reason a bank asks for ID.",
            "Without it most exchanges limit or block deposits, withdrawals and trading; which "
            "features need it is set by the exchange's rules for your country.",
            "Give it only inside the official app or website — never to someone who messages "
            "you, and never through a link sent to you."]


_BITCOIN_SAFE: Final = re.compile(r"\bwhat\s+is\s+bitcoin\b.{0,80}\b(?:safe|scam\w*|beginner"
                                  r"\w*|trust)\b|\bbitcoin\b[^?]{0,40}\bsafe\s+for\s+beginners?"
                                  r"\b", re.I)


def _bitcoin_safe(text: str) -> Lines | None:
    if not _BITCOIN_SAFE.search(text):
        return None
    return ["Bottom line: Bitcoin is a digital currency that runs on a public network no single "
            "company controls; the coin itself is not a scam, but its price swings hard — it has "
            "fallen more than 70% from a peak more than once.",
            "Most beginners who lose money to scams lose it to people, not to Bitcoin: fake "
            "investment groups, \"guaranteed\" returns, fake exchanges and fake support accounts.",
            "Safe first steps: buy only on a well-known exchange you reached yourself, start with "
            "an amount you could lose, use no leverage, and never share your password, 2FA code "
            "or seed phrase."]


_WALLET: Final = re.compile(r"\bwhat\s+(?:is|are|even\s+is)\s+(?:a\s+|an\s+)?(?:crypto"
                           r"(?:currency)?\s+)?wallets?\b|\b(?:do|will)\s+i\s+(?:even\s+|really\s+)?"
                           r"need\s+(?:a\s+|an\s+)?(?:crypto(?:currency)?\s+)?wallet\b|\bwhich\s+"
                           r"(?:crypto(?:currency)?\s+)?wallet\b", re.I)


def _wallet(text: str) -> Lines | None:
    if not _WALLET.search(text):
        return None
    return _first("what even is a crypto wallet do i need one to use bitget")


_FOMO: Final = re.compile(
    r"\b(?P<name>[A-Za-z]{2,10})\s+(?:will|is\s+going\s+to|gonna|could|can|to)\s+(?:hit|reach|"
    r"go\s+to|touch)\s+\$?(?P<target>\d[\d,.]*k?)\b", re.I)
_HYPE: Final = re.compile(r"\bfomo\b|\beveryone\s+(?:says|is\s+saying|keeps\s+saying)\b|"
                          r"\b(?:telegram|tiktok|youtube|twitter|influencer)\b[^?]{0,30}\bsays?\b",
                          re.I)


def _fomo(text: str) -> Lines | None:
    """"FOMO!!! everyone says SOL will hit 500, should i buy right now" got an honest first line
    under 2,700 characters of percentiles and path matches (round 42 newcomer, 19)."""
    m = _FOMO.search(text)
    if m is None or not _HYPE.search(text):
        return None
    from argus.lui.research import research_symbols

    found = research_symbols(m.group("name"))[0]
    if not found:
        return None
    symbol = found[0]
    raw = m.group("target").replace(",", "").lower()
    target = float(raw.rstrip("k")) * (1000 if raw.endswith("k") else 1)
    coin = symbol.removesuffix("USDT")
    try:
        from argus.market.bitget import fetch_tickers

        now = float(fetch_tickers()[symbol].last)
    except Exception:
        now = 0.0
    need = f", so {target:,.0f} needs a {target / now - 1:+.0%} move" if now > 0 else ""
    out = [f"Bottom line: nobody knows whether {coin} reaches {target:,.0f}"
           + (f" — it is {now:,.2f} now{need}" if now > 0 else "")
           + ". \"Everyone says\" is how a top feels: by the time a coin is everywhere, many of "
             "the people saying it already own it."]
    try:
        from argus.lui.research.drawdown_sizing import max_drawdown
        from argus.lui.research.rule_test import daily_closes

        stamps, closes, _ = daily_closes(symbol)
        worst, peak, trough = max_drawdown({symbol: 1.0}, {symbol: closes})
        out.append(f"{coin} has fallen {abs(worst):.0%} from a peak before ({stamps[peak]:%b %Y} "
                   f"to {stamps[trough]:%b %Y}, Bitget's daily closes) — hype can reverse that "
                   "hard.")
    except Exception:
        pass
    out.append("If you buy anyway: a small amount you could lose, on spot, no leverage, and "
               "decide before you buy at what loss you would sell. Buying in a rush because "
               "others are excited is the most common way beginners buy near a top.")
    return out


_HACKED: Final = re.compile(r"\b(?:account|wallet|exchange\s+account)\b[^?]{0,20}\b(?:is\s+|"
                           r"gets?\s+|was\s+|got\s+)?(?:hack\w*|compromised|stolen|drained)\b|\b"
                           r"(?:hack\w*|compromised)\b[^?]{0,20}\b(?:my\s+)?(?:account|wallet)\b",
                           re.I)


def _hacked(text: str) -> Lines | None:
    if not _HACKED.search(text):
        return None
    return ["Bottom line: act at once — contact the exchange's support through its official app or "
            "website (never a link someone sends you) and ask them to freeze the account; then "
            "change your email and exchange passwords and reset two-factor authentication.",
            "Speed matters because crypto sent out cannot be pulled back; an exchange can freeze "
            "an account and sometimes stop a withdrawal that has not left yet.",
            "Afterwards, report it to your local police cyber-crime unit, and from then on use an "
            "authenticator app (not SMS), a unique password and a withdrawal whitelist if the "
            "exchange offers one. A self-custody wallet that was drained cannot be frozen by "
            "anyone — move whatever is left to a new wallet with a new seed phrase."]


_FX: Final = re.compile(r"\b(?:exchange|conversion)\s+rate\b[^?]{0,40}\b(?:crypto|bitcoin|buy\w*)"
                        r"\b|\b(?:buy\w*)\b[^?]{0,30}\bcrypto\b[^?]{0,30}\b(?:exchange|conversion)\s+"
                        r"rate\b", re.I)


def _fx(text: str) -> Lines | None:
    if not _FX.search(text):
        return None
    return ["Bottom line: crypto is priced in US dollars or dollar stablecoins (USDT) almost "
            "everywhere, so buying it with reais, rupees or lira involves a currency conversion "
            "first — and that conversion has its own rate and cost.",
            "If you pay by card or bank, the provider converts your money at its own rate, often "
            "a little worse than the market rate, plus any card fee; if you buy USDT person-to-"
            "person (P2P), the seller sets the rate. Compare the final amount of coin you get, "
            "not the headline price.",
            "Then the exchange charges its trading fee on the purchase itself (0.10% a side on "
            "Bitget spot at the standard level). Being afraid of losing everything is a good "
            "reason to start with a small amount, on spot, with no leverage."]


_FIRSTS: Final[tuple[Callable[[str], Lines | None], ...]] = (
    _seed, _wallet, _loss_felt, _win_back, _scam_double, _friend_leverage, _emergency, _gas,
    _tokenised,
    _first_buy, _sip, _exchange_scam, _kyc_why, _bitcoin_safe, _hacked, _fx, _fomo)


# --- follow-ups, read with the turn before -------------------------------------------------------

def _short(text: str) -> bool:
    return len(text.split()) <= 14


def _follow_seed(text: str, before: str) -> Lines | None:
    if not _SEED.search(before):
        return None
    if re.search(r"\bscreenshot|\bphoto|\bpicture|\bcloud|\bgoogle\s+drive|\bnotes?\s+app|\bemail"
                 r"|\bsave\s+it\s+(?:on|in)\b", text, re.I):
        return ["Bottom line: no — do not screenshot or photograph it. Photos sync to cloud "
                "accounts and get seen by apps; anyone who gets that image can empty the wallet.",
                "Write the words on paper (two copies in two safe places is better than one), "
                "check each word, and keep them offline.",
                "Never type them into a website, a chat or an email — no real support team will "
                "ever ask for them."]
    if re.search(r"\blose\b|\blost\b|\bforget\b|\bforgot\b", text, re.I):
        return _first("i forgot my seed phrase is my money gone forever")
    return None


def _follow_fees(text: str, before: str) -> Lines | None:
    if not re.search(r"\bfees?\b", before, re.I):
        return None
    m = re.match(r"^\W*(?:and|what\s+about|how\s+about)?\s*(?:the\s+)?(?:fees?\s+)?(?:for|on|in)?"
                 r"\s*(?P<what>futures|perps?|perpetuals?|leverage\s+trading|derivatives|spot)"
                 r"\s*(?:fees?)?\W*$", text, re.I)
    if m is None:
        return None
    if m.group("what").lower() == "spot":
        return ["Bottom line: on spot, Bitget's standard (VIP 0) fee is 0.10% a side — about $0.10 "
                "on a $100 buy (Bitget's fee schedule)."]
    return ["Bottom line: on Bitget's USDT perpetual futures the standard (VIP 0) fee is 0.02% for "
            "a maker order (one that waits on the order book) and 0.06% for a taker order (one "
            "that fills at once) — Bitget's own contract list.",
            "Futures also carry funding: every few hours (every 8 on most contracts) longs pay "
            "shorts or the other way round, depending on the market, which can cost more than "
            "the trading fee if you hold for weeks.",
            "And futures use leverage, which can liquidate you; for a first trade, spot is the "
            "simpler place to start."]


def _follow_leverage_move(text: str, before: str) -> Lines | None:
    lev = re.search(r"\b(\d{1,3})\s*x\b", before, re.I)
    move = re.search(r"\b(?:moves?|drops?|falls?|goes)\b[^?]{0,15}?(\d{1,2}(?:\.\d+)?)\s*%|"
                     r"(\d{1,2}(?:\.\d+)?)\s*%\s*(?:against|down|drop|move)", text, re.I)
    if lev is None or move is None:
        return None
    times = int(lev.group(1))
    pct = float(move.group(1) or move.group(2))
    hit = times * pct
    if hit >= 100:
        verdict = (f"at {times}x, a {pct:g}% move against you is a {hit:.0f}% loss on the margin — "
                   "the whole margin is gone and the exchange liquidates the position, usually a "
                   "little before that because of fees and the maintenance margin.")
    else:
        verdict = (f"at {times}x, a {pct:g}% move against you is a {hit:.0f}% loss on the margin "
                   f"you put up, and a {100 / times:.1f}% move would liquidate it.")
    return [f"Bottom line: {verdict}",
            f"On spot with no leverage the same {pct:g}% move is a {pct:g}% loss you can wait "
            "out; leverage multiplies the move, it does not change the coin."]


def _follow_order_type(text: str, before: str) -> Lines | None:
    if not (re.search(r"\blimit\b", before, re.I) and re.search(r"\bmarket\b", before, re.I)):
        return None
    if not re.search(r"\bwhich\b|\bbetter\b|\bshould\s+i\b|\bbeginner|\bfor\s+me\b", text, re.I):
        return None
    return ["Bottom line: as a beginner, a limit order at or near the current price — you choose "
            "the most you will pay, so a fast market cannot fill you somewhere surprising.",
            "A market order is fine for a small buy of a big coin like BTC, where the gap between "
            "buy and sell prices is tiny; it costs the taker fee (0.10% on Bitget spot) and fills "
            "at once.",
            "Avoid market orders on small, thinly traded coins: there the price you get can be "
            "noticeably worse than the one you saw."]


def _follow_tax(text: str, before: str) -> Lines | None:
    if not re.search(r"\btax\w*\b|\bitr\b|\btds\b", before, re.I):
        return None
    india = bool(re.search(r"\bindia\b|\btds\b|\binr\b|\brupee", before, re.I))
    if re.search(r"\bhold\b|\bnever\s+sell\b|\bdon'?t\s+sell\b|\bjust\s+keep\b", text, re.I):
        return (["Bottom line: in India, simply holding is not taxed — the 30% tax and the 1% TDS "
                 "apply when you transfer the crypto: sell it, swap it or spend it.",
                 "Gifts are the exception to watch: crypto received as a gift can be taxed in "
                 "the receiver's hands above a limit.",
                 "Keep a record of what you paid for each coin; it is the only cost you can "
                 "deduct when you eventually sell. Not tax advice."] if india else
                ["Bottom line: in most countries simply holding crypto is not taxed — tax usually "
                 "applies when you sell, swap or spend it at a gain.",
                 "Rules differ by country (some tax staking rewards or airdrops when received); "
                 "check your own country's tax authority. Not tax advice."])
    if re.search(r"\bswap\w*|\bexchang\w*\s+(?:one|a)\s+coin|\bconvert\w*|\btrade\s+one\b", text,
                 re.I):
        return (["Bottom line: in India, yes — swapping one coin for another counts as a transfer "
                 "of the first coin, so any gain on it is taxed at 30%, and 1% TDS applies to "
                 "the swap.",
                 "Losses cannot be set off against gains, so each swap is taxed on its own gain. "
                 "Not tax advice."] if india else
                ["Bottom line: in many countries, yes — swapping one coin for another is treated "
                 "as selling the first, so a gain on it is taxable even though no cash reached "
                 "you. Check your own country's rules. Not tax advice."])
    return None


def _follow_kyc(text: str, before: str) -> Lines | None:
    if not re.search(r"\bkyc\b|\bpassport\b|\bid\b|\bidentity\b|\baadhaar\b|\bselfie\b", before,
                     re.I):
        return None
    if not re.search(r"\bsafe\b|\btrust\b|\bshould\s+i\s+(?:give|upload|send)\b|\brisk\w*\b",
                     text, re.I):
        return None
    return ["Bottom line: giving ID to a large, regulated exchange through its official app is "
            "normal and required by law, but no company's data is perfectly safe — so give it "
            "only there, and protect the account itself.",
            "Upload it only inside the official app or the website you typed yourself; never "
            "send a passport photo to anyone who messages you as \"support\" — that is a common "
            "identity-theft scam.",
            "Then turn on two-factor authentication (an authenticator app, not SMS if you can) "
            "and use a unique password, so a leaked ID alone cannot open your account."]


def _follow_scammer(text: str, before: str) -> Lines | None:
    if not re.search(r"\bguarant\w*|\btelegram|\bwhatsapp|\bsignal\s+group|\b\d+x\b|\bpromis\w*",
                     before, re.I):
        return None
    if not re.search(r"\bscam\w*|\blegit\b|\bfake\b|\btrust\b|\bhonest\b", text, re.I):
        return None
    return ["Bottom line: someone who guarantees a return is the sign itself — nobody can "
            "guarantee what a market does, so a guarantee is a sales pitch, and most often a "
            "scam.",
            "Other signs: they contacted you first, they want you to move fast, they ask you to "
            "send crypto to them or to use an app or site they choose, and they show screenshots "
            "of profits you cannot check.",
            "A real adviser is licensed and can be looked up with your country's regulator; a "
            "Telegram stranger cannot. Do not send money, and block them."]


_FOLLOWS: Final[tuple[Callable[[str, str], Lines | None], ...]] = (
    _follow_seed, _follow_fees, _follow_leverage_move, _follow_order_type, _follow_tax,
    _follow_kyc, _follow_scammer)


def early(text: str, prior: Sequence[str]) -> Lines | None:
    """A beginner's answer, or None to let the rest of the console read the question."""
    plain = _plain(text)
    if prior and _short(plain):
        before = " ".join(prior[-2:])
        for follow in _FOLLOWS:
            said = follow(plain, before)
            if said:
                return said
    found = _first_of(plain)
    if found is None:
        return None
    lines, answered = found
    if answered in _WHOLE:
        return lines
    # "is there tax on crypto, and what happens if my account is hacked?" asks two things; the
    # Turkish original was refused whole (round 42 newcomer, 38): each half gets its answer, in
    # the order asked
    parts = [x for x in re.split(r"\?\s+|,?\s+and\s+(?=what|how|if|is|do|can|should|which|why)",
                                 plain) if len(x.split()) >= 3]
    if len(parts) < 2:
        return lines
    answers: list[Lines] = []
    for part in parts:
        hit = _first_of(part)
        other = hit[0] if hit is not None else _first(part)
        if other:
            answers.append(other)
    if len(answers) < 2 or answers[0][0] == answers[1][0]:
        return lines
    head, second = answers[0], answers[1]
    return [*head[:2], "Also asked — " + second[0].removeprefix("Bottom line: "), *second[1:2]]


def _first_of(text: str) -> tuple[Lines, Callable[[str], Lines | None]] | None:
    for first in _FIRSTS:
        said = first(text)
        if said:
            return said, first
    return None


_WHOLE: Final = frozenset({_seed, _wallet, _bitcoin_safe, _first_buy, _gas, _kyc_why, _fx,
                           _loss_felt, _win_back, _scam_double, _friend_leverage, _tokenised,
                           _sip, _exchange_scam, _fomo})
"""Answers that already cover the usual second half of their question ("what is bitcoin, is it
safe for beginners"); only the others are split and answered part by part."""


__all__ = ["early"]
