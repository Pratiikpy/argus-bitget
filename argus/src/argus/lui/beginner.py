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
                 (r"\bmsg\b", "message"), (r"\bsumthin\b", "something"), (r"\br\b", "are"),
                 (r"\bcoo+in\b", "coin"), (r"\bbitcon\b|\bbitcoim\b|\bbtcoin\b", "bitcoin"),
                 (r"\bshud\b", "should"), (r"\b2l8\b|\btoo\s*l8\b", "too late"),
                 (r"\binve+sting\b", "investing"), (r"\bhlp\b", "help"), (r"\bsum\b", "some"),
                 (r"\bdo\s+i\s+no\b", "do i know"), (r"\bpls\b|\bplz\b", "please"),
                 (r"\bwaht\b", "what"), (r"\brugpull", "rug pull"),
                 # round 44: slang and typos that kept a clean question from its answer
                 (r"\bev(?:e)?ry?thin\b|\bevrything\b", "everything"), (r"\bevry\b", "every"),
                 (r"\brn\b", "right now"), (r"\bpayd\b", "paid"),
                 (r"\bhap+ens?\b|\bhapen\b", "happens"),
                 (r"\bliqi?dat(\w*)", r"liquidat\1"),
                 (r"\bdivid[ae]nts?\b|\bdividen\b|\bdivdends?\b", "dividend"),
                 (r"\bcand[ae]ls?\s*sti?c?ks?\b|\bcandlestiks?\b", "candlestick"),
                 (r"\bwrng\b|\brong\b", "wrong"), (r"\badd?ress\b|\badres{1,2}\b", "address"),
                 (r"\b2\b(?=\s+(?:the\s+)?(?:wrong|another|different|diff|other)\b)", "to"),
                 (r"\by\b(?=\s+(?:are|is|do|does|did|not|would|should|can|my|the|so|some)\b)",
                  "why"))


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


# --- round 43: safety, basics and country tax ---------------------------------------------------
#
# Round 43's newcomer audit (Activity/audits/round43_newcomer.md) found 11 safety questions and 27
# basics refused or misread. Every fact below was read from its source on 2026-10-06:
#
# - Bitget P2P safety (Bitget help centre, "How to Prevent P2P Trading Fraud?" and "Common
#   Questions About Bitget P2P Trading"): orders are held in escrow; release coins only after the
#   payment is confirmed in your own account; screenshots and e-mails of payment can be faked; the
#   payer's name must match the verified name on their account; keep the chat inside the order.
# - Bitget 2FA reset (Bitget support, "How to Reset Security Items When Multiple Security Items Are
#   Lost?" and "How to Recover Your Bitget Account After Losing Access?"): a lost authenticator is
#   reset through the support page's self-service flow; withdrawals are paused for 24 hours after
#   a security change; Bitget's "Why Is My Withdrawal Restricted or Suspended?" gives the same 24
#   hours after a password or 2FA change and after a P2P purchase, and says the withdrawal limit
#   follows verification and VIP level (no figure is stated here: the withdrawal screen shows
#   the one that applies to the account).
# - Bitget security settings (Bitget Academy, "Advanced Account Security"): an anti-phishing code
#   that appears in genuine e-mails and texts, and a withdrawal address whitelist.
# - Crypto ATMs (US Federal Trade Commission, "Bitcoin ATMs: A payment portal for scammers", Sept
#   2024, and consumer.ftc.gov): scammers who claim a fine, a tax, a hacked account or a "safe
#   locker" tell people to deposit cash in a bitcoin ATM; "only scammers will tell you to do that".
# - Reporting: US reportfraud.ftc.gov and ic3.gov; UK "Report Fraud", which replaced Action Fraud
#   on 4 Dec 2025 (reportfraud.police.uk, 0300 123 2040; Scotland: Police Scotland on 101) and the
#   FCA's scam reporting.
# - UK tax (gov.uk, "Check if you need to pay tax when you sell cryptoassets", and "Capital Gains
#   Tax allowances": annual exempt amount GBP 3,000, tax year 6 April to 5 April; disposal means
#   selling, swapping for another cryptoasset, spending or giving away; the 30-day rule).
# - US tax (irs.gov/filing/digital-assets: digital assets are property; one year or less is short
#   term, more than one year long term; Form 8949; 1099-DA from brokers for 2025 transactions).
# - Nigeria (Nigeria Tax Act 2025, in force 1 Jan 2026, tax body now the Nigeria Revenue Service;
#   reputable reports disagree on the capital-gains rate for individuals, so no rate is stated).
# - Brazil (Receita Federal: sales of up to R$35,000 in a month are exempt from the capital-gains
#   tax, above that the gain is taxed from 15%; holdings of R$5,000 or more per coin go in the
#   annual return, "Bens e Direitos", group 08; IN RFB 1,888 reporting by exchanges).
# - Inflation: US consumer prices rose 9.1% in the twelve months to June 2022, the most in about 40
#   years (US Bureau of Labor Statistics); Bitcoin's Bitget daily closes were 47,215.90 USDT on 31
#   Dec 2021 and 16,557.99 on 31 Dec 2022, a fall of 64.9%.
# - Stablecoins: TerraUSD (UST) lost its peg from 7 to 9 May 2022 and was near 0.10 by 13 May; USDC
#   fell to about 0.87 on 11 March 2023 after Circle disclosed reserves held at Silicon Valley
#   Bank and regained $1 four days later (CNBC, 11 and 13 March 2023).
# - Lump sum versus averaging: Vanguard, "Dollar-cost averaging just means taking risk later"
#   (2012, refreshed since): lump sum beat a 12-month phase-in in about two thirds of historical
#   periods in the US, UK and Australia. Stocks and bonds, not crypto.
# - Bitcoin is divisible to eight decimals (one satoshi is 0.00000001 BTC) and its supply is capped
#   at 21 million (the Bitcoin protocol).
#
# Fees repeated here are the ones this file already carries: 0.10% a side on Bitget spot at the
# standard level.


def _fixed(pattern: re.Pattern[str], answer: tuple[str, ...],
           also: re.Pattern[str] | None = None,
           never: re.Pattern[str] | None = None) -> Callable[[str], Lines | None]:
    """A reader that answers ``answer`` when ``pattern`` (and ``also``, if given) match."""
    def read(text: str) -> Lines | None:
        if not pattern.search(text):
            return None
        if also is not None and not also.search(text):
            return None
        if never is not None and never.search(text):
            return None
        return list(answer)
    return read


_P2P: Final = re.compile(r"\bp2p\b|\bpeer[\s-]to[\s-]peer\b", re.I)
_P2P_ASK: Final = re.compile(r"\b(?:safe|safety|scam\w*|risk\w*|trust\w*|fraud\w*|legit|dangerous|"
                             r"careful|secure|avoid|reliable)\b", re.I)
_P2P_A: Final = (
    "Bottom line: it can be safe, if you release coins only after the money is in your own bank "
    "account, and never because someone shows you proof. Most P2P losses are a buyer faking a "
    "payment, or a bank payment that is later reversed.",
    "The usual tricks: a screenshot or an e-mail \"proof\" of payment (easy to fake), a bank "
    "transfer that is recalled after you release the coins, a payer whose name is not the name "
    "on their verified account, and pressure to move the chat to WhatsApp or Telegram.",
    "Safe habits: check the money in your own banking app, not in a message; keep everything "
    "inside the platform's order and chat, so its escrow and support can help; if the payer's "
    "name does not match, do not release, contact support.",
    "Escrow protects the coins while an order is open; it cannot undo a bank payment that comes "
    "back after you have released them.",
)
_P2P_S: Final = (
    "Bottom line: P2P means you trade with another person, so the person is the risk.",
    "Only give the coins after the money is really in your own bank app. A picture of a payment "
    "can be fake.",
    "Stay inside the platform's chat and order. If the name on the payment is not theirs, stop "
    "and ask support.",
)

_SIGNAL_GROUP: Final = re.compile(
    r"\b(?:signals?|trading|vip|crypto|investment|investing|pump)\s+(?:group|channel|chat|"
    r"community)s?\b|\b(?:telegram|whatsapp|discord)\s+(?:group|channel|server)s?\b", re.I)
_SIGNAL_ASK: Final = re.compile(r"\b(?:join\w*|added|invit\w*|should|safe|legit|scam\w*|worth|"
                                r"trust\w*|pay|paid|paying|free)\b", re.I)
_SIGNAL_A: Final = (
    "Bottom line: be very careful — \"signal groups\" on Telegram or WhatsApp are one of the "
    "most common ways people get scammed, and a friend adding you does not make it safe: your "
    "friend may be a victim, or their account may have been taken over.",
    "How it goes wrong: the admins push a coin, members buy, and the admins sell into the rush "
    "(a pump and dump); or you are asked to pay for a \"VIP\" tier; or a link leads to a fake "
    "\"trading platform\" that lets you deposit but not withdraw.",
    "Rules: never pay to join, never send money or coins to an admin or a \"manager\", never log "
    "in or connect a wallet through a link from the group, and never copy a signal with money "
    "you cannot afford to lose. Leaving the group costs nothing.",
    "A quick test: ask your friend whether they have actually withdrawn profit to their own bank "
    "account. Screenshots of wins prove nothing.",
)
_SIGNAL_S: Final = (
    "Bottom line: be careful. Many signal groups are scams, even when a friend invites you.",
    "Often the owners tell you to buy a coin, then sell it to you when the price jumps.",
    "Do not pay, do not send money to anyone in the group, and do not click its links. You can "
    "just leave.",
)

_PHISH_CHANNEL: Final = re.compile(r"\b(?:e-?mail|sms|text(?:\s+message)?|message|dm|whatsapp|"
                                   r"call(?:ed)?|letter|notification|link)\b", re.I)
_PHISH_BAIT: Final = re.compile(r"\b(?:click|link|verify|suspend\w*|locked|frozen|confirm\s+your|"
                                r"update\s+your|urgent\w*|within\s+\d+\s+hours?)\b", re.I)
_PHISH_BRAND: Final = re.compile(r"\b(?:bitget|binance|coinbase|kraken|okx|bybit|exchange|crypto"
                                 r"\s+platform|wallet)\b|\bmy\s+account\b", re.I)
_PHISH_GOT: Final = re.compile(r"\b(?:got|received|get|receive|sent|saying|says?|claim\w*|asks?|"
                               r"asking|came|arrived|real|fake|legit|genuine)\b", re.I)
_PHISH_A: Final = (
    "Bottom line: treat it as phishing — do not click the link, do not type your password or any "
    "code, and do not reply.",
    "To check without trusting the message: open the Bitget app yourself, or type the address "
    "into your browser, and look for the same notice in your account messages. If it is real, "
    "it will be there.",
    "Bitget lets you set an anti-phishing code that appears in its genuine e-mails and texts; a "
    "message without your code is fake. No real exchange employee asks for your password, your "
    "2FA code or your recovery phrase.",
    "If you already clicked and typed something: change your password from the official app at "
    "once, reset two-factor authentication, and contact support from inside the app.",
)
_PHISH_S: Final = (
    "Bottom line: it is probably a fake message. Do not click it.",
    "Open the app yourself, or type the website address yourself, and see if the same message "
    "is there. Real companies never need your password or your codes.",
    "If you already typed your password, change it now from the real app.",
)

_FORGOT_2FA: Final = re.compile(
    r"\b(?:forgot|forgotten|lost|lose|losing|reset|new\s+phone|changed\s+(?:my\s+)?phone|"
    r"broke\w*|deleted|reinstall\w*|wiped|stolen|can'?t\s+(?:find|access|use|get|open)|cannot|"
    r"not\s+working|doesn'?t\s+work|no\s+longer)\b[^?]{0,50}\b(?:2fa|two[\s-]?factor|2[\s-]?step|"
    r"authenticator|google\s+auth\w*|authentication\s+code|verification\s+code)\b|"
    r"\b(?:2fa|two[\s-]?factor|2[\s-]?step|authenticator|google\s+auth\w*)\b[^?]{0,30}\b(?:forgot|"
    r"lost|reset|not\s+working|stopped|expired|gone)\b", re.I)
_FORGOT_2FA_A: Final = (
    "Bottom line: you can get back in, but only through the exchange's own reset process — "
    "expect to prove who you are again (ID and a face check) and to wait.",
    "On Bitget: use the account-recovery or \"reset security item\" option on the login page or "
    "in the support centre's self-service section, and follow its steps. After a security "
    "change Bitget pauses withdrawals for 24 hours, which is normal.",
    "Be wary of anyone who messages you offering to \"unlock\" the account: that is the scam. "
    "Real support never starts the chat and never asks for your password or codes.",
    "For next time: save the backup key shown when you set up the authenticator, and keep it "
    "offline.",
)
_FORGOT_2FA_S: Final = (
    "Bottom line: you can get back in, but only through the exchange's own reset page.",
    "They will ask you to prove who you are again with your ID and your face. It can take a "
    "while.",
    "Do not trust anyone who messages you to \"fix\" it. That is a scam.",
)

_SCAM_COIN: Final = re.compile(
    r"\b(?:know|tell|check|spot|identify|sure|figure\s+out|find\s+out|detect|recogni[sz]e)\b"
    r"[^?]{0,40}\b(?:coin|token|crypto(?:currency)?|project|memecoin|altcoin|shitcoin)s?\b"
    r"[^?]{0,30}\b(?:scam\w*|fake|rug\w*|legit\w*|fraud\w*|real|safe|trustworthy)\b|"
    r"\b(?:avoid|spot|identify|recogni[sz]e)\b[^?]{0,20}\b(?:scam|fake|rug\s*pull)\s*"
    r"(?:coins?|tokens?|projects?)\b|\bis\s+(?:this|that|the|a)\s+(?:coin|token|project)\s+"
    r"(?:a\s+)?(?:scam|fake|legit|rug)\b", re.I)


def _scam_coin(text: str) -> Lines | None:
    if not _SCAM_COIN.search(text):
        return None
    return _first("how do i know if a coin is a scam")


_RUG: Final = re.compile(r"\brug\s*-?\s*pull\w*|\brugged\b", re.I)
_RUG_A: Final = (
    "Bottom line: a rug pull is when the people behind a coin or project take the money and "
    "vanish — they drain the funds from the trading pool and abandon the project — so the "
    "price falls to nearly zero and you cannot get out.",
    "Warning signs: an anonymous team; most of the supply held by a few wallets; no independent "
    "audit; a promised or guaranteed return; pressure to buy now; and a contract that lets "
    "buyers in but blocks selling (a \"honeypot\").",
    "To lower the risk: stay with coins listed on a large exchange, check the holders and the "
    "liquidity on a blockchain explorer, and never put in more than you could lose entirely. "
    "No check is perfect.",
)
_RUG_S: Final = (
    "Bottom line: a rug pull is when the owners of a coin take everyone's money and disappear.",
    "The price drops to almost nothing and you cannot sell.",
    "Signs: nobody knows who the owners are, a few wallets hold most coins, and they promise "
    "big safe profits. Only use coins from big exchanges, and risk only what you can lose.",
)

_ATM: Final = re.compile(
    r"\b(?:crypto(?:currency)?|bitcoin|btc)\s+(?:atms?|kiosks?|machines?)\b|"
    r"\batm\b[^?]{0,60}\b(?:fine|tax|taxes|irs|police|warrant|arrest|deposit\s+cash|scam\w*)\b|"
    r"\b(?:fine|tax|taxes|irs|police|warrant|arrest)\b[^?]{0,60}\batm\b", re.I)
_ATM_A: Final = (
    "Bottom line: crypto ATMs are expensive and final — the fees are high, and once your cash "
    "has become coins sent to an address, it cannot be pulled back.",
    "They are a favourite scam tool: anyone who tells you to withdraw cash and put it in a "
    "bitcoin ATM — to pay a fine or a tax, to \"protect\" your money, to clear a warrant or fix a "
    "hacked account — is a scammer, every time. No police officer, tax office, bank or tech "
    "support worker takes payment this way (US Federal Trade Commission).",
    "If you ever use one for yourself: send only to a wallet you control, try a small amount "
    "first, and compare the price and fee with an exchange. Buying on a regulated exchange is "
    "usually far cheaper.",
    "If it has already happened, phone the ATM operator (the number is on the machine) and the "
    "police straight away, then report it to your national fraud service.",
)
_ATM_S: Final = (
    "Bottom line: crypto ATMs cost a lot, and you cannot get the money back.",
    "If someone tells you to put cash in one to pay a fine or a tax, it is a scam. Always.",
    "Hang up, tell someone you trust, and call the police.",
)

_ATM_DEMAND: Final = re.compile(r"\b(?:fine|tax|taxes|irs|police|warrant|arrest|told|tells?|"
                                r"said|asked|must|have\s+to|pay)\b", re.I)


def _atm(text: str) -> Lines | None:
    """A crypto ATM question; when someone is telling the asker to pay at one, the scam verdict
    comes first."""
    if not _ATM.search(text):
        return None
    if _ATM_DEMAND.search(text):
        return ["Bottom line: it is a scam — do not put any cash in. Anyone who tells you to pay "
                "a fine or a tax, \"protect\" your money, clear a warrant or fix a hacked account "
                "at a bitcoin ATM is a scammer, every time; no police officer, tax office, bank or "
                "tech support worker takes payment this way (US Federal Trade Commission).",
                *_ATM_A[2:]]
    return list(_ATM_A)


_SIM_SWAP: Final = re.compile(
    r"\bsim[\s-]?(?:swap\w*|jack\w*)|\bsim\s+card\b[^?]{0,40}\b(?:hack\w*|steal\w*|stolen|"
    r"clone\w*|swap\w*)|\b(?:phone\s+)?number\b[^?]{0,50}\b(?:hack\w*|steal|stolen|take\s+over|"
    r"taken\s+over|hijack\w*|swap\w*)\b|\b(?:someone|anyone|hacker|scammer|thief|criminal)\b"
    r"[^?]{0,30}\b(?:gets?|has|steals?|takes?|knows?|finds?)\s+(?:my\s+)?(?:phone\s+)?number\b",
    re.I)
_SIM_SWAP_A: Final = (
    "Bottom line: yes, it can happen — in a \"SIM swap\" a criminal persuades your phone company "
    "to move your number to their SIM, so your texts, including login codes, reach them.",
    "That is why codes sent by text message (SMS) are the weakest protection: with your number, "
    "they can ask for a password reset and receive the code.",
    "Protect yourself: use an authenticator app, not SMS, for two-factor authentication; ask your "
    "phone company to put a PIN or a port-out lock on your number; turn on the exchange's "
    "withdrawal address whitelist and anti-phishing code; and use an e-mail address for the "
    "exchange that has its own strong 2FA.",
    "If your phone suddenly loses signal for no reason, treat it as an emergency: call your "
    "carrier from another phone, then lock your exchange account.",
)
_SIM_SWAP_S: Final = (
    "Bottom line: yes. A thief can trick your phone company into giving them your number.",
    "Then the codes sent by text go to the thief. Use an authenticator app instead of text "
    "codes, and ask your phone company for a PIN on your number.",
    "Turn on the withdrawal whitelist too, so money can only go to addresses you chose.",
)

_REPORT_SCAM: Final = re.compile(
    r"\b(?:report|reporting)\b[^?]{0,30}\b(?:scam\w*|fraud\w*|crypto\s+theft|stolen\s+crypto|"
    r"been\s+scammed)\b|\bi\s+(?:was|got|have\s+been|been|am)\s+(?:just\s+)?scammed\b|"
    r"\bwho\s+(?:do|can|should)\s+i\s+(?:tell|report|call|contact)\b[^?]{0,30}\b(?:scam\w*|"
    r"fraud\w*)\b", re.I)
_REPORT_SCAM_A: Final = (
    "Bottom line: stop sending anything, keep every piece of evidence, and report it in three "
    "places — the exchange or platform you sent money from, the police or national fraud "
    "service, and the site or app where you met the scammer.",
    "Keep: screenshots of the chats, usernames, wallet addresses, transaction IDs, dates, "
    "amounts and the website address. Do not delete the conversation.",
    "Contact the exchange you sent from at once, through its official app: it can sometimes "
    "freeze funds that have not yet left the platform.",
    "US: reportfraud.ftc.gov and ic3.gov (FBI). UK: Report Fraud (reportfraud.police.uk or 0300 "
    "123 2040; in Scotland, Police Scotland on 101) and the FCA. Elsewhere: your national police "
    "cyber-crime unit or consumer authority. Beware anyone who offers to \"recover\" the money for "
    "a fee: that is a second scam.",
)
_REPORT_SCAM_S: Final = (
    "Bottom line: stop sending money, and keep all the messages and payment details.",
    "Tell the exchange you sent from, and the police. In the US also reportfraud.ftc.gov; in "
    "the UK, Report Fraud.",
    "If someone offers to get your money back for a fee, that is another scam.",
)

_EXCHANGE_TRUST: Final = re.compile(
    r"\b(?:know|tell|check|sure|figure\s+out|find\s+out|decide)\b[^?]{0,30}\b(?:exchange|"
    r"platform|app|site|broker)\b[^?]{0,30}\b(?:trustworthy|trusted|reliable|reputable|safe|good|"
    r"legit|regulated|licen[cs]ed)\b|\bwhat\s+makes\s+(?:an?\s+)?(?:crypto\s+)?exchange\s+"
    r"(?:trustworthy|safe|good|reliable)\b|\bhow\s+(?:to|do\s+i|can\s+i)\s+(?:choose|pick|find)\s+"
    r"(?:a\s+|the\s+)?(?:good\s+|safe\s+|best\s+)?(?:crypto\s+)?exchange\b", re.I)


def _exchange_trust(text: str) -> Lines | None:
    if not _EXCHANGE_TRUST.search(text):
        return None
    return _exchange_scam("how do i know if an exchange is a scam")


_PUMP: Final = re.compile(r"\bpump(?:\s+and\s+|\s*(?:&|n|'n'?)\s*|[\s-]+and[\s-]+)dump\b|"
                          r"\bpump\s*(?:&|n)\s*dump\b", re.I)
_PUMP_A: Final = (
    "Bottom line: a pump and dump is when a group pushes a small coin's price up with hype and "
    "coordinated buying, then sells its own coins to everyone who joined late — the price "
    "collapses and the latecomers are left holding.",
    "How to spot one: loud hype from groups or influencers telling you to buy now; a tiny coin "
    "with little real trading; a huge jump with no news; a countdown (\"buy at 3 pm\").",
    "How to avoid it: never buy because a group says so, skip anything you cannot explain, and "
    "stay out of groups that announce a coin before it moves. If you hear about it late, you "
    "are the one they sell to.",
)
_PUMP_S: Final = (
    "Bottom line: a pump and dump is a trick. A group makes a small coin's price jump, then "
    "sells to the people who join late.",
    "Then the price falls and the late buyers lose. If someone says \"buy now, fast\", be "
    "careful and do not.",
)

_CHART: Final = re.compile(
    r"\bhow\s+(?:do\s+i|to|can\s+i|does\s+one)\s+(?:read|understand|use|look\s+at)\s+(?:a\s+|"
    r"the\s+)?(?:price\s+|crypto\s+|trading\s+|candlestick\s+)*(?:charts?|candles?|"
    r"candlesticks?)\b|\bwhat\s+(?:is|are)\s+(?:a\s+)?candle(?:stick)?s?\b|"
    r"\bwhat\s+do\s+(?:the\s+)?(?:green|red)\s+(?:and\s+(?:green|red)\s+)?(?:bars?|candles?)\s+"
    r"mean\b", re.I)
_CHART_A: Final = (
    "Bottom line: a price chart is a record of past trades. Each candle is one time slot (a "
    "minute, an hour, a day): its thick body runs from the price at the start of the slot to "
    "the price at the end, and its thin lines (wicks) reach the highest and lowest price in it.",
    "Colours: usually green means it ended higher than it began and red lower, but some apps "
    "reverse them, so check yours. The bars along the bottom are volume, how much was traded.",
    "Start with three things: the time frame (a daily chart shows the big picture, a one-minute "
    "chart is mostly noise), the trend (higher highs and higher lows, or the reverse), and "
    "volume (a move on little trading is weaker).",
    "A chart shows what already happened; it cannot tell you what comes next, and patterns that "
    "look obvious afterwards often fail. Not advice.",
)
_CHART_S: Final = (
    "Bottom line: a chart shows how the price moved in the past. Each bar is one stretch of "
    "time, like a day.",
    "The fat part shows where the price started and ended. The thin lines show the highest and "
    "lowest price. Green usually means up, red means down.",
    "It shows the past, not the future.",
)

_SMALL_SUM: Final = re.compile(
    r"\b(?:can|could|is\s+it\s+(?:ok|okay|possible|worth)|do\s+i\s+need)\b[^?]{0,30}"
    # "can i start with like 20 bucks" (round 43 live re-ask) carries a filler word
    r"\b(?:invest\w*|start\w*|begin)\b[^?]{0,30}\b(?:(?:only|just)\s+|with\s+(?:like\s+|about\s+|"
    r"around\s+|maybe\s+|roughly\s+)?(?=\$?\d{1,3}\s*"
    r"(?:dollars?|usd|bucks|\$)?\W*$))(?:\$\s?(?P<a>\d{1,3})\b(?!\s*(?:k|,\d|m\b|million|"
    r"billion|thousand))|(?P<b>\d{1,3})\s*(?:dollars?|usd|bucks)\b)", re.I)

def _small_sum(text: str) -> Lines | None:
    m = _SMALL_SUM.search(text)
    if m is None:
        return None
    amount = int(m.group("a") or m.group("b"))
    if not 1 <= amount <= 200:
        return None
    fee = amount * 0.001
    return [f"Bottom line: yes — you can start with ${amount}; coins can be bought in small "
            "pieces, so you do not need a whole one.",
            f"But small amounts meet costs that weigh more: the trading fee is small (0.10% a "
            f"side at Bitget's standard level is about ${fee:.2f} on ${amount}), yet every market "
            "has a minimum order size, and a withdrawal fee, charged per coin and network, can "
            "be a large share of a small amount. Check both on the order and withdrawal screens "
            "before you buy.",
            "A small start is a sound way to learn on real prices without risking money that "
            "hurts to lose. Keep it to money you could lose."]


_SMALL_SUM_S: Final = (
    "Bottom line: yes, you can start with a very small amount, even $10.",
    "You can buy a small piece of a coin. Look at the smallest order allowed and at the fee "
    "to take money out, because on a tiny amount fees can matter.",
    "Only use money you can afford to lose.",
)

_DCA: Final = re.compile(
    r"\b(?:what(?:'?s|\s+is|\s+does)|explain|define|meaning\s+of)\s+(?:the\s+)?"
    r"(?:dollar[\s-]cost\s+averag\w*|dca)\b(?![^?]{0,20}\b(?:into|on|in|for)\s+[A-Z]{2,5}\b)|"
    r"\b(?:dollar[\s-]cost\s+averag\w*|dca)\b[^?]{0,30}\b(?:vs\.?|versus|or|better|worse)\b|"
    r"\b(?:vs\.?|versus|better\s+than)\s+(?:dollar[\s-]cost\s+averag\w*|dca|lump[\s-]?sum)\b|"
    r"\blump[\s-]?sum\b", re.I)
_DCA_A: Final = (
    "Bottom line: dollar-cost averaging (DCA) means splitting a sum into equal buys at regular "
    "intervals; a lump sum means investing it all at once. Neither is always better.",
    "In studies of stocks (Vanguard, 2012, refreshed since), investing the lump sum came out "
    "ahead of a 12-month phase-in in about two thirds of historical periods in the US, UK and "
    "Australia, because prices rise more often than they fall. DCA wins when prices fall soon "
    "after you start. That research is on stocks and bonds, not crypto.",
    "What DCA really does is limit the regret of putting everything in just before a fall, and "
    "it fits money you earn month by month. Pick the one you can stick to, and watch the fee on "
    "each buy.",
)
_DCA_S: Final = (
    "Bottom line: DCA means buying the same small amount again and again, say every month. A "
    "lump sum means putting all the money in on one day.",
    "For shares, putting it all in at once did better about two times out of three, but it hurts "
    "more if the price falls right after. DCA feels calmer.",
    "Pick the way you can stick to.",
)

_EXCH_WALLET: Final = re.compile(
    r"\b(?:difference|different|vs\.?|versus|which|better)\b[^?]{0,40}\bexchange\b[^?]{0,30}"
    r"\bwallet\b|\bwallet\b[^?]{0,30}\b(?:vs\.?|versus|or|and)\b[^?]{0,20}\bexchange\b[^?]{0,30}"
    r"\b(?:difference|different|better|which)\b", re.I)
_EXCH_WALLET_A: Final = (
    "Bottom line: on an exchange (like Bitget) the company keeps your coins for you and you log "
    "in with a password — easy to use and you can sell for cash, but you depend on the "
    "company. In a wallet app you hold the keys yourself through a recovery phrase — nobody can "
    "freeze your coins, and nobody can help if you lose the phrase.",
    "Use an exchange for buying, selling and trading. Use your own wallet for coins you plan to "
    "hold a long time, or to use apps on a blockchain.",
    "Beginners: buy on a regulated exchange first. Move to your own wallet only when you "
    "understand the recovery phrase, and never share it with anyone.",
)
_EXCH_WALLET_S: Final = (
    "Bottom line: an exchange keeps your coins for you, like a bank. A wallet app lets you keep "
    "them yourself.",
    "With an exchange you can reset your password if you forget it. With your own wallet, if you "
    "lose the secret words, the coins are gone.",
    "Start with an exchange.",
)

_BULL_TRAP: Final = re.compile(r"\b(?:bull|bear)\s*trap\b", re.I)
_BULL_TRAP_A: Final = (
    "Bottom line: a bull trap is a price that breaks upward and looks like the start of a rise, "
    "draws buyers in, then turns round and falls — leaving the people who bought the breakout "
    "holding a loss. (A bear trap is the mirror image: a drop that reverses upward.)",
    "It can only be recognised afterwards: at the time it looks the same as a real breakout. "
    "Signs of weakness are little trading volume behind the move and a quick fall back below "
    "the level it broke.",
    "Protection: decide where you would admit you were wrong before you buy, keep the position "
    "small, and do not buy only because the price just jumped.",
)
_BULL_TRAP_S: Final = (
    "Bottom line: a bull trap is when the price jumps up, people buy because they think it will "
    "keep rising, and then it falls.",
    "You cannot tell in advance. Buy small, and know before you buy where you would sell if you "
    "were wrong.",
)

_BUBBLE: Final = re.compile(
    r"\bis\s+(?:the\s+)?(?:bitcoin|btc|crypto(?:currency)?|ethereum|eth)\s+(?:market\s+)?"
    r"(?:in\s+)?(?:a\s+)?bubble\b", re.I)
_BUBBLE_A: Final = (
    "Bottom line: nobody knows — a bubble is only certain looking back. Bitcoin has had steep "
    "rises followed by falls of more than half (about 65% in 2022 alone: 47,216 USDT at the "
    "end of 2021 to 16,558 at the end of 2022, Bitget's daily closes), so it can behave like "
    "one.",
    "After earlier crashes it later set new highs, but that is a record, not a promise that it "
    "will again.",
    "A better question than \"is it a bubble\" is \"could I leave this money alone through a 50% "
    "to 70% fall?\". If not, the amount is too big.",
)
_BUBBLE_S: Final = (
    "Bottom line: nobody knows. A bubble is only clear after it pops.",
    "Bitcoin has dropped by more than half before, for example about 65% in 2022. It has also "
    "come back before, but that is not a promise.",
    "Only use money you could leave alone if it dropped by half.",
)

_INFLATION: Final = re.compile(
    r"\b(?:does|do|can|will|would|is|are)\s+(?:bitcoin|btc|crypto(?:currency)?)\b[^?]{0,40}"
    r"\binflation\b|\binflation\b[^?]{0,40}\b(?:does|do|can|will|would)\s+(?:bitcoin|btc|crypto)"
    r"\b", re.I)
_INFLATION_NOT: Final = re.compile(r"\b(?:vs\.?|versus|research|study|portfolio|gold)\b", re.I)
_INFLATION_A: Final = (
    "Bottom line: inflation is the general rise in prices, so the same money buys less each "
    "year (at 3% a year, something that costs $100 today costs about $103 next year). Bitcoin "
    "has not reliably protected against it.",
    "In 2022 US prices rose 9.1% over twelve months (June 2022, US Bureau of Labor Statistics, "
    "the most in about 40 years) and Bitcoin fell about 65% over the same year (Bitget's daily "
    "closes, 47,216 to 16,558 USDT).",
    "Some call Bitcoin \"digital gold\" because only 21 million can ever exist, but in that "
    "episode its price moved with risk appetite and interest rates, not with shop prices. Over "
    "longer stretches it has risen a lot, with falls that would hurt a saver.",
    "For money whose buying power must be kept, people use things like inflation-linked "
    "government bonds or savings that pay more than inflation; none of them is risk-free.",
)
_INFLATION_S: Final = (
    "Bottom line: inflation means prices go up over time, so your money buys less.",
    "Bitcoin does not reliably protect you from it. In 2022 prices rose fast, and Bitcoin fell "
    "by about two thirds.",
    "Money you must keep safe is better in things made for that, like a savings account or "
    "inflation-linked bonds.",
)

_WD_LIMIT: Final = re.compile(
    r"\b(?:limit|limits|maximum|max|cap|capped)\b[^?]{0,40}\bwithdraw\w*|\bwithdraw\w*\b[^?]{0,"
    r"30}\b(?:limit|limits|maximum|max|cap|capped)\b|\bhow\s+much\b[^?]{0,15}\b(?:can|could)\s+i"
    r"\s+withdraw\b", re.I)
_WD_LIMIT_A: Final = (
    "Bottom line: yes — an exchange sets how much you may withdraw in a day, and the limit "
    "differs by exchange, by account, and by how fully you have verified your identity (on "
    "Bitget it also follows your VIP level).",
    "I cannot see your account, so the number that applies to you is the one on Bitget's "
    "withdrawal screen and in its Help Center; do not trust a figure from a website or a "
    "message.",
    "Separate from the limit, Bitget pauses withdrawals for 24 hours after a security change "
    "(a new password or 2FA) and after a P2P purchase. That is a normal safety delay, not a "
    "block.",
)
_WD_LIMIT_S: Final = (
    "Bottom line: yes. The exchange decides how much you can take out each day.",
    "The amount depends on your account and on how much you have shown them about who you are.",
    "Look at the withdrawal screen in the app to see yours.",
)

_NOW_OR_WAIT: Final = re.compile(
    r"\b(?:buy|invest|get\s+in)\b[^?]{0,20}\bnow\b[^?]{0,20}\b(?:or|vs\.?|versus)\b[^?]{0,20}"
    r"\b(?:wait\w*|dip|later|lower)\b[^?]{0,20}\b(?:dip|drop|pullback|lower|cheaper)\b|"
    r"\bwait\s+for\s+(?:a|the)\s+(?:dip|drop|pullback)\b", re.I)
_NOW_OR_WAIT_A: Final = (
    "Bottom line: nobody can time it reliably — not you, and not this console — so the better "
    "question is how to buy so that neither answer can hurt you much.",
    "Waiting for a dip has a cost too: the price may never fall as far as you hoped, and a dip "
    "can be followed by a bigger one. Buying everything at once feels bad if the price falls "
    "the next week.",
    "A common middle path is to split the amount into equal buys over weeks or months (that is "
    "dollar-cost averaging), using only money you can leave alone for years.",
    "To see what a bad stretch looks like first, ask \"how much could I lose on BTC in a bad "
    "week with $500\".",
)
_NOW_OR_WAIT_S: Final = (
    "Bottom line: nobody knows if the price will go up or down next.",
    "If you wait, it may never get cheaper. If you buy all at once, it may fall the next week.",
    "Many people buy a small amount each month. Use only money you can leave alone.",
)

_PAPER_LOSS: Final = re.compile(
    r"\b(?:shows?|showing|says?|saying)\s+(?:a\s+)?(?:loss|red|negative|minus|i'?m\s+down)\b|"
    r"\bi(?:'?m|\s+am)\s+(?:down|in\s+the\s+red|losing\s+money)\b[^?]{0,30}\b(?:what|should|"
    r"now)\b|\bmy\s+(?:portfolio|coins?|crypto|investments?|bitcoin|account)\b[^?]{0,20}\b(?:is|"
    r"are)\s+(?:down|red|losing)\b[^?]{0,40}\b(?:what|should|do|now)\b", re.I)
_PAPER_LOSS_A: Final = (
    "Bottom line: a loss on the screen is a loss on paper — it only becomes real if you sell at "
    "that price, so the first move is to do nothing in a hurry.",
    "Do not panic-sell because it hurts, and do not \"average down\" by buying more just to feel "
    "better, least of all with borrowed money or leverage.",
    "Ask what your reason for buying was and whether it has changed. If it holds and you do not "
    "need the money soon, waiting can be fine; if you cannot sleep, the position is too big and "
    "selling part of it is a legitimate choice.",
    "If the position uses leverage, check your liquidation price: that is the one case where "
    "the exchange can close it and make the loss real without you choosing.",
)
_PAPER_LOSS_S: Final = (
    "Bottom line: a loss on your screen is only real if you sell. Do not rush.",
    "Do not sell in a panic, and do not buy more just to feel better.",
    "Ask yourself why you bought. If that reason is still true and you do not need the money "
    "soon, you can wait. If it keeps you awake, the amount was too big.",
)

_BOUGHT_TOP: Final = re.compile(
    r"\b(?:bought|buy|got\s+in|entered|aped)\b[^?]{0,25}\b(?:at\s+the\s+top|the\s+top|"
    r"the\s+peak|the\s+ath|an?\s+all[\s-]time\s+high|the\s+high)\b|\bam\s+i\s+(?:cooked|screwed|"
    r"rekt|doomed|toast|done\s+for)\b", re.I)
_BOUGHT_TOP_A: Final = (
    "Bottom line: not necessarily cooked — buying at a high price is only a paper loss until you "
    "sell, and some prices that fell have recovered over the years (some never do, so this is "
    "not a promise).",
    "What matters now: was it money you can leave alone for a long time, and is any of it "
    "leveraged or borrowed? If you can answer \"yes\" and \"no\", you can wait without doing "
    "anything; if not, reduce the size.",
    "Avoid the two classic mistakes: selling in panic at the bottom, and buying a lot more at "
    "once to \"fix\" your average price. If you still believe in it later, small regular buys "
    "are different.",
)
_BOUGHT_TOP_S: Final = (
    "Bottom line: you are not finished. It is a loss only if you sell.",
    "Check two things: can you leave the money alone for a long time, and did you borrow any of "
    "it? If yes and no, you can wait.",
    "Do not sell in a panic, and do not buy a lot more to fix it.",
)

_TOO_LATE: Final = re.compile(
    r"\btoo\s+late\s+to\s+(?:start|begin|get\s+started|get\s+into\s+(?:investing|crypto|"
    r"trading))\b|\btoo\s+old\s+to\s+(?:start|begin|invest\w*|trade|trading)\b|"
    r"\b(?:too\s+late|late)\b[^?]{0,40}\b(?:i'?m|im|i\s+am)\s+\d{2}\b|\b(?:i'?m|im|i\s+am)\s+"
    r"\d{2}\b[^?]{0,40}\b(?:too\s+late|too\s+old)\b", re.I)
_AGE: Final = re.compile(r"\b(?:i'?m|im|i\s+am|aged?|at)\s+(?P<age>\d{2})\b", re.I)


def _too_late(text: str) -> Lines | None:
    if not _TOO_LATE.search(text):
        return None
    age = _AGE.search(text)
    years = ""
    if age is not None and 20 <= int(age.group("age")) <= 59:
        left = 65 - int(age.group("age"))
        years = f" If you stop work around 65, that is still about {left} years."
    return ["Bottom line: no — it is not too late. Many people start investing in their 40s; "
            "what matters most is how long you can leave the money alone and how much you can "
            f"put in regularly, not the age you begin.{years}",
            "A sensible order: first an emergency fund (three to six months of essential "
            "expenses) and paying off high-interest debt; then regular small amounts into broad, "
            "low-cost things; no leverage; and crypto only as a small part you could lose.",
            "Be wary of anyone who says you must \"catch up fast\": that urgency is how people "
            "take too much risk or get scammed."]


_TOO_LATE_S: Final = (
    "Bottom line: no, it is not too late. Many people start at 40 or later.",
    "What counts is how long you can leave the money alone, and how much you can add each month.",
    "First keep some cash for emergencies and pay off debts with high interest. Then add small "
    "amounts. Do not borrow to invest.",
)

_WEN_MOON: Final = re.compile(r"\b(?:should|shud)\s+i\s+(?:just\s+)?ape\b|\bwen\s+(?:moon|lambo)\b"
                              r"[^?]{0,30}\b(?:should|ape|buy|ser)\b|\bape\s+in\b[^?]{0,20}\b"
                              r"(?:should|now|ser)\b", re.I)
_WEN_MOON_A: Final = (
    "Bottom line: nobody knows when, or if — and \"aping in\" means buying fast without "
    "checking, which is how many people end up buying a top.",
    "When a coin is the talk of social media, many of the people saying it will moon already "
    "own it; by the time it is everywhere, the cheap part is usually over.",
    "If you still want some: an amount you could lose entirely, no leverage, a coin listed on a "
    "reputable exchange with real trading volume, and a decision beforehand about the loss at "
    "which you would sell. Never borrow to ape.",
)
_WEN_MOON_S: Final = (
    "Bottom line: nobody knows when a coin will shoot up, or if it will.",
    "\"Aping in\" means buying fast without checking. That is how people buy at the top.",
    "If you buy anyway, use only money you can lose, and do not borrow.",
)

_PRICE_DIFF: Final = re.compile(
    r"\b(?:why|how\s+come)\b[^?]{0,30}\bprice\b[^?]{0,40}\b(?:different|differs?|not\s+the\s+same|"
    r"doesn'?t\s+match|mismatch\w*)\b|\bprice\b[^?]{0,20}\bdiffer\w*\b[^?]{0,20}\b(?:between|on|"
    r"across)\b[^?]{0,20}\b(?:apps?|exchanges?|sites?|websites?)\b", re.I)
_PRICE_DIFF_A: Final = (
    "Bottom line: it is normal for two apps to show slightly different prices — each one shows "
    "its own market, or a delayed or averaged figure.",
    "Common reasons: each exchange has its own order book, so its last trade differs a little; "
    "one app shows the last trade, another the middle between the buy and the sell price, or an "
    "average across exchanges; data can be seconds apart; and quotes in dollars, USDT or your "
    "local currency differ by the exchange rate.",
    "A gap of a fraction of a percent on a big coin is normal. A large gap, or a price far from "
    "every other place, is a warning sign — do not buy there.",
    "The price you will actually pay is on the order screen of the exchange you trade on, "
    "including its fee and the spread.",
)
_PRICE_DIFF_S: Final = (
    "Bottom line: two apps can show different prices, and that is normal.",
    "Each one looks at its own market, or at a slightly older or averaged price.",
    "Small gaps are fine. A big gap is a warning sign. Trust the price on the screen where you "
    "buy.",
)

_FRACTION: Final = re.compile(
    r"\b(?:buy|own|get)\s+(?:a\s+|just\s+a\s+)?(?:fraction|part|piece|bit|slice|portion)\s+of\s+"
    r"(?:a\s+|an\s+|one\s+)?(?:whole\s+)?(?P<what>share|stock|coin|bitcoin|btc|eth|ethereum)\b|"
    r"\bfractional\s+(?P<what2>shares?|coins?)\b|\b(?:buy|own)\s+(?:less|under|part)\s+(?:than\s+)?"
    r"(?:a\s+|one\s+)(?:whole\s+)?(?:share|coin|bitcoin)\b", re.I)


def _fraction(text: str) -> Lines | None:
    m = _FRACTION.search(text)
    if m is None:
        return None
    what = (m.group("what") or m.group("what2") or text).lower()
    if re.search(r"coin|bitcoin|btc|eth", what):
        return ["Bottom line: yes — coins are divisible. One Bitcoin is 100,000,000 small units "
                "(satoshis), so you can buy as little as the exchange's minimum order allows; "
                "you buy by amount (say $10 of BTC), not by whole coin.",
                "Check the minimum order size on the order screen, and remember that fees weigh "
                "more on a tiny order."]
    return _first("can i buy a fraction of a share")


_FRACTION_S: Final = (
    "Bottom line: yes. You buy by amount of money, not by whole share or coin.",
    "For example, you can buy $10 of Bitcoin. There is a smallest order size, shown on the "
    "order screen.",
)

_FAST: Final = re.compile(
    r"\bis\s+it\s+normal\b[^?]{0,40}\b(?:price|prices|move|moves|moving|swing\w*|drop\w*|"
    r"jump\w*|pump\w*|volatil\w*)\b[^?]{0,30}\b(?:fast|so\s+much|this\s+much|quick\w*|crazy|"
    r"wild|like\s+this)\b|\bwhy\s+(?:is|does)\s+(?:the\s+)?price\s+(?:moving|move|jumping|"
    r"swinging)\s+so\s+(?:fast|much)\b", re.I)


def _fast(text: str) -> Lines | None:
    if not _FAST.search(text):
        return None
    out = ["Bottom line: yes — for crypto, fast moves are normal: Bitcoin often moves several "
           "percent in a day, and smaller coins move much more, far more than most shares do."]
    try:
        from argus.lui.research.rule_test import daily_closes

        stamps, closes, _ = daily_closes("BTCUSDT")
        moves = [(closes[i] / closes[i - 1] - 1, stamps[i]) for i in range(1, len(closes))][-365:]
        big = sum(1 for move, _day in moves if move <= -0.03)
        worst, day = min(moves)
        out.append(f"In the last {len(moves)} days Bitcoin closed down 3% or more on {big} days; "
                   f"its worst day was {worst:.1%} ({day:%d %b %Y}), Bitget's daily closes.")
    except Exception:
        pass
    out.append("Fast moves cut both ways: with leverage they become forced sales (liquidation), "
               "and thinly traded small coins can move 20% in minutes. If the speed makes you "
               "anxious, the amount is probably too big.")
    return out


_FAST_S: Final = (
    "Bottom line: yes. Crypto prices jump around a lot, more than most shares.",
    "Bitcoin can move several percent in one day. Small coins move even more.",
    "If it scares you, you put in too much. Use less.",
)

_BITCOIN_BUY: Final = re.compile(r"\bwhat\s+is\s+bitcoin\b[^?]{0,40}\bhow\b[^?]{0,15}\b(?:do\s+i|"
                                 r"to|can\s+i)\s+(?:buy|get)\b", re.I)


def _bitcoin_buy(text: str) -> Lines | None:
    if not _BITCOIN_BUY.search(text):
        return None
    how = _first("how do i buy bitcoin first time")
    if how is None:
        return None
    return ["Bottom line: Bitcoin is a digital currency that runs on a public network no company "
            "controls; people buy it as an investment, and its price swings a lot.",
            "To buy some, you do it on an exchange:", *how[:3]]


_WHALE: Final = re.compile(r"\bwhales?\b", re.I)
_WHALE_ASK: Final = re.compile(r"\b(?:follow\w*|copy\w*|track\w*|trust\w*|should|worth|listen)\b",
                               re.I)
_WHALE_A: Final = (
    "Bottom line: a whale is a person or fund holding so much of a coin that one trade of "
    "theirs can move the price. Should you follow them? Usually not.",
    "Their trades show only after the fact: by the time you copy a big buy, they are already in "
    "at a better price, and a whale can sell into your buying. A big wallet moving coins may "
    "also just be moving them between its own wallets or to an exchange for safekeeping.",
    "Whales can bait too: a big visible order can be placed to pull others in and then cancelled. "
    "Watching them is fine as one clue; copying them with money you cannot lose is not a plan.",
)
_WHALE_S: Final = (
    "Bottom line: a whale is someone with so many coins that their trades move the price.",
    "You usually should not copy them. You see what they did late, so they get a better price "
    "than you, and they can sell to you.",
)

_MARKET_CAP: Final = re.compile(
    r"^\W*(?:what\s+(?:does|is|do|are)|what'?s|whats|explain|define|meaning\s+of)\s+(?:the\s+|a\s+)?"
    r"(?:crypto\s+|coin\s+)?market\s*cap(?:italization|italisation)?(?:\s+(?:mean|means|meaning|"
    r"even\s+mean|actually\s+mean|for\s+(?:crypto|coins?)|in\s+crypto))?\W*$", re.I)
_MARKET_CAP_A: Final = (
    "Bottom line: market cap is the price of one coin times the number of coins in circulation. "
    "It measures how big a coin is, not whether it is cheap or expensive.",
    "Example: a coin at $0.01 with 100 billion coins out has a market cap of $1 billion; a coin "
    "at $100 with 1 million coins has $100 million. The coin with the tiny price is the bigger "
    "one.",
    "So a low price does not mean cheap: with a trillion coins out, a coin would need a $1 "
    "trillion market cap to be worth $1 each. Look at market cap and daily trading volume, and "
    "at how many coins are still to be released, not the price alone. For shares it is the same "
    "idea: share price times shares.",
)
_MARKET_CAP_S: Final = (
    "Bottom line: market cap is how much all the coins of one kind are worth together: the "
    "price of one coin times how many coins exist.",
    "A coin that costs one cent can be worth more in total than a coin that costs $100, if "
    "there are a lot more of them. A low price does not mean cheap.",
)

_STABLE: Final = re.compile(
    r"^\W*(?:what\s+(?:is|are)|what'?s|whats|explain|define)\s+(?:a\s+|an\s+)?stable\s*coins?"
    r"(?:\s+(?:and\s+)?(?:is\s+it\s+|are\s+they\s+)?(?:safe|risky|legit)(?:\s+to\s+(?:use|hold|"
    r"buy))?)?\W*$", re.I)
_STABLE_A: Final = (
    "Bottom line: a stablecoin is a crypto token meant to stay worth one US dollar (such as USDT "
    "or USDC), so you can hold and move \"dollars\" on the crypto system without the usual "
    "swings.",
    "It is not risk-free: its value rests on the reserves, or the mechanism, behind it. "
    "TerraUSD (UST) fell from $1 to about $0.10 within days in May 2022 and became nearly "
    "worthless; USDC, backed by reserves, still dipped to about $0.87 for a weekend in March "
    "2023 when a bank holding part of its reserves failed.",
    "Safer habits: keep as a stablecoin only what you need, spread it over more than one "
    "issuer, prefer large issuers that publish reserve reports, and remember it is not a bank "
    "deposit and is not insured.",
)
_STABLE_S: Final = (
    "Bottom line: a stablecoin is a crypto coin made to always be worth one dollar.",
    "It can go wrong. In 2022 one called TerraUSD fell to ten cents and was lost.",
    "Do not keep all your money in one, and do not think it is as safe as a bank.",
)
_STABLE_DOWN_A: Final = (
    "Bottom line: if a stablecoin \"goes down\", it has lost its dollar peg — your \"dollars\" are "
    "then worth less than you counted on.",
    "How bad it is depends on what stands behind it: the one that collapsed, TerraUSD, had no "
    "real reserves and went to about $0.10 in days (May 2022) and did not return; one backed by "
    "real reserves, USDC, fell to about $0.87 in March 2023 and was back at $1 four days later.",
    "What helps: hold only what you need as a stablecoin, spread it across more than one large "
    "issuer, avoid coins that promise high yields for holding them, and do not panic-sell "
    "without checking why it moved.",
)
_HOLD_DOWN_A: Final = (
    "Bottom line: if it keeps falling, that has happened before — Bitcoin lost about 65% in 2022 "
    "alone (47,216 to 16,558 USDT, Bitget's daily closes), and many smaller coins fell far "
    "more or never came back.",
    "So the question is not what happens if it falls, but whether you can sit through it: only "
    "hold money you do not need for years, and decide now, before it happens, at what loss you "
    "would sell.",
    "Do not use leverage or borrowed money (a leveraged position can be closed for you), and do "
    "not add more only to feel better about the price you paid.",
)

_WITHDRAW_STEPS: Final = (
    "Bottom line: the steps to get your money out, in order:",
    "1. On Bitget, close or sell what you want to take out (sell the coin for cash or USDT).",
    "2. Make sure the money is in your spot account (move it over from futures or funding if "
    "needed).",
    "3. Press Withdraw: to a bank (by selling for cash, where Bitget offers it for your country) "
    "or to a crypto address.",
    "4. For a crypto withdrawal, check the network and the address twice before confirming — a "
    "transfer on the wrong network is usually lost; try a small amount first.",
)
_WITHDRAW_S: Final = (
    "Bottom line: four steps to get your money out.",
    "1. Sell the coin in the app.",
    "2. Put the money in your spot account.",
    "3. Press Withdraw and pick your bank or an address.",
    "4. Check the network and the address twice, then confirm.",
)
_STAKING_S: Final = (
    "Bottom line: staking means you lock your coins for a while, and you get a reward.",
    "The risks: the price can still fall; you may not be able to sell while they are locked; "
    "and the coins sit with the exchange or the network, not only with you.",
    "If you try it, use a small amount and pick a version you can leave at any time.",
)
_WHAT_NOW_S: Final = (
    "Bottom line: I cannot pick for you, but three questions will help you choose.",
    "1. How much could you lose in a bad week without it hurting? Ask me with your own coin and "
    "amount.",
    "2. How long can you leave the money alone? Money you need within a year is safest in a "
    "bank.",
    "3. Start with one big, simple thing and do not borrow, so the most you can lose is what "
    "you put in.",
)

# --- tax by country ---------------------------------------------------------------------------

_TAX_Q: Final = re.compile(r"\btax\w*|\bhmrc\b|\bcapital\s+gains?\b|\b8949\b|\bdeclar\w*|"
                           r"\bnrs\b|\bfirs\b|\breceita\b|\bimposto\b", re.I)
_CRYPTO_WORD: Final = re.compile(r"\bcrypto\w*|\bbitcoin\b|\bbtc\b|\beth\b|\bcoins?\b|\btokens?\b|"
                                 r"\bdigital\s+assets?\b|\bnfts?\b|\bstaking\b|\bairdrops?\b",
                                 re.I)
_UK: Final = re.compile(r"\b(?:uk|u\.k\.|united\s+kingdom|britain|british|england|english|hmrc)\b",
                        re.I)
_US: Final = re.compile(r"(?i:\busa\b|\bunited\s+states\b|\bamerica\b|\birs\b|\bu\.s\.a?\b|"
                        r"\bform\s+8949\b)|\bUS\b")
_NG: Final = re.compile(r"\b(?:nigeria\w*|naira|firs|nrs|lagos)\b", re.I)
_BR: Final = re.compile(r"\b(?:brazil\w*|brasil\w*|receita\s+federal|reais|real\s+brasileiro)\b",
                        re.I)
_TAX_UK_A: Final = (
    "Bottom line: in the UK, HMRC treats crypto as an asset: selling it, swapping it for "
    "another coin, spending it, or giving it to someone other than your spouse or a charity is a "
    "\"disposal\", and a gain on it can be liable to Capital Gains Tax.",
    "You pay only on your total gains for the tax year (6 April to 5 April) above the annual "
    "exempt amount, currently £3,000 (gov.uk). If your total gain is above it you must report "
    "it to HMRC; losses can be reported to reduce gains.",
    "Keep a record of every disposal: the date, how many tokens, and their value in pounds. "
    "Selling within 30 days of buying the same token is worked out under a different rule. "
    "Crypto received as pay, or for work, is Income Tax instead.",
    "Not tax advice; HMRC's guide is \"Check if you need to pay tax when you sell "
    "cryptoassets\" on gov.uk.",
)
_TAX_US_A: Final = (
    "Bottom line: in the US, the IRS treats crypto as property, not currency, so selling it, "
    "swapping it for another coin, or spending it is a taxable event: you owe tax on the gain "
    "(what you received minus what you paid, your \"basis\").",
    "If you held it one year or less, the gain is short-term, taxed like ordinary income; if "
    "more than one year, it is long-term, usually at lower rates.",
    "Sales and swaps go on Form 8949 (the totals carry to Schedule D). Staking, mining and "
    "similar rewards are ordinary income. Your tax return asks whether you sold, exchanged or "
    "received digital assets. From 2025 transactions, brokers also send you a Form 1099-DA.",
    "Keep records of every purchase and sale. Not tax advice; irs.gov/filing/digital-assets "
    "has the rules.",
)
_TAX_NG_A: Final = (
    "Bottom line: in Nigeria, from 1 January 2026 the Nigeria Tax Act 2025 treats crypto "
    "(digital assets) as taxable: gains from selling it are taxed, and the tax authority is the "
    "Nigeria Revenue Service (NRS, formerly FIRS).",
    "How much depends on whether you sell now and then (a capital gain) or trade as a business "
    "(profits taxed as income, at rates for individuals that go up to 25%). The reports I "
    "checked disagree on the exact capital-gains rate, so confirm it with the NRS or a Nigerian "
    "tax adviser rather than trust a figure here.",
    "Exchanges and similar platforms in Nigeria must register for tax and report. Keep records "
    "of what you paid, when you sold, and the naira value of each trade. Not tax advice.",
)
_TAX_BR_A: Final = (
    "Bottom line: in Brazil, profit from selling crypto is taxed as a capital gain by the "
    "Receita Federal, but sales totalling R$35,000 or less in a month are exempt.",
    "Above R$35,000 of sales in a month, the profit on those sales is taxed, from 15% (progressive "
    "rates), payable by the last business day of the following month.",
    "Also report your crypto in your annual return (\"Bens e Direitos\", group 08) when a coin is "
    "worth R$5,000 or more at purchase cost; exchanges report to the Receita under IN RFB "
    "1,888. Rules change: check the Receita Federal. Not tax advice.",
)
_TAX_S: Final = (
    "Bottom line: in most countries you pay tax when you sell crypto for a profit, swap it for "
    "another coin, or spend it. Holding it is usually not taxed.",
    "Each country has its own rules and tax office. Keep a record of what you paid and when you "
    "sold.",
)


def _tax_country(text: str) -> Lines | None:
    if not (_TAX_Q.search(text) and _CRYPTO_WORD.search(text)):
        return None
    return _country_tax(text)


def _country_tax(text: str) -> Lines | None:
    found = [(m.start(), answer) for rx, answer in ((_UK, _TAX_UK_A), (_US, _TAX_US_A),
                                                      (_NG, _TAX_NG_A), (_BR, _TAX_BR_A))
             if (m := rx.search(text)) is not None]
    if not found:
        return None
    return list(min(found, key=lambda pair: pair[0])[1])


def _phish(text: str) -> Lines | None:
    if (_PHISH_CHANNEL.search(text) and _PHISH_BAIT.search(text) and _PHISH_BRAND.search(text)
            and _PHISH_GOT.search(text) and not re.search(r"\bairdrops?\b", text, re.I)):
        return list(_PHISH_A)
    if _FAKE_SUPPORT.search(text) and _SECRET_ASKED.search(text):
        return list(_FAKE_SUPPORT_A)
    return None


_FAKE_SUPPORT: Final = re.compile(
    r"\b(?:support|admin|agent|staff|help\s*desk|customer\s+service|moderator|official)\b", re.I)
_SECRET_ASKED: Final = re.compile(
    r"\b(?:2fa|two[\s-]?factor|otp|one[\s-]?time|verification\s+code|code|password|seed|"
    r"recovery\s+phrase|private\s+key|remote\s+access|anydesk|screen\s*share)\b", re.I)
# "someone on telegram says they are bitget support and need my 2FA code" was declined (round 43
# live re-ask). No product name for checking a contact is given: the verification page the site
# links could not be read to confirm its name (2026-10-06), so the answer points to the app and
# the official site only.
_FAKE_SUPPORT_A: Final = (
    "Bottom line: it is a scam — real Bitget support never asks for your 2FA code, password, "
    "seed phrase or remote access to your phone, by Telegram, WhatsApp or anywhere else. Do not "
    "send the code.",
    "Anyone holding a fresh 2FA code can log in or approve a withdrawal as you. If you already "
    "sent one, change your password and reset 2FA in the official app now, and check withdrawals "
    "and API keys.",
    "Reach support only through the official app's own help or live chat, or by typing the "
    "exchange's address yourself — never through someone who messaged you first.",
)


_ROUND43: Final[tuple[Callable[[str], Lines | None], ...]] = (
    _fixed(_P2P, _P2P_A, also=_P2P_ASK),
    _fixed(_SIGNAL_GROUP, _SIGNAL_A, also=_SIGNAL_ASK, never=re.compile(r"\brecover\w*", re.I)),
    _phish,
    _fixed(_FORGOT_2FA, _FORGOT_2FA_A),
    _scam_coin,
    _fixed(_RUG, _RUG_A),
    _atm,
    _fixed(_SIM_SWAP, _SIM_SWAP_A),
    _fixed(_REPORT_SCAM, _REPORT_SCAM_A),
    _exchange_trust,
    _fixed(_PUMP, _PUMP_A),
    _fixed(_CHART, _CHART_A),
    _small_sum,
    _fixed(_DCA, _DCA_A),
    _fixed(_EXCH_WALLET, _EXCH_WALLET_A),
    _fixed(_BULL_TRAP, _BULL_TRAP_A),
    _fixed(_BUBBLE, _BUBBLE_A),
    _fixed(_INFLATION, _INFLATION_A, never=_INFLATION_NOT),
    _fixed(_WD_LIMIT, _WD_LIMIT_A),
    _fixed(_NOW_OR_WAIT, _NOW_OR_WAIT_A),
    _fixed(_PAPER_LOSS, _PAPER_LOSS_A),
    _fixed(_BOUGHT_TOP, _BOUGHT_TOP_A),
    _too_late,
    _fixed(_WEN_MOON, _WEN_MOON_A),
    _fixed(_PRICE_DIFF, _PRICE_DIFF_A),
    _fraction,
    _fast,
    _bitcoin_buy,
    _fixed(_WHALE, _WHALE_A, also=_WHALE_ASK),
    _fixed(_MARKET_CAP, _MARKET_CAP_A),
    _fixed(_STABLE, _STABLE_A),
    _tax_country,
)
"""The round-43 readers, in the order tried. Each is narrow: a question that is not exactly
one of these falls through to the rest of the console."""


# --- round 44: emergencies, scams and the basics a first-time investor asks ----------------------
#
# Round 44's newcomer audit (Activity/audits/round44_newcomer.md) found one critical miss (a user
# who had just sent ETH to the wrong network was told to wait out a "paper loss") and fourteen
# major ones. Facts below, each read from its source on 2026-10-06:
#
# - Wrong network or address: Bitget Help Center, "How to Deal with Wrong Coin or Wrong
#   Blockchain Deposits?" (bitget.com/support/articles/12560603820594): stop further transfers;
#   prepare the transaction hash (TxID), the deposit address, the coin and the amount; submit the
#   deposit-recovery form; Bitget checks whether recovery is feasible, tells the user about fees
#   "if applicable", and complex cases may take 10 business days or longer; "not all networks
#   support refunds or fund recovery". US Federal Trade Commission, "What To Know About
#   Cryptocurrency and Scams" (consumer.ftc.gov): if you send crypto to the wrong person "no one
#   can step in to help you recover your funds". One address works on every EVM network (Ethereum,
#   BNB Chain, Polygon, Arbitrum, Base) because they share the address format.
# - Recovery services: FTC consumer alert, "Worried about crypto exchange losses? Don't pay money
#   for help recovering money" (Nov 2022): "Don't pay anyone who contacts you, offering to
#   recover money you lost to a scam"; recovery scammers buy lists of earlier victims and ask a
#   "retainer" or "processing fee"; report at ReportFraud.ftc.gov.
# - Fake apps and sites: FTC, "What To Know About Cryptocurrency and Scams": fake investment
#   sites let you deposit and then block withdrawals or charge high fees; never click links in an
#   unexpected message; search the name with "review", "scam" or "complaint".
# - One exchange or several: FTX halted withdrawals and filed for bankruptcy in November 2022.
#   Bitget publishes proof-of-reserves reports (stated earlier in this file).
# - Rug pull versus dump: a rug pull is the insiders removing the liquidity or abandoning a
#   project after taking the money; a dump is heavy selling that pushes a price down, a pump and
#   dump being the planned version (Sumsub, "Pump-and-Dump vs Rug Pull"; Britannica Money, "Pump-
#   and-Dump Schemes & Crypto Rug Pulls Explained").
# - HODL: from a Bitcointalk post titled "I AM HODLING" (18 Dec 2013), a misspelling of "hold".
#   Bitcoin's 2022 fall (47,216 to 16,558 USDT, Bitget's daily closes) is the figure used above.
# - Index fund and stock split: SEC Investor.gov glossary definitions (an index fund tracks a
#   market index; a split changes the number of shares and the price per share, not the value
#   held). The S&P 500 price index fell about 19% in 2022 (S&P Dow Jones Indices: -19.44%).
# - Benefits: no rule is stated; the answer names what to check, because benefit rules differ by
#   country and by programme.
# - Fees: Bitget spot standard 0.10% a side (stated earlier in this file).

_NO_TICKER: Final = re.compile(r"\b[A-Z]{2,5}\b")


_WRONG_BAD: Final = (r"(?:(?:wr[oi]ng|incorrect|mistaken)\s+(?:network|chain|blockchain|address|"
                     r"addy|wallet|coin|token|memo|tag)|(?:different|another|other|diff)\s+"
                     r"(?:network|chain|blockchain))")
_WRONG_SEND: Final = re.compile(
    r"\b(?:sent|send|sending|transferr?ed|transfer|withdr[ae]w|withdrawn|deposit(?:ed)?|paid|"
    r"moved|bridged|put)\b[^?]{0,70}\b(?:to\s+)?(?:the\s+|a\s+)?" + _WRONG_BAD + r"\b|"
    r"\b" + _WRONG_BAD + r"\b[^?]{0,40}\b(?:sent|send|transfer\w*|withdr\w*|deposit\w*)\b|"
    r"\bnetwork\s+mismatch\b|\bmismatched\s+network\b|\b(?:erc-?20|trc-?20|bep-?20|bsc|polygon|"
    r"solana|tron)\b[^?]{0,40}\b(?:instead\s+of|rather\s+than)\b[^?]{0,20}\b(?:erc-?20|trc-?20|"
    r"bep-?20|bsc|polygon|solana|tron|ethereum)\b", re.I)
_WRONG_NOT: Final = re.compile(r"\b(?:avoid|prevent|make\s+sure|how\s+to\s+not)\b", re.I)
_WRONG_SEND_A: Final = (
    "Bottom line: stop and do not send anything else yet — whether you can get it back depends on "
    "where it went, and sometimes a blockchain transfer cannot be reversed.",
    "1. Open the transaction by its hash (the long ID) on the block explorer of the network you "
    "used and read the address it went to. Is that your own wallet? Then the coins are often "
    "still there: on Ethereum-style networks one address works on all of them, so add that "
    "network to the wallet, or open the same recovery phrase in an official wallet app that "
    "supports it (never on a website).",
    "2. Did it go to an exchange's deposit address? Contact that exchange's support with the "
    "hash, the address, the coin and the amount. Bitget, for example, has a deposit-recovery "
    "form, may charge a fee, and says not every network can be recovered.",
    "3. A smart-contract address or a stranger's wallet is usually not recoverable: nobody can "
    "reverse a blockchain transfer.",
    "Beware: anyone who messages you offering to \"recover\" it for a fee is a scammer (US FTC). "
    "Real help is the receiving platform's own support, reached through its official app.",
)
_WRONG_SEND_S: Final = (
    "Bottom line: stop sending. Whether you get it back depends on where it went.",
    "Look up the transaction hash. Your own wallet: the coins are often still there. An exchange: "
    "contact its support with the hash. A stranger or a contract: usually gone.",
    "Anyone offering to recover it for a fee is a scammer.",
)
_WRONG_BACK: Final = re.compile(
    r"\b(?:get|got|have|bring|win|take)\s+(?:it|them|that|this|my\s+\w+|the\s+\w+)\s+back\b|"
    r"\b(?:recover\w*|retriev\w*|revers\w*|undo|refund\w*|any\s+hope|is\s+it\s+gone|gone\s+"
    r"forever|lost\s+forever)\b", re.I)
_WRONG_BACK_A: Final = (
    "Bottom line: sometimes — it depends on who controls the address it went to, and the sooner "
    "you act the better.",
    "Your own wallet on another network: usually yes, you still own it. An exchange deposit "
    "address: maybe, if that exchange supports the network and agrees to recover it, which can "
    "take days and cost a fee. A contract or a stranger's wallet: almost never.",
    "Next: write down the transaction hash and the address, then contact the receiving platform's "
    "support through its official app. Do not send the coins again, and do not pay anyone who "
    "contacts you offering to get them back.",
)


def _wrong_send(text: str) -> Lines | None:
    if not _WRONG_SEND.search(text) or _WRONG_NOT.search(text):
        return None
    return list(_WRONG_SEND_A)


_RECOVERY: Final = re.compile(
    r"\brecovery\s+(?:service|company|agency|expert|specialist|firm|agent|team|hacker)s?\b|"
    r"\b(?:recover|retrieve|get\s+back|trace|return)\w*\b[^?]{0,50}\b(?:for\s+a\s+(?:small\s+)?fee|"
    r"upfront|up\s+front|if\s+i\s+pay|charge[sd]?|deposit\s+first|percent|%)|"
    r"\b(?:says?|claims?|offers?|offered|contacted|messaged|dm'?d)\b[^?]{0,60}\b(?:recover|get\s+"
    r"(?:my|your|the)\b[^?]{0,20}\bback|retrieve)\b", re.I)
_RECOVERY_A: Final = (
    "Bottom line: it is almost certainly a scam — do not pay, and do not give them your recovery "
    "phrase, passwords or remote access to your device.",
    "Scammers buy lists of people who were already scammed and then charge a \"retainer\", a "
    "\"processing fee\" or a \"tax\" to get the money back; the US FTC says no legitimate company "
    "will contact you and offer to recover your money for a fee.",
    "Signs: they contacted you first, ask for money or crypto up front, promise a percentage back "
    "or a guaranteed result, and push you to hurry or keep it secret.",
    "What to do: report the original theft to the exchange you used and to the police (US: "
    "reportfraud.ftc.gov and ic3.gov), keep every record, and ignore unsolicited help.",
)


def _recovery_scam(text: str) -> Lines | None:
    if not _RECOVERY.search(text):
        return None
    return list(_RECOVERY_A)


_FAKE_APP: Final = re.compile(
    r"\b(?:fake|scam|counterfeit|clone[sd]?|malicious|phishing|spoof\w*)\s+(?:crypto(?:currency)?"
    r"\s+|wallet\s+|exchange\s+|trading\s+|investment\s+)*(?:apps?|applications?)\b|\b(?:is|are)"
    r"\s+(?:this|that|the)\s+(?:crypto(?:currency)?\s+|trading\s+|wallet\s+|exchange\s+)*app\s+"
    r"(?:real|legit|fake|safe|a\s+scam|genuine|official)\b", re.I)
_FAKE_APP_ASK: Final = re.compile(r"\b(?:spot|tell|know|identify|avoid|detect|check|recogni[sz]e|"
                                  r"verify|sure|real|legit|genuine|how|is)\b", re.I)
_FAKE_APP_DONE: Final = re.compile(r"\b(?:downloaded|installed|already|lost|drained|used)\b", re.I)
_FAKE_APP_A: Final = (
    "Bottom line: check where you got the app and who published it before you open it — most "
    "fakes copy a real app's name and logo.",
    "1. Get it only from a link on the company's own website, typed by you, or from the official "
    "store listing that site points to — never from a link in a message or an ad, or a file "
    "sent to you (an APK, or a \"test version\" invite).",
    "2. In the store, check the publisher name matches the company, that the app has a long "
    "history with many reviews and downloads, and read the one-star reviews: \"could not "
    "withdraw\" is the warning.",
    "3. Test small: deposit a little and withdraw it before more. A fake shows balances and gains "
    "but finds reasons you cannot take money out (US FTC).",
    "4. A real app never asks for your recovery phrase and never guarantees a return.",
)
_FAKE_APP_DONE_A: Final = (
    "If you already installed it: delete the app, change the passwords you used, and contact the "
    "real exchange from its official app. If a wallet app held your coins, move what is left to "
    "a new wallet with a new recovery phrase, made on a clean device.",
)


def _fake_app(text: str) -> Lines | None:
    if not (_FAKE_APP.search(text) and _FAKE_APP_ASK.search(text)):
        return None
    if _FAKE_APP_DONE.search(text):
        return [*_FAKE_APP_A[:3], *_FAKE_APP_DONE_A, _FAKE_APP_A[4]]
    return list(_FAKE_APP_A)


_CUSTODY: Final = re.compile(
    r"\b(?:keep|store|hold|leave|put|spread|split|have)\w*\b[^?]{0,40}\b(?:crypto\w*|coins?|"
    r"bitcoin|funds|money|assets?|all)\b[^?]{0,50}\b(?:(?:on\s+)?(?:one|a\s+single|single|just\s+"
    r"one|1)\s+(?:exchange|platform)|(?:several|multiple|many|more\s+than\s+one|different|two|"
    r"few)\s+(?:exchanges?|platforms?))\b|\bone\s+exchange\b[^?]{0,30}\b(?:or|vs\.?|versus)\b"
    r"[^?]{0,20}\b(?:several|multiple|many|more|two|spread)\b", re.I)
_CUSTODY_A: Final = (
    "Bottom line: neither is automatically safer — one exchange is a single point of failure, but "
    "several accounts mean more passwords to protect and more places to be phished.",
    "What spreading does: if one exchange fails, freezes withdrawals or is hacked, not everything "
    "is stuck — customers of FTX could not withdraw when it collapsed in November 2022. What it "
    "costs: more logins, more 2FA, more places to withdraw from.",
    "A middle path many people use: keep on an exchange only what you trade or may sell soon, on "
    "a large, regulated one that publishes proof-of-reserves (Bitget does), and move long-term "
    "holdings to a wallet you control once you understand the recovery phrase.",
    "Either way: an authenticator app for 2FA, a unique password, a withdrawal whitelist, and "
    "only an amount you could lose.",
)

_BENEFIT_WORD: Final = (r"(?:benefits?|welfare|universal\s+credit|ssi|ssdi|snap|medicaid|food\s+"
                        r"stamps?|unemployment|dole|pension\s+credit|housing\s+benefit|"
                        r"centrelink)")
_BENEFITS: Final = re.compile(
    r"\b(?:claim\w*|receiv\w*|collect\w*|on|my|get\w*|lose|affect\w*|apply\w*\s+for)\s+(?:\w+\s+)"
    r"{0,2}" + _BENEFIT_WORD + r"\b|\b(?:universal\s+credit|ssi|ssdi|medicaid|food\s+stamps?|"
    r"centrelink|dole)\b", re.I)
_BENEFITS_NOT: Final = re.compile(r"\bbenefits?\s+of\b|\bbenefits?\s+and\s+(?:risks?|downsides?)",
                                  re.I)
_BENEFITS_A: Final = (
    "Bottom line: it depends on your country's benefit rules, and they differ by benefit — so do "
    "not guess; check before you trade.",
    "What usually matters: many means-tested benefits look at both your income and your savings "
    "or assets, and a sale, a gain, or simply holding crypto can fall under either; some "
    "programmes set a savings limit above which payments are reduced or stopped.",
    "What to check: the official guidance for your own benefit (search its name with \"capital\", "
    "\"savings\" or \"income\"), and ask the benefit office directly, in writing if you can, "
    "whether crypto holdings and gains must be reported, and when.",
    "Tax is a separate question from benefits: a gain can be taxable and also affect a benefit. "
    "Not legal or tax advice.",
)


def _benefits(text: str) -> Lines | None:
    if not (_BENEFITS.search(text) and _CRYPTO_WORD.search(text)) or _BENEFITS_NOT.search(text):
        return None
    return list(_BENEFITS_A)


_HARDWARE: Final = re.compile(r"\bhardware\s+wallets?\b|\bcold\s+(?:wallet|storage)\b|\b(?:ledger"
                              r"\s+(?:nano|wallet|device|stax|flex)|trezor)\b", re.I)
_HARDWARE_A: Final = (
    "Bottom line: a hardware wallet is a small physical device that keeps your crypto's private "
    "keys offline and signs transactions on the device itself, so malware on your phone or "
    "computer cannot take the keys.",
    "It suits people holding coins for a long time, or in amounts they would hate to lose; for "
    "a small amount you trade often, an exchange account with strong security is simpler.",
    "It does not remove every risk: you still must keep the recovery phrase it gives you (anyone "
    "with it can empty the wallet, and if you lose it and the device breaks, the coins are "
    "gone), buy it only from the maker or an authorised reseller — never used or pre-opened — "
    "and check the address on its own screen before you confirm.",
)

_STOCK_SPLIT: Final = re.compile(
    r"\b(?:stock|share)s?\s+splits?\b|\bsplit\s+(?:of\s+)?(?:a\s+|the\s+)?(?:stock|shares?)\b|"
    r"\b\d+[\s-]*(?:for|-for-)[\s-]*\d+\s+(?:stock\s+)?split\b|\breverse\s+(?:stock\s+)?split\b",
    re.I)
_STOCK_SPLIT_ASK: Final = re.compile(r"\b(?:what|mean\w*|explain|how|why|happen\w*|should|does|"
                                     r"affect|change\w*|good|bad)\b", re.I)
_STOCK_SPLIT_A: Final = (
    "Bottom line: a stock split divides each share into several, with the price cut by the same "
    "factor, so the value of what you own does not change — in a 2-for-1 split, 10 shares at "
    "$100 ($1,000) become 20 shares at $50 ($1,000).",
    "The company is the same size; only the number of pieces changed, like cutting a pizza into "
    "more slices. Companies usually split to make one share cheaper to buy; the split itself "
    "adds no value, so any later move comes from the news and the buyers.",
    "A reverse split does the opposite (say 1-for-10): fewer shares at a higher price, again "
    "with the same total value. It is often used by companies whose price has fallen very low, "
    "so read why it was done.",
)

_INDEX_FUND: Final = re.compile(r"\bindex\s+funds?\b", re.I)
_INDEX_FUND_NOT: Final = re.compile(r"\b(?:vs\.?|versus|compared?|better\s+than|or)\b[^?]{0,30}"
                                    r"\b(?:bitcoin|btc|crypto\w*|eth)\b|\b(?:bitcoin|btc|crypto"
                                    r"\w*)\b[^?]{0,30}\b(?:vs\.?|versus|or|better\s+than)\b",
                                    re.I)
_INDEX_FUND_A: Final = (
    "Bottom line: an index fund is a fund that holds all, or a large sample, of the companies in "
    "a market index such as the S&P 500, so one purchase gives you a small slice of hundreds of "
    "companies instead of a bet on one.",
    "It simply follows the index with no manager picking winners, which is why its yearly fee "
    "is usually low; compare the fee (the expense ratio) and what the index holds. Both ETFs "
    "and mutual funds can be index funds, and they are bought through a regular broker.",
    "It is diversified, not safe: if the whole market falls the fund falls with it (the S&P 500 "
    "index lost about 19% in 2022). Only money you can leave alone for years belongs in it.",
)

_BUY_SHARES: Final = re.compile(
    r"\b(?:buy|buying|purchase|purchasing|get|getting)\s+(?:some\s+|my\s+first\s+|a\s+few\s+)?"
    r"(?:shares?|stocks?)\b|\bfirst[\s-]time\b[^?]{0,30}\b(?:shares?|stocks?)\b|\b(?:invest\w*\s+"
    r"in\s+)(?:shares?|stocks?)\b", re.I)
_BUY_SHARES_ASK: Final = re.compile(r"\bfirst\b|\bstart\w*|\bbegin\w*|\bwhere\b|\bnew\s+to\b|"
                                    r"\bbeginner\b|\bhow\s+(?:do|can|to|should)\b", re.I)
_BUY_SHARES_NOT: Final = re.compile(
    r"\bcrypto\w*|\bbitcoin\b|\bcoins?\b|\btokeni[sz]ed\b|\brtokens?\b|\bipo\b|\b(?:shares?|"
    r"stocks?)\s+(?:of|in)\s+[A-Z]{2,5}\b", re.I)
_BUY_SHARES_A: Final = (
    "Bottom line: a share is a small ownership stake in a company, and you buy one through a "
    "broker — a firm licensed in your country to place stock orders for you — not through a "
    "crypto exchange.",
    "First steps: pick a broker that your country's financial regulator lists as registered (look "
    "the name up on the regulator's own site), open and verify an account, put in money you will "
    "not need soon, then place an order; a limit order lets you set the most you will pay.",
    "Many brokers sell fractions of a share, so you can start small; check the fees and any "
    "minimum. A fund that holds many companies (an index fund) spreads the risk more than one "
    "share does.",
    "On Bitget, some US stocks are available as tokenised stocks (rTokens): they track the share's "
    "price, but you do not own the share and have no vote, and they trade around the clock. If "
    "owning the actual share matters, use a broker.",
)


def _buy_shares(text: str) -> Lines | None:
    if not (_BUY_SHARES.search(text) and _BUY_SHARES_ASK.search(text)):
        return None
    if _BUY_SHARES_NOT.search(text) and not re.search(r"\bbitget\b", text, re.I):
        return None
    if _NO_TICKER.search(text) and not re.search(r"\b(?:I|A|US|UK|ETF)\b", text):
        return None
    return list(_BUY_SHARES_A)


_BALANCE_FX: Final = re.compile(
    r"\b(?:balance|portfolio|holdings?|total|account\s+value|value)\b[^?]{0,40}\b(?:differ\w*|"
    r"different|changes?|not\s+the\s+same|doesn'?t\s+match|mismatch\w*|two\s+numbers|bigger|"
    r"smaller|higher|lower)\b[^?]{0,40}\b(?:usd|eur|gbp|inr|dollars?|euros?|pounds?|currenc\w*)\b|"
    r"\b(?:usd|dollars?)\b[^?]{0,20}\b(?:and|vs\.?|versus|or)\b[^?]{0,10}\b(?:eur|euros?|gbp|inr)"
    r"\b[^?]{0,40}\b(?:different|differ\w*|not\s+the\s+same)\b", re.I)
_BALANCE_FX_A: Final = (
    "Bottom line: the amount of coin you hold has not changed — only its estimated value in each "
    "currency, which is the coin's price converted at the current exchange rate between the two "
    "currencies.",
    "For example, if one dollar is worth 0.90 euro at the moment, a balance of 1,000 USD shows as "
    "900 EUR: a different number for the same holding.",
    "The rate moves through the day, and apps use different rate sources and update times, so "
    "two screens can differ a little even in the same currency. This console cannot see your "
    "account; the rate your app uses is shown in its settings or details.",
)

_ALL_IN: Final = re.compile(
    r"\b(?:put|invest|throw|pour|go|move|stick|dump|bet|buy)\w*\s+(?:all|everything|my\s+(?:whole|"
    r"entire)|all\s+of|the\s+whole)\b[^?]{0,40}\b(?:in|into|on|to)\s+(?:the\s+|some\s+)?(?:btc|"
    r"bitcoin|eth|ethereum|sol|solana|crypto\w*|coins?|doge\w*|xrp|alts?|memecoins?|[a-z]{2,6}"
    r"coin)\b|\ball[\s-]in\b[^?]{0,20}\b(?:btc|bitcoin|eth|crypto\w*|coins?)\b", re.I)
_ALL_IN_A: Final = (
    "Bottom line: no, not everything — put in only an amount you could lose without it changing "
    "your life, because Bitcoin can fall by more than half and has done so more than once (about "
    "65% in 2022: 47,216 to 16,558 USDT, Bitget's daily closes).",
    "FOMO (fear of missing out) is the feeling that makes people buy at the worst time: when "
    "everyone is excited, much of the rise has often already happened. A rush to put it all in "
    "is the pattern to distrust.",
    "If you still want in: a small amount you could lose, then more later in equal steps if you "
    "still want to; an emergency fund in cash first; no leverage and no borrowed money.",
    "Wait an hour, or a day, before you click: a good reason to buy will still be there "
    "tomorrow.",
)

_SLIPPAGE: Final = re.compile(
    r"\bwhat\s+(?:is|does|do|are)\b[^?]{0,40}\bslip(?:page|ped|s|ping)?\b|\b(?:my|the)\b[^?]{0,30}"
    r"\bslipped\b|\bwhy\b[^?]{0,30}\bslipp(?:ed|age)\b", re.I)
_SLIPPAGE_NOT: Final = re.compile(r"\$\s?\d|\b\d+\s*[km]\b|\bdepth\b|\bwould\b|\bback-?test\w*|"
                                  r"\bbook\b", re.I)
_SLIPPAGE_A: Final = (
    "Bottom line: slippage is the gap between the price you expected when you pressed buy or "
    "sell and the price you actually got — \"my order slipped\" means it filled at a worse price "
    "than the one on screen.",
    "It happens with market orders because the price moves between your click and the fill, or "
    "because your order is bigger than what is offered at the best price and takes the next, "
    "worse prices in the order book. It is larger in thin, fast markets and for big orders.",
    "To limit it: use a limit order (you set the worst price you accept, so it fills there or "
    "better, or not at all), trade smaller sizes, avoid thin small coins and the minutes around "
    "big news, and read the estimated price on the order screen before confirming.",
)

_RUG_DUMP: Final = re.compile(
    r"\brug\w*\b[^?]{0,40}\b(?:vs\.?|versus|or|and|from|different\w*|difference)\b[^?]{0,20}\b"
    r"dump\w*|\bdump\w*\b[^?]{0,40}\b(?:vs\.?|versus|or|and|from|different\w*|difference)\b"
    r"[^?]{0,20}\brug\w*", re.I)
_RUG_DUMP_A: Final = (
    "Bottom line: a rug pull is theft by the people who run a project — they drain the money "
    "from the trading pool or abandon the project after taking it — while a dump is just a sharp "
    "fall in price caused by heavy selling, which is not always a crime.",
    "Rug pull: insiders build a token that looks real, draw in money, then remove the liquidity "
    "or vanish; the price goes to near zero and you often cannot sell at all.",
    "Dump: a large holder, or many holders at once, sell fast and push the price down; the coin "
    "can still be traded and may bounce. It becomes a scam when it is planned (a pump and dump: "
    "hype the price up, then sell to the late buyers).",
    "Both look alike on a chart. Check who holds most of the supply, whether liquidity is locked "
    "and whether the team is known, and never put in more than you could lose.",
)

_DIVIDEND: Final = re.compile(r"\bwhat\s+(?:is|are|does)\s+(?:a\s+|an\s+)?dividends?\b|\bdividends?"
                              r"\b[^?]{0,40}\b(?:every\s+month|monthly|how\s+often|get\s+paid|"
                              r"paid\s+every)\b", re.I)
_DIVIDEND_PAY: Final = re.compile(r"\bpaid\b|\bpay\w*\b|\bevery\s+month\b|\bmonthly\b|"
                                  r"\bhow\s+often\b", re.I)
_DIVIDEND_A: Final = (
    "Bottom line: a dividend is a share of a company's profit paid to its shareholders, usually "
    "in cash — and not necessarily every month: most US companies pay four times a year, many "
    "European ones once or twice, and a few pay monthly.",
    "You receive it only if you own the share before the ex-dividend date (the first day a "
    "buyer no longer gets the next payment); the share price usually drops by about the dividend "
    "that day, so it is not free money.",
    "Dividends are not guaranteed: a company can cut or stop them, and many growth companies pay "
    "none. Bitcoin pays no dividend; staking rewards are a different thing, and a tokenised "
    "stock is not the share itself, so check how it treats dividends before assuming you get one.",
)

_LIQ_HAPPENS: Final = re.compile(r"\bwhat\s+(?:happens?|will\s+happen|would\s+happen)\b[^?]{0,30}"
                                 r"\bliquidat\w*\b", re.I)
_LIQ_HAPPENS_A: Final = (
    "Bottom line: if you are liquidated, the exchange closes your leveraged position by force "
    "because its losses have used up the margin behind it — you lose that margin and the "
    "position is gone (with cross margin, the whole futures balance can be drawn on first).",
    "It is automatic: when the price reaches your liquidation price the close happens, a fee may "
    "be charged, and you cannot get that position back, only open a new one.",
    "Before it happens: your liquidation price is shown on the position screen, and the higher "
    "the leverage the closer it is (at 10x, a move of a little under 10% against you). Lower "
    "leverage, a smaller size and a stop-loss placed before the liquidation price reduce the "
    "risk; buying on spot cannot be liquidated.",
)

_HODL: Final = re.compile(r"\bhodl\w*\b", re.I)
_HODL_ASK: Final = re.compile(r"\b(?:is|real|thing|works?|strategy|mean\w*|copium|cope|good|"
                              r"worth|actually|even|legit|what)\b", re.I)
_HODL_NOT: Final = re.compile(r"\b(?:fomo|fud|rekt|ngmi|wagmi|dyor|degen|ath|diamond)\b", re.I)
_HODL_A: Final = (
    "Bottom line: HODL began as a misspelling of \"hold\" in a 2013 Bitcoin forum post and now "
    "means buying and holding through the swings instead of selling — a real strategy, but it "
    "only works if what you hold comes back, and \"copium\" (coping with losses) is what it "
    "becomes when it does not.",
    "The evidence cuts both ways: Bitcoin fell about 65% in 2022 (47,216 to 16,558 USDT, "
    "Bitget's daily closes) and later set new highs, but many smaller coins that fell never came "
    "back, so holding one coin is not the same as holding the market.",
    "If you hold: only money you can leave alone for years, no leverage (a liquidation forces a "
    "sale), and a decision beforehand about when you would sell — or buy small regular amounts "
    "instead of one big bet.",
)

_CAUTION_ASK: Final = re.compile(
    r"\b(?:watch\s+(?:out\s+)?for|look\s+out\s+for|careful|beware|mistakes?|pitfalls?|know\s+"
    r"before|should\s+i\s+know|things\s+to\s+(?:know|watch|check)|what\s+to\s+(?:watch|know|"
    r"check|avoid))\b", re.I)
_CAUTION_FIRST: Final = re.compile(r"\bfirst[\s-]time\b|\bfor\s+the\s+first\b|\bbeginner\b|"
                                   r"\bnew\s+to\b|\bjust\s+start\w*\b|\bnever\s+(?:bought|"
                                   r"invested)\b", re.I)
_CAUTION_BUY: Final = re.compile(r"\b(?:buy\w*|purchas\w*|invest\w*)\b[^?]{0,40}\b(?:bitcoin|btc|"
                                 r"crypto\w*|ethereum|eth)\b|\b(?:bitcoin|btc|crypto\w*)\b[^?]{0,"
                                 r"40}\b(?:buy\w*|purchas\w*|invest\w*)\b", re.I)
_CAUTION_A: Final = (
    "Bottom line: for a first purchase, start small — an amount you could lose without it "
    "hurting — and treat the first weeks as learning, not earning.",
    "Watch for: price swings (Bitcoin has fallen more than half more than once, about 65% in "
    "2022), fees on the buy and on any withdrawal, and where you buy: a large regulated "
    "exchange you reached yourself, not a link from a message.",
    "Protect the account: a unique password, two-factor authentication with an authenticator app, "
    "and never share a code or recovery phrase; no real support team asks for them.",
    "Avoid: leverage and borrowed money, anyone promising guaranteed returns, and buying in a "
    "rush because the price is rising. Bitget's standard spot fee is 0.10% a side, about $0.10 "
    "on $100.",
)


def _first_caution(text: str) -> Lines | None:
    if _CAUTION_ASK.search(text) and _CAUTION_FIRST.search(text) and _CAUTION_BUY.search(text):
        return list(_CAUTION_A)
    return None


_ROUND44: Final[tuple[Callable[[str], Lines | None], ...]] = (
    _recovery_scam, _wrong_send, _fake_app,
    _fixed(_HARDWARE, _HARDWARE_A),
    _fixed(_STOCK_SPLIT, _STOCK_SPLIT_A, also=_STOCK_SPLIT_ASK, never=_NO_TICKER),
    _fixed(_INDEX_FUND, _INDEX_FUND_A, never=_INDEX_FUND_NOT),
    _buy_shares, _benefits,
    _fixed(_CUSTODY, _CUSTODY_A),
    _fixed(_BALANCE_FX, _BALANCE_FX_A),
    _fixed(_ALL_IN, _ALL_IN_A),
    _fixed(_SLIPPAGE, _SLIPPAGE_A, never=_SLIPPAGE_NOT),
    _fixed(_RUG_DUMP, _RUG_DUMP_A),
    _fixed(_DIVIDEND, _DIVIDEND_A, also=_DIVIDEND_PAY, never=_NO_TICKER),
    _fixed(_LIQ_HAPPENS, _LIQ_HAPPENS_A),
    _fixed(_HODL, _HODL_A, also=_HODL_ASK, never=_HODL_NOT),
    _first_caution,
)
"""The round-44 readers, tried before the round-43 ones: an emergency or a scam question must
never be read as a price dip or as chasing a loss."""


_FIRSTS: Final[tuple[Callable[[str], Lines | None], ...]] = (
    _seed, *_ROUND44, *_ROUND43, _wallet, _loss_felt, _win_back, _scam_double, _friend_leverage,
    _emergency, _gas, _tokenised,
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


# --- round 43 follow-ups ------------------------------------------------------------------------

_SIMPLER: Final = re.compile(
    r"\b(?:explain|say|put|tell)\b[^?]{0,30}\b(?:simpler|simply|plainer|plain|easier|shorter|"
    r"like\s+i'?m\s+(?:5|five|a\s+kid))\b|^\W*(?:simpler|simplify|eli5)\W*$|\bin\s+simple\s+"
    r"(?:words|terms|english)\b|\bdumb\s+it\s+down\b|\beli5\b|"
    # round 44: "say it shorter", "shorter please", "make it shorter", "tl;dr", "too long"
    r"^\W*(?:(?:ok(?:ay)?|so|can\s+you|please|pls)\W+){0,2}(?:make\s+it\s+|say\s+it\s+|be\s+)?"
    r"(?:shorter|short|briefer|brief)(?:\s+(?:please|pls))?\W*$|\btl;?\s*dr\b|"
    r"\b(?:that'?s|that\s+is|too)\s+(?:too\s+)?long\b", re.I)
_WITHDRAWING: Final = re.compile(r"\bwithdraw\w*|\bcash\s*(?:it\s+)?out\b|\bget\s+(?:my\s+)?"
                                 r"(?:money|funds|cash)\s+(?:back\s+)?out\b|\btake\s+(?:my\s+)?"
                                 r"(?:money|funds)\s+out\b", re.I)
_STAKING: Final = re.compile(r"\bstak\w*", re.I)
_WHAT_NEXT: Final = re.compile(
    r"^\W*(?:(?:so|ok(?:ay)?|then|alright|hmm+|thanks?|thank\s+you|thx)\W+){0,3}(?:what\s+"
    r"(?:should|do|shall)\s+i\s+do(?:\s+(?:now|next))?|what\s+now|what\s+next|now\s+what|then\s+"
    r"what)\W*$", re.I)
_WHAT_NEXT_ANY: Final = re.compile(
    r"\bwhat\s+(?:should|do|shall)\s+i\s+do\b|\bwhat\s+(?:now|next)\b|"
    r"\bnow\s+what\b", re.I)
_STABLE_ANY: Final = re.compile(r"\bstable\s*coins?\b|\busdt\b|\busdc\b|\btether\b", re.I)
_IF_DOWN: Final = re.compile(
    r"^\W*(?:(?:and|but|so|ok(?:ay)?)\W+)*(?:what\s+)?(?:if|when)\s+(?:it|they|the\s+price|that)\s+"
    r"(?:goes?|go|falls?|drops?|crash\w*|tanks?|dips?)(?:\s+down)?\W*$|^\W*(?:and\s+)?what\s+if\s+"
    r"(?:it|the\s+price)\s+(?:goes?|keeps?)\s+(?:down|falling|dropping)\W*$", re.I)
_LOSS_THREAD: Final = re.compile(r"\bloss\w*|\bhold\w*|\bbought\b|\bdown\b|\bred\b|\bfell\b|"
                                 r"\bdrop\w*|\bnot\s+selling\b", re.I)


def _follow_simpler(text: str, before: str) -> Lines | None:
    """"explain simpler" after one of this layer's answers: its own shorter words, the whole
    list, not the same sentences with a prefix (round 43 newcomer, minor 1)."""
    if not _SIMPLER.search(text):
        return None
    questions = [_plain(q) for q in before.split("\n") if q.strip()]
    for asked in reversed(questions):
        said = _shorter_for(asked, " ".join(questions))
        if said:
            return said
    return None


_PRESALE: Final = re.compile(r"\bpre-?sales?\b|\bico\b|\bido\b|\bieo\b|\btoken\s+sale\b", re.I)
_BORROW: Final = re.compile(r"\bborrow\w*|\bloans?\b|\bcredit\s+card\b", re.I)
_LEVERAGE: Final = re.compile(r"\b\d{1,3}\s*x\b|\bleverage\w*|\bfutures\b|\bmargin\b|\bperps?\b",
                              re.I)
_PRESALE_S: Final = (
    "Bottom line: a presale is riskier than buying a listed coin: nothing trades yet, so there "
    "is no way to sell.",
    "The team can vanish with the money, and early buyers' tokens often unlock after listing "
    "and get sold.",
    "Most presales list below their presale price or never list. Use only money you could lose "
    "entirely.",
)
_BORROW_S: Final = (
    "Bottom line: no, do not borrow to buy. The loan must be repaid whatever the price does, "
    "and the interest costs you every month.",
    "\"It always comes back\" is not a promise: ETH stayed more than half below its 2021 high "
    "for most of 2022 and all of 2023 (Yahoo Finance daily closes), and some coins never "
    "come back.",
    "If you buy, use money you could leave alone through a bad year.",
)
_NEW_TOPICS: Final[tuple[tuple[re.Pattern[str], tuple[str, ...] | None], ...]] = (
    (_PRESALE, _PRESALE_S), (_BORROW, _BORROW_S), (_LEVERAGE, None), (_RECOVERY, None),
    (_FAKE_APP, None),
    (_CUSTODY, None), (_BENEFITS, None), (_HARDWARE, None), (_STOCK_SPLIT, None),
    (_INDEX_FUND, None), (_BUY_SHARES, None), (_BALANCE_FX, None), (_ALL_IN, None),
    (_SLIPPAGE, None), (_RUG_DUMP, None), (_DIVIDEND, None), (_LIQ_HAPPENS, None), (_HODL, None),
    (_CAUTION_BUY, None), (_WRONG_SEND, _WRONG_SEND_S))


def _clause(line: str) -> str:
    """The first clause of one line of an answer: its point, without the support."""
    body = line.removeprefix("Bottom line: ")
    cuts = [m.start() for m in re.finditer(r"(?<=[.!?])\s+(?=[A-Z\"0-9])|;\s| — | \(", body)]
    first = body[:next((c for c in cuts if c >= 45), len(body))].strip()
    if len(first) > 170:
        cut = first[:170]
        first = cut.rsplit(",", 1)[0] if "," in cut[60:] else cut.rsplit(" ", 1)[0]
    first = first.rstrip(" ,:")
    if not first.endswith((".", "!", "?", '"')):
        first += "."
    return ("Bottom line: " if line.startswith("Bottom line: ") else "") + first


def _shorten(lines: Lines) -> Lines | None:
    """A genuinely shorter version of ``lines``: every line keeps its first clause, so no point
    is dropped. None when that is not shorter than the original."""
    short = [_clause(line) for line in lines[:5]]
    if len(" ".join(short)) >= 0.8 * len(" ".join(lines)):
        return None
    return short


def _answer_for(question: str) -> Lines | None:
    found = _first_of(question)
    return found[0] if found is not None else _first(question)


def _shorter_for(asked: str, whole: str) -> Lines | None:
    """The shorter version of the answer to one earlier question: this layer's own plain words
    where it has them, otherwise the answer cut to the first clause of each line."""
    if _WRONG_SEND.search(asked) and not _WRONG_NOT.search(asked):
        return list(_WRONG_SEND_S)
    if _RECOVERY.search(asked):
        full = _answer_for(asked)
        return _shorten(full) if full else None
    best: tuple[int, tuple[str, ...] | None] | None = None
    for pattern, simple in (*_SIMPLE_TOPICS, *_NEW_TOPICS):
        hits = list(pattern.finditer(asked))
        if hits and (best is None or hits[-1].end() >= best[0]):
            best = (hits[-1].end(), simple)
    if best is None:
        full = _answer_for(asked)
        return _shorten(full) if full else None
    if best[1] is _WHAT_NOW_S and _WITHDRAWING.search(whole):
        return list(_WITHDRAW_S)
    if best[1] is not None:
        return list(best[1])
    full = _answer_for(asked)
    return _shorten(full) if full else None


_SIMPLE_TOPICS: Final[tuple[tuple[re.Pattern[str], tuple[str, ...]], ...]] = (
        (_P2P, _P2P_S), (_SIGNAL_GROUP, _SIGNAL_S), (_PHISH_BAIT, _PHISH_S),
        (_FORGOT_2FA, _FORGOT_2FA_S), (_RUG, _RUG_S), (_ATM, _ATM_S), (_SIM_SWAP, _SIM_SWAP_S),
        (_REPORT_SCAM, _REPORT_SCAM_S), (_PUMP, _PUMP_S), (_CHART, _CHART_S),
        (_SMALL_SUM, _SMALL_SUM_S), (_DCA, _DCA_S), (_EXCH_WALLET, _EXCH_WALLET_S),
        (_BULL_TRAP, _BULL_TRAP_S), (_BUBBLE, _BUBBLE_S), (_INFLATION, _INFLATION_S),
        (_WD_LIMIT, _WD_LIMIT_S), (_NOW_OR_WAIT, _NOW_OR_WAIT_S), (_PAPER_LOSS, _PAPER_LOSS_S),
        (_BOUGHT_TOP, _BOUGHT_TOP_S), (_TOO_LATE, _TOO_LATE_S), (_WEN_MOON, _WEN_MOON_S),
        (_PRICE_DIFF, _PRICE_DIFF_S), (_FRACTION, _FRACTION_S), (_FAST, _FAST_S),
        (_WHALE, _WHALE_S), (_MARKET_CAP, _MARKET_CAP_S), (_STABLE, _STABLE_S),
        (_STAKING, _STAKING_S), (_TAX_Q, _TAX_S), (_WITHDRAWING, _WITHDRAW_S),
        (_WHAT_NEXT_ANY, _WHAT_NOW_S))


def _follow_if_down(text: str, before: str) -> Lines | None:
    """"and if it goes down?" after a stablecoin or a holding-and-loss answer: the previous turn's
    subject, not a ticker the words were never (round 43 newcomer, major 2 and 3)."""
    if not _IF_DOWN.search(text):
        return None
    earlier = _plain(before)
    if _STABLE_ANY.search(earlier):
        return list(_STABLE_DOWN_A)
    if _LOSS_THREAD.search(earlier):
        return list(_HOLD_DOWN_A)
    return None


def _follow_withdraw_steps(text: str, before: str) -> Lines | None:
    """"ok so what should I do" after a withdrawal answer: the steps, in order, not the generic
    three checks about investing (round 43 newcomer, major 1)."""
    if not _WHAT_NEXT.search(text):
        return None
    if _WITHDRAWING.search(before) and not _WITHDRAWING.search(text):
        return list(_WITHDRAW_STEPS)
    return None


def _follow_tax_country(text: str, before: str) -> Lines | None:
    """"what about the US" after a tax answer (round 43 newcomer, major 9)."""
    if not re.search(r"\btax\w*|\bhmrc\b|\birs\b|\bcapital\s+gains?\b", before, re.I):
        return None
    if not re.match(r"^\W*(?:and\s+|what\s+about\s+|how\s+about\s+|ok(?:ay)?\s+(?:and\s+)?)"
                    r"(?:in\s+|for\s+|if\s+i(?:'?m|\s+am)\s+in\s+)?(?:the\s+)?[A-Za-z.\s]{2,30}\W*$",
                    text, re.I):
        return None
    return _country_tax(text)


# --- the declined-language notice ---------------------------------------------------------------
#
# When the language model is paused for a visitor, `lui/server.py` (the "unread language" branch,
# `PAUSED[code][1]`, "Without the language model this question could not be read; ask it in
# English.") has a fixed refusal for fr, de, es, pt, zh, ja and ko only; Vietnamese, Turkish and
# Indonesian questions got the English line (round 43 newcomer, major 13 and 14, minor 8). These
# are the same three sentences for those languages, in the shape of `translate.PAUSED` plus the
# second line that follows the refusal.

DECLINED_LANGUAGE: Final[dict[str, tuple[str, str, str]]] = {
    "vi": ("Trả lời bằng tiếng Anh: mô hình ngôn ngữ đang "
           "tạm dừng cho mạng của bạn tối đa {minutes} phút "
           "(hết hạn mức theo giờ). Các con số vẫn được "
           "tính trực tiếp.",
           "Không có mô hình ngôn ngữ (đang tạm dừng cho "
           "mạng của bạn tối đa {minutes} phút) nên không "
           "đọc được câu hỏi này. Hãy hỏi bằng "
           "tiếng Anh, hoặc hỏi lại sau.",
           "Khi không có mô hình ngôn ngữ, câu hỏi bằng ngôn "
           "ngữ này chỉ được đọc khi hỏi giá; các "
           "câu khác được trả lời bằng tiếng Anh, ví "
           "dụ \"what did TSLA's latest earnings report say\"."),
    "tr": ("\u0130ngilizce yan\u0131t: dil modeli ağ\u0131n\u0131z için en fazla {minutes} "
           "dakika duraklat\u0131ld\u0131 (saatlik kota doldu). Rakamlar yine canl\u0131 "
           "hesaplan\u0131r.",
           "Dil modeli ağ\u0131n\u0131z için en fazla {minutes} dakika duraklat\u0131ld\u0131ğ"
           "\u0131ndan bu soru okunamad\u0131. Lütfen \u0130ngilizce sorun ya da biraz sonra "
           "tekrar deneyin.",
           "Dil modeli olmadan bu dildeki bir soru yaln\u0131zca fiyat soruluyorsa okunur; diğer "
           "her şey \u0130ngilizce yan\u0131tlan\u0131r, örneğin \"what did TSLA's "
           "latest earnings report say\"."),
    "id": ("Jawaban dalam bahasa Inggris: model bahasa dijeda untuk jaringan Anda paling lama "
           "{minutes} menit (kuota per jam habis). Angka tetap dihitung langsung.",
           "Tanpa model bahasa (dijeda untuk jaringan Anda paling lama {minutes} menit), "
           "pertanyaan ini tidak dapat dibaca. Tanyakan dalam bahasa Inggris, atau coba lagi "
           "nanti.",
           "Tanpa model bahasa, pertanyaan dalam bahasa ini hanya dibaca bila menanyakan harga; "
           "selebihnya dijawab dalam bahasa Inggris, misalnya \"what did TSLA's latest earnings "
           "report say\"."),
    # round 44: Italian, Polish and Swahili got the English refusal, Russian and Hindi (which
    # `translate.target_language` detects) had no fixed text at all, and neither had Arabic or Thai
    "it": ("Risposta in inglese: il modello linguistico è in pausa per la tua rete per al "
           "massimo {minutes} minuti (quota oraria esaurita). I numeri sono comunque calcolati "
           "in tempo reale.",
           "Senza il modello linguistico, in pausa per la tua rete per al massimo {minutes} "
           "minuti, non è stato possibile leggere questa domanda. Falla in inglese, oppure "
           "riprova più tardi.",
           "Senza il modello linguistico, una domanda in questa lingua viene letta solo se "
           "chiede un prezzo; tutto il resto riceve risposta in inglese, per esempio \"what did "
           "TSLA's latest earnings report say\"."),
    "pl": ("Odpowiedź po angielsku: model językowy jest wstrzymany dla Twojej sieci na "
           "maksymalnie {minutes} min (wyczerpany limit godzinowy). Liczby nadal są liczone "
           "na żywo.",
           "Bez modelu językowego, wstrzymanego dla Twojej sieci na maksymalnie {minutes} min, "
           "nie udało się odczytać tego pytania. Zadaj je po angielsku lub spróbuj ponownie "
           "później.",
           "Bez modelu językowego pytanie w tym języku jest odczytywane tylko wtedy, gdy "
           "dotyczy ceny; na resztę odpowiadamy po angielsku, na przykład \"what did TSLA's "
           "latest earnings report say\"."),
    "sw": ("Jibu kwa Kiingereza: modeli ya lugha imesitishwa kwa mtandao wako kwa hadi dakika "
           "{minutes} (kikomo cha saa kimeisha). Namba bado zinahesabiwa moja kwa moja.",
           "Bila modeli ya lugha, iliyositishwa kwa mtandao wako kwa hadi dakika {minutes}, "
           "swali hili halikuweza kusomwa. Uliza kwa Kiingereza, au jaribu tena baadaye.",
           "Bila modeli ya lugha, swali katika lugha hii husomwa tu linapouliza bei; mengine "
           "hujibiwa kwa Kiingereza, kwa mfano \"what did TSLA's latest earnings report say\"."),
    "ru": ("Ответ на английском: языковая модель приостановлена для вашей сети не более "
           "чем на {minutes} мин. (часовой лимит исчерпан). Числа по-прежнему считаются "
           "в реальном времени.",
           "Без языковой модели, приостановленной для вашей сети не более чем на {minutes} "
           "мин., этот вопрос прочитать не удалось. Задайте его по-английски или повторите "  # noqa: RUF001
           "позже.",
           "Задайте вопрос по-английски, например \"what is NVDA's price right now\"."),
    "hi": ("अंग्रेज़ी में जवाब: भाषा मॉडल आपके नेटवर्क के लिए अधिकतम {minutes} मिनट के लिए रुका "
           "हुआ है (घंटे की सीमा पूरी हो गई)। आँकड़े अब भी लाइव गणना से आते हैं।",
           "भाषा मॉडल के बिना, जो आपके नेटवर्क के लिए अधिकतम {minutes} मिनट के लिए रुका हुआ है, यह "
           "प्रश्न पढ़ा नहीं जा सका। कृपया इसे अंग्रेज़ी में पूछें या बाद में दोबारा कोशिश करें।",
           "कृपया अंग्रेज़ी में पूछें, उदाहरण के लिए \"what is NVDA's price right now\"।"),
    "ar": ("الرد بالإنجليزية: نموذج اللغة متوقف مؤقتًا لشبكتك لمدة أقصاها {minutes} دقيقة "
           "(استُنفدت الحصة الساعية). الأرقام ما زالت تُحسب مباشرة.",
           "بدون نموذج اللغة، المتوقف مؤقتًا لشبكتك لمدة أقصاها {minutes} دقيقة، تعذّرت قراءة "
           "هذا السؤال. اطرحه بالإنجليزية أو حاول مرة أخرى لاحقًا.",
           "اطرح سؤالك بالإنجليزية، مثل \"what is NVDA's price right now\"."),
    "th": ("ตอบเป็นภาษาอังกฤษ: โมเดลภาษาหยุดชั่วคราวสำหรับเครือข่ายของคุณนานสูงสุด {minutes} นาที "
           "(โควตารายชั่วโมงหมด) ตัวเลขยังคำนวณสดอยู่",
           "หากไม่มีโมเดลภาษา ซึ่งหยุดชั่วคราวสำหรับเครือข่ายของคุณนานสูงสุด {minutes} นาที "
           "จึงอ่านคำถามนี้ไม่ได้ กรุณาถามเป็นภาษาอังกฤษ หรือลองใหม่ภายหลัง",
           "กรุณาถามเป็นภาษาอังกฤษ เช่น \"what is NVDA's price right now\""),
}
"""Per language code (as `translate.target_language` returns it): the note above an English answer,
the refusal, and the line after it, each with ``{minutes}`` where the wait is said (the third has
none). Written for the three languages `translate.PAUSED` lacks."""


def declined_language(code: str | None, minutes: int) -> tuple[str, str, str] | None:
    """(note, refusal, hint) in the question's language with the wait filled in, or None when
    ``code`` is not one of the languages in :data:`DECLINED_LANGUAGE`."""
    if code is None or code not in DECLINED_LANGUAGE:
        return None
    note, refusal, hint = DECLINED_LANGUAGE[code]
    return note.format(minutes=minutes), refusal.format(minutes=minutes), hint


# --- round 44 follow-ups: "so is it safe", "what should I check first", "so can I get it back" ---
#
# Read with the question before them. The audit found all three answered about the console, or
# declined, because the follow-up lost its subject.

_SAFE_FOLLOW: Final = re.compile(
    r"^\W*(?:(?:so|ok(?:ay)?|but|and|then|well|hmm+)\W+){0,3}(?:is\s+(?:it|that|this|they|he|"
    r"she)|are\s+(?:they|these|those)|would\s+(?:it|that)\s+be|can\s+i\s+trust\s+(?:it|them|"
    r"that)|should\s+i\s+trust\s+(?:it|them|that))\s*(?:really\s+|actually\s+|even\s+|all\s+)?"
    r"(?:safe|legit|legitimate|risky|ok|okay|a\s+scam|dangerous|trustworthy|secure|fine|"
    r"worth\s+it)(?:\s+or\s+not|\s+though|\s+then)?\W*$", re.I)
_CHECK_FOLLOW: Final = re.compile(
    r"^\W*(?:(?:so|ok(?:ay)?|but|and|then|well|hmm+)\W+){0,3}(?:what\s+(?:should|do|can|must|"
    r"shall)\s+i\s+(?:check|verify|look\s+(?:at|for|into)|confirm|do)\s+first|what\s+(?:should|"
    r"do)\s+i\s+(?:check|verify)|what\s+to\s+(?:check|verify)(?:\s+first)?|how\s+(?:do|can)\s+i\s+"
    r"(?:check|verify)\s+(?:it|that|this)|where\s+(?:do|should)\s+i\s+start|what\s+first|"
    r"checklist|what'?s\s+the\s+first\s+thing\s+(?:i\s+)?(?:should\s+)?(?:check|do))\W*$", re.I)

_TOPIC_TABLE: Final[tuple[tuple[re.Pattern[str], str], ...]] = (
    (_SIM_SWAP, "other"), (_FORGOT_2FA, "other"), (_CHART, "other"), (_SMALL_SUM, "other"),
    (_BULL_TRAP, "other"), (_BUBBLE, "other"), (_INFLATION, "other"), (_WD_LIMIT, "other"),
    (_NOW_OR_WAIT, "other"), (_PAPER_LOSS, "other"), (_BOUGHT_TOP, "other"),
    (_TOO_LATE, "other"), (_WEN_MOON, "other"), (_PRICE_DIFF, "other"), (_FRACTION, "other"),
    (_FAST, "other"), (_WHALE, "other"), (_MARKET_CAP, "other"), (_SEED, "wallet"),
    (_GAS, "other"), (_KYC_WHY, "kyc"), (_BITCOIN_SAFE, "other"), (_HACKED, "other"),
    (_FX, "other"), (_EMERGENCY, "other"), (_SCAM_DOUBLE, "other"), (_FRIEND_LEVERAGE, "leverage"),
    (_LOSS_FELT, "other"), (_REPORT_SCAM, "other"), (_PHISH_BAIT, "other"),
    (_P2P, "p2p"), (_SIGNAL_GROUP, "signal"), (_ATM, "atm"), (_STABLE_ANY, "stable"),
    (_RUG, "rug"), (_PUMP, "rug"), (_SCAM_COIN, "rug"), (_EXCHANGE_SCAM, "fake_app"),
    (_EXCH_WALLET, "wallet"), (_WALLET, "wallet"), (_TOKENISED, "shares"), (_DCA, "dca"),
    (_SIP, "dca"), (_STAKING, "staking"), (_TAX_Q, "tax"),
    (_PRESALE, "presale"), (_BORROW, "borrow"), (_LEVERAGE, "leverage"),
    (_RECOVERY, "recovery"), (_FAKE_APP, "fake_app"), (_CUSTODY, "wallet"),
    (_BENEFITS, "benefits"), (_HARDWARE, "wallet"), (_STOCK_SPLIT, "split"),
    (_INDEX_FUND, "index"), (_BUY_SHARES, "shares"), (_BALANCE_FX, "other"),
    (_ALL_IN, "all_in"), (_SLIPPAGE, "other"), (_RUG_DUMP, "rug"), (_DIVIDEND, "dividend"),
    (_LIQ_HAPPENS, "leverage"), (_HODL, "hodl"), (_CAUTION_BUY, "first_buy"),
    (_WRONG_SEND, "wrong_send"))
"""Every topic this layer answers, to the key its follow-ups are looked up under; ``other`` takes
the general safe and check answers."""


def _topic(before: str) -> str | None:
    """The topic of the latest question in ``before`` that is one of this layer's."""
    best: tuple[int, int, str] | None = None
    for index, asked in enumerate(q for q in before.split("\n") if q.strip()):
        plain = _plain(asked)
        for pattern, key in _TOPIC_TABLE:
            ends = [m.end() + (100000 if key in ("wrong_send", "recovery") else 0)
                    for m in pattern.finditer(plain)]
            if ends and (best is None or (index, max(ends)) >= best[:2]):
                best = (index, max(ends), key)
    return best[2] if best is not None else None


_SAFE_BY_TOPIC: Final[dict[str, tuple[str, ...]]] = {
    "presale": (
        "Bottom line: no — a presale is among the riskier things you can buy: the token often has "
        "no working product yet, nobody independent checks the claims, and many never list or "
        "list far below the presale price.",
        "An anonymous team, a promised return, or \"last chance\" pressure are warnings. Only an "
        "amount you could lose entirely belongs in one, bought from the project's official site "
        "that you reached yourself."),
    "borrow": (
        "Bottom line: no — borrowing to invest is not safe: the loan must be repaid whatever the "
        "price does, and the interest is a cost every month.",
        "A fall then loses money you do not have, and a lender can force you to sell at the "
        "worst moment. Invest only money you could lose, never borrowed money."),
    "index": (
        "Bottom line: safer than one stock or coin, because it spreads your money over many "
        "companies, but not safe — if the whole market falls the fund falls with it (the S&P 500 "
        "index lost about 19% in 2022).",
        "Markets have tended to recover over long periods, but that is a record, not a promise; "
        "only money you can leave alone for years belongs in it."),
    "staking": (
        "Bottom line: not risk-free — the reward is paid in the coin, whose price can fall by "
        "more than the reward, and locked coins may not be sellable until the lock ends.",
        "Check who holds the coins while they are staked, and be wary of an unusually high rate: "
        "it usually pays for higher risk."),
    "leverage": (
        "Bottom line: for a beginner, no — leverage multiplies losses as well as gains, and a "
        "small move against you can liquidate the position.",
        "On spot with no leverage the most you can lose is what you paid; if you ever use "
        "leverage, keep it low and set a stop-loss first."),
    "p2p": (
        "Bottom line: it can be, if you release coins only after the money is in your own bank "
        "app and keep everything inside the platform's order and chat.",
        "The risk is the other person: fake payment screenshots and bank payments that are "
        "reversed later are the usual losses."),
    "signal": (
        "Bottom line: no — treat any signal group as unsafe until proven otherwise; many are "
        "pump-and-dump schemes or lead to fake platforms.",
        "Never pay to join, never send money to an admin, and never log in through a link from "
        "the group."),
    "atm": (
        "Bottom line: not for paying anyone else — if someone told you to use a crypto ATM to pay "
        "a fine or a tax, or to \"protect\" your money, it is a scam.",
        "Even for your own purchase the fees are high and the transfer cannot be reversed."),
    "stable": (
        "Bottom line: safer than a volatile coin, not safe like a bank deposit — TerraUSD lost "
        "its peg in May 2022 and USDC dipped to about $0.87 in March 2023.",
        "It is not insured; hold only what you need, and spread it across more than one large "
        "issuer."),
    "wallet": (
        "Bottom line: each has a different risk — an exchange can be hacked or freeze "
        "withdrawals, and a wallet you control puts the whole risk on you, because nobody can "
        "recover a lost recovery phrase.",
        "Whichever you use: a unique password, an authenticator app, the recovery phrase on paper "
        "and offline, and only an amount you could lose."),
    "shares": (
        "Bottom line: it is a different risk from owning the thing itself — a tokenised stock "
        "tracks the price but you are not a shareholder, and it trades around the clock so its "
        "price can drift from the stock's.",
        "The price still moves like the stock, so you can lose money on it; a regulated broker is "
        "the place to own actual shares."),
    "dca": (
        "Bottom line: it reduces the risk of one bad entry day, not the risk of the asset itself "
        "— regular buys into something that falls 70% still fall 70%.",
        "Keep the monthly amount to what you could lose without it hurting."),
    "rug": (
        "Bottom line: no coin can be called safe from a rug pull or a pump and dump, but there "
        "are ways to lower the risk.",
        "Stay with coins listed on a large exchange, check who holds most of the supply and "
        "whether liquidity is locked, and never put in more than you could lose."),
    "recovery": (
        "Bottom line: no — a recovery service that contacts you and asks for a fee is not safe; "
        "it is a second scam.",
        "Do not pay, and give no recovery phrase, password or remote access. Report to the "
        "exchange you used and to the police."),
    "fake_app": (
        "Bottom line: only if it comes from the company's own website or the official store "
        "listing that site links to, and the publisher name matches.",
        "Test with a small deposit and withdraw it before more; an app that blocks withdrawals "
        "is a fake."),
    "wrong_send": (
        "Bottom line: sending it again is not safe until you know where the first transfer went "
        "— a second mistake costs a second amount.",
        "Look up the transaction hash first, then ask the receiving platform's support; ignore "
        "anyone who offers to recover it for a fee."),
    "all_in": (
        "Bottom line: no — putting everything in one coin is not safe: it can lose more than half "
        "its value, as Bitcoin did in 2022 (about 65%).",
        "A small amount you could lose, bought in steps, is a much safer way in."),
    "dividend": (
        "Bottom line: a dividend is never guaranteed: a company can cut or stop it, and the share "
        "price drops by about the dividend on the ex-dividend date.",
        "It is a payment out of the company's own value, not extra money; reinvested or not, the "
        "share can still fall."),
    "split": (
        "Bottom line: a split itself is neutral: you own more shares at a lower price each, with "
        "the same total value.",
        "What can still hurt you is the price move that follows for other reasons, and a reverse "
        "split by a company whose price had collapsed."),
    "hodl": (
        "Bottom line: holding is not safe in itself: it only works if what you hold recovers, and "
        "Bitcoin fell about 65% in 2022 while many smaller coins never came back.",
        "Only money you can leave alone for years, no leverage, and a loss level decided in "
        "advance make it a plan rather than a hope."),
    "first_buy": (
        "Bottom line: a first buy is a limited risk only if it is small: Bitcoin can lose more "
        "than half its value, and has.",
        "On a large regulated exchange you reached yourself, with no leverage and money you could "
        "lose, the risk is the price itself rather than a scam or a forced sale."),
    "kyc": (
        "Bottom line: giving ID to a large, regulated exchange through its official app is "
        "normal and required by law, but no company's data is perfectly safe.",
        "Give it only there, never to someone who messages you, and turn on 2FA."),
}
_SAFE_GENERIC: Final = (
    "Bottom line: nothing in crypto is safe in every case, so the honest answer is: it depends "
    "on who runs it, whether you can get your money out, and how much you put in.",
    "Safer: a large regulated exchange you reached yourself, a small amount you could lose, no "
    "leverage or borrowed money, and a test withdrawal before more.",
    "Unsafe signs: guaranteed returns, pressure to hurry, a stranger who contacted you first, or "
    "being asked for a code, a password or a recovery phrase.",
)
_CHECK_BY_TOPIC: Final[dict[str, tuple[str, ...]]] = {
    "presale": (
        "Bottom line: check five things first: who is behind it, whether there is an independent "
        "audit, how the tokens are split, what is promised, and where the link came from.",
        "1. Named people with a record you can verify, not an anonymous team. 2. A security audit "
        "you can read. 3. How much the team and early buyers hold, and when they may sell. 4. Any "
        "promised return or \"last chance\" is a warning. 5. The official site, reached by you, "
        "never a link sent to you.",
        "Then only an amount you could lose entirely."),
    "borrow": (
        "Bottom line: check four things before borrowing for any investment — and if one fails, "
        "do not borrow.",
        "1. The interest rate against what the investment could realistically earn. 2. Whether "
        "you could repay from your income if it went to zero. 3. Whether the lender can demand "
        "repayment or sell your assets. 4. Whether you already have an emergency fund."),
    "index": (
        "Bottom line: check what it follows, what it costs, and where you buy it.",
        "1. The index it tracks and what that holds. 2. The yearly fee (the expense ratio): lower "
        "is better. 3. A broker regulated in your country. 4. Whether you can leave the money "
        "alone for five years or more."),
    "staking": (
        "Bottom line: check the lock-up, who holds the coins, and where the reward comes from.",
        "1. How long coins are locked and how long unstaking takes. 2. Whether the platform or "
        "the network holds them. 3. A rate far above the rest is a warning. 4. The coin's price "
        "can still fall by more than the reward."),
    "leverage": (
        "Bottom line: check your liquidation price, your size and your stop-loss before you open "
        "anything.",
        "1. The liquidation price on the order screen. 2. That the position is small enough to "
        "lose entirely. 3. A stop-loss set before the liquidation price. 4. The fees and funding "
        "you will pay while it is open."),
    "p2p": (
        "Bottom line: check the money in your own bank app before you release anything.",
        "1. The payment really arrived, in your own banking app. 2. The payer's name matches their "
        "verified name. 3. The chat stayed inside the platform. 4. Release only then; never "
        "because of a screenshot."),
    "signal": (
        "Bottom line: check whether anyone has actually withdrawn profit, and who asks for money.",
        "1. Ask a member whether they have withdrawn profit to their own bank. 2. Any request to "
        "pay, or to send money to an admin, is a stop. 3. Never log in through a link from the "
        "group."),
    "atm": (
        "Bottom line: check who told you to use it — if someone else did, stop.",
        "1. A fine, tax or \"safe locker\" demand is a scam. 2. Compare the ATM's price and fee "
        "with an exchange. 3. Send only to a wallet you control, a small amount first."),
    "stable": (
        "Bottom line: check who issues it, what backs it and whether it has held its peg.",
        "1. The issuer, and whether it publishes reserve reports. 2. Its past dips below $1. "
        "3. That you are not putting everything in one stablecoin."),
    "wallet": (
        "Bottom line: check where the recovery phrase is, where you got the app or device, and "
        "the address on every transfer.",
        "1. The phrase is written on paper, offline. 2. The wallet is from the maker's own site or "
        "official store. 3. A small test transfer first. 4. Address and network checked twice."),
    "shares": (
        "Bottom line: check whether you are buying the share or a token that tracks it, and who "
        "runs the place you buy.",
        "1. Owning the share needs a regulated broker; a tokenised stock tracks the price only. "
        "2. The fees. 3. The trading hours. 4. Your country's regulator lists the broker."),
    "dca": (
        "Bottom line: check the amount, the fee on each buy and whether you can stick to it.",
        "1. A monthly sum you could lose without it hurting. 2. The fee per purchase. 3. That "
        "you will keep going when the price falls."),
    "rug": (
        "Bottom line: check who holds the supply, whether liquidity is locked and whether the "
        "team is known.",
        "1. Holders on a block explorer: a few wallets with most of it is a warning. 2. Liquidity "
        "locked, and for how long. 3. An independent audit. 4. Is it listed on a large exchange. "
        "5. No promised returns."),
    "recovery": (
        "Bottom line: check who contacted whom — if they came to you, stop.",
        "1. Did they message you first? 2. Do they want money or crypto up front? 3. Do they "
        "promise a result? Any yes is a scam. Report the theft to the exchange and the police."),
    "fake_app": (
        "Bottom line: check the publisher, the source of the link and a small withdrawal.",
        "1. The publisher name matches the company. 2. The link came from the company's own "
        "website, typed by you. 3. Reviews and download history are long. 4. A small deposit "
        "can be withdrawn."),
    "wrong_send": (
        "Bottom line: check three things in order: the transaction hash, the address it went to, "
        "and the network you used.",
        "1. Find the hash in your wallet or exchange history and open it on that network's block "
        "explorer. 2. Is the address yours, an exchange's deposit address, or a contract? 3. Does "
        "the receiver support the network you used? Then contact the receiving platform's "
        "support through its official app.",
        "Do not send again, and ignore anyone offering to recover it for a fee."),
    "all_in": (
        "Bottom line: check four things before any of it goes in.",
        "1. That it is money you could lose without changing your life. 2. That you have an "
        "emergency fund in cash first. 3. That none of it is borrowed and none is leveraged. 4. "
        "That you still want it after waiting a day."),
    "tax": (
        "Bottom line: check your country's rules and your records before you sell.",
        "1. Your tax authority's page on crypto. 2. The date, amount and price of each buy and "
        "sale. 3. Whether swaps and spending count as sales where you live. Not tax advice."),
    "benefits": (
        "Bottom line: check your benefit's own rules on savings, assets and income, and ask the "
        "office in writing.",
        "1. The official guidance for your benefit. 2. Whether holdings or only gains count. "
        "3. When a change must be reported."),
    "dividend": (
        "Bottom line: check when it is paid, whether the company can afford it, and what the "
        "share is worth without it.",
        "1. The ex-dividend date: you must own the share before it. 2. Whether the company has "
        "paid steadily and earns more than it pays out. 3. That the price drops by about the "
        "dividend on that date. 4. Any tax on it in your country."),
    "split": (
        "Bottom line: check what kind of split it is and why it was done.",
        "1. The ratio (2-for-1 doubles the shares and halves the price) or a reverse split. 2. "
        "That your total value is unchanged on the day. 3. The company's reason: a reverse split "
        "after a collapse is a warning."),
    "hodl": (
        "Bottom line: check the amount, the time you can wait, and your exit rule.",
        "1. Money you could leave alone for years. 2. No leverage, so you cannot be forced out. "
        "3. The loss at which you would sell, decided now. 4. One coin or several: single coins "
        "can fail."),
    "first_buy": (
        "Bottom line: check the exchange, the amount and the account's security before the "
        "first buy.",
        "1. A large regulated exchange you reached yourself. 2. An amount you could lose without "
        "it hurting. 3. 2FA with an authenticator app and a unique password. 4. The fees on the "
        "buy and on withdrawal."),
    "kyc": (
        "Bottom line: check that you are in the official app or on the address you typed.",
        "1. Never upload ID from a link in a message. 2. Turn on 2FA first. 3. Use a unique "
        "password."),
}
_CHECK_GENERIC: Final = (
    "Bottom line: check three things first: who is behind it, whether you can get your money out, "
    "and how much you could lose.",
    "1. Who runs it, and can you verify that outside their own message or site? 2. Try a small "
    "amount and withdraw it before more. 3. Put in only what you could lose entirely, with no "
    "leverage or borrowed money.",
    "Anyone who promises a return, rushes you, or asks for a code or recovery phrase fails the "
    "check.",
)


def _follow_safe(text: str, before: str) -> Lines | None:
    """"so is it safe or not" read with the subject of the question before it."""
    if not _SAFE_FOLLOW.search(text):
        return None
    key = _topic(before)
    if key is None:
        return None
    return list(_SAFE_BY_TOPIC.get(key, _SAFE_GENERIC))


def _follow_checklist(text: str, before: str) -> Lines | None:
    """"what should I check first" read with the subject of the question before it."""
    if not _CHECK_FOLLOW.search(text):
        return None
    key = _topic(before)
    if key is None:
        return None
    return list(_CHECK_BY_TOPIC.get(key, _CHECK_GENERIC))


def _follow_get_back(text: str, before: str) -> Lines | None:
    """"so can I get it back" after a wrong-network or wrong-address send: the recovery answer,
    not the chasing-losses one (round 44 newcomer, major 5)."""
    if not _WRONG_BACK.search(text) or _topic(before) != "wrong_send":
        return None
    return list(_WRONG_BACK_A)


_FOLLOWS: Final[tuple[Callable[[str, str], Lines | None], ...]] = (
    _follow_get_back, _follow_simpler, _follow_if_down, _follow_withdraw_steps,
    _follow_tax_country, _follow_seed, _follow_fees, _follow_leverage_move, _follow_order_type,
    _follow_tax, _follow_kyc, _follow_scammer, _follow_checklist, _follow_safe)


def early(text: str, prior: Sequence[str]) -> Lines | None:
    """A beginner's answer, or None to let the rest of the console read the question."""
    plain = _plain(text)
    if prior and _short(plain):
        before = "\n".join(prior[-2:])
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


_WHOLE: Final = frozenset({*_ROUND44, *_ROUND43, _seed, _wallet, _bitcoin_safe, _first_buy, _gas,
                           _kyc_why, _fx, _loss_felt, _win_back, _scam_double,
                           _friend_leverage, _tokenised,
                           _sip, _exchange_scam, _fomo})
"""Answers that already cover the usual second half of their question ("what is bitcoin, is it
safe for beginners"); only the others are split and answered part by part."""


__all__ = ["DECLINED_LANGUAGE", "declined_language", "early"]
