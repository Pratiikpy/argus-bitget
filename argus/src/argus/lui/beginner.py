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
                 (r"\bwaht\b", "what"), (r"\brugpull", "rug pull"))


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
    "vanish — they drain the funds from the trading pool, or sell their huge holdings all at "
    "once — so the price falls to nearly zero and you cannot get out.",
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
    r"\b(?:invest\w*|start\w*|begin)\b[^?]{0,30}\b(?:(?:only|just)\s+|with\s+(?=\$?\d{1,3}\s*"
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
    return None


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


_FIRSTS: Final[tuple[Callable[[str], Lines | None], ...]] = (
    _seed, *_ROUND43, _wallet, _loss_felt, _win_back, _scam_double, _friend_leverage,
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
    r"(?:words|terms|english)\b|\bdumb\s+it\s+down\b|\beli5\b", re.I)
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
    earlier = _plain(before)
    topics: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
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
    best: tuple[int, tuple[str, ...]] | None = None
    for pattern, simple in topics:
        hits = list(pattern.finditer(earlier))
        if hits and (best is None or hits[-1].end() >= best[0]):
            best = (hits[-1].end(), simple)
    if best is None:
        return None
    if best[1] is _WHAT_NOW_S and _WITHDRAWING.search(earlier):
        return list(_WITHDRAW_S)
    return list(best[1])


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


_FOLLOWS: Final[tuple[Callable[[str, str], Lines | None], ...]] = (
    _follow_simpler, _follow_if_down, _follow_withdraw_steps, _follow_tax_country,
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


_WHOLE: Final = frozenset({*_ROUND43, _seed, _wallet, _bitcoin_safe, _first_buy, _gas,
                           _kyc_why, _fx, _loss_felt, _win_back, _scam_double,
                           _friend_leverage, _tokenised,
                           _sip, _exchange_scam, _fomo})
"""Answers that already cover the usual second half of their question ("what is bitcoin, is it
safe for beginners"); only the others are split and answered part by part."""


__all__ = ["DECLINED_LANGUAGE", "declined_language", "early"]
