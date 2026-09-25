"""Is a post abusive? The screen that keeps slurs, profanity and threats out of the crowd read.

**Why this exists (2026-09-25).** `market/social_pulse.py` counted every post that named a ticker
and published the text of the most-carried stories, verbatim, into a snapshot the hosted console
reads. Nothing looked at what the text said. Run over TweetEval's hate and offensive test sets
through the real `pulse()` path (`eval/sentiment_tweeteval.py`), every abusive post was counted
as crowd volume, and a brigade of three accounts posting one slur became a "coordinated story"
that `lines_for` quoted back to the user as its example. This module is the fix: a post it flags
is withheld from every count and never quoted, and the snapshot says how many were withheld.

**What it is.** A lexicon-and-pattern screen, pure Python, no model and no network — the pulse
runs on the desk machine inside a sweep, and `pyproject.toml` keeps torch out of the runtime
dependencies. Three matchers, any one of which flags a post:

1. **Obscenity's curated English set**, ported from ``jo3-l/obscenity`` (MIT, read at commit
   121bc42, ``src/preset/english.ts:103-379``): 69 phrases, each a set of patterns in their small
   syntax (``|`` a word boundary, ``[x]`` optional, ``?`` any one character) plus the whitelisted
   terms that stop the Scunthorpe problem (``assess``, ``tetanus``, ``cockney``, ``kung fu``).
   Adapted, changed: the TypeScript ``pattern`` tag is compiled to a Python regex by
   :func:`_compile` following their ``compilePatternToRegExp`` (``src/pattern/Util.ts:8-37``),
   with ``re.ASCII`` so ``\\b`` means what it means in JavaScript; their confusables table is
   replaced by Unicode NFKD compatibility decomposition, which folds full-width and accented
   letters but not every lookalike they map; the two exact duplicates in the upstream ``anal``
   phrase (``lanal`` and ``lan al`` listed twice) are listed once.
2. **Cuss, rating 2**, from ``words/cuss`` (MIT, read at commit 6bab3fe, ``index.js:6-1802``):
   the 1,255 English entries its author rates *likely* to be used as a profanity or slur, copied
   verbatim (:data:`CUSS_RATED_2`, pinned by :data:`CUSS_RATED_2_SHA256`), matched as whole
   tokens after the same transforms, with a plain ``s`` plural for entries of four letters or
   more. Cuss's own README says not to build a profanity filter from it; the answer to that
   warning is not to trust the filter but to measure it, which is what the eval module does.
3. **ARGUS's own threat and harassment patterns** (:data:`THREAT_PATTERNS`): "kill yourself",
   "should be shot/hanged/lynched/gassed", dehumanising predicates ("they are vermin"), "go back
   to your country", "stfu". Written here, not taken from anywhere, and deliberately narrow.

The transforms before matching are obscenity's recommended English chain
(``src/preset/english.ts:13-30``): leetspeak resolved with their dictionary
(``src/transformer/resolve-leetspeak/dictionary.ts:1-12``: ``@``/``4``→a, ``$``/``5``→s, …),
ASCII lower-cased, then runs of one character collapsed to one copy — two for ``b e o l s g``, so
``ass``, ``boob`` and ``nigger`` survive — per their collapse-duplicates transformer
(``src/transformer/collapse-duplicates/transformer.ts:18-27``). A match is dropped when its span
in the original text lies wholly inside a whitelisted term, exactly their
``IntervalCollection.query`` rule (``src/matcher/IntervalCollection.ts:13-33``). Masked words
(``f*ck``, ``sh*t``, ``a**``) are matched by treating each run of ``*`` as that many letters.

**What was rejected, and why — measured, not argued.**

- *Cuss entries rated 1* ("maybe profane", 285 more entries): on TweetEval's validation splits
  they raised offensive-class recall from 0.377 to 0.479 and hate-class recall from 0.454 to
  0.515, but withheld 8.6% of ordinary tweets instead of 4.5% and 10.2% of clean OffensEval posts
  instead of 5.0%, because the rating-1 list carries words like "abortion", "addict" and "god".
  A crowd read that silently drops one post in eleven is worse than one that misses some
  insults. (Re-measured 2026-09-25 with upstream ``index.js`` read outside the repository and
  this module's matcher unchanged; rating-1 entries are not shipped, so this one comparison
  cannot be re-run from the repository — every shipped variant's can.)
- *Suffix stripping* (``-ing``, ``-es``, ``-y``): it turned "pricing" into a match for a rating-2
  entry and "spicy" into an ethnic slur. Found on the desk's own 25 September snapshot of real X
  and Reddit posts, not on the benchmark. Only the plain plural survives.
- *Two-letter entries* ("fu", "ho"): "ho ho ho" and "kung fu" are not abuse. Obscenity's own
  ``|fu|`` pattern, which carries a "kung fu" whitelist, is kept.
- *A trained toxicity model.* TweetEval's own RoBERTa (RoB-RT), whose published test predictions
  the eval re-scores on the same items, reaches macro-F1 0.555 on hate and 0.816 on offensive
  against this screen's 0.524 and 0.695, and finds 66.7% of offensive posts where this finds
  40.4% — it is better, and the artefact says by how much. It would put torch and a checkpoint
  into the runtime, and ARGUS's shipped code stays dependency-light by standing rule. Detoxify
  was not run: no checkpoint is on this machine.

**The market-jargon allowlist is ours** (:data:`MARKET_ALLOWLIST`). Some ordinary trading phrases
contain a listed word: "top ETF gainers & losers" and "let's dumb it down" were withheld on the
desk's real snapshots, "dumb money" in probes; "sucker rally", "shitcoin", "rapeseed" futures,
"Dick's Sporting Goods" and "cockpit" are the same failure waiting to happen. A match inside one
of these phrases is ignored the way obscenity ignores a whitelisted term. The profane word
*outside* the phrase still counts: "this shitcoin is shit" is withheld. On TweetEval's test sets
the allowlist changes no verdict at all (it excused one validation post), so it costs the
benchmark nothing; its effect is on market text, and the eval reports it there.

**What it cannot do, stated so nobody has to discover it.** It reads words, not intent. Hostility
with no listed word in it passes and is still counted: 38.6% of the posts TweetEval's hate test
set labels hateful and 59.6% of those its offensive test set labels offensive, measured
2026-09-25. And it withholds profanity whatever it is aimed at, so 5.3% of ordinary sentiment
test tweets are withheld too — a crowd read that drops an excited post because it swears loses a
real voice, and that is the price of never quoting abuse. `eval/sentiment_tweeteval.py`
publishes every one of these numbers, per sentiment class, in ``data/sentiment_tweeteval.json``.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field

REASON_TERM = "abusive_term"
REASON_THREAT = "threat_or_harassment"

_LEET: dict[str, str] = {
    "@": "a", "4": "a", "(": "c", "3": "e", "6": "g", "1": "i", "!": "i", "/": "l", "0": "o",
    "$": "s", "5": "s", "7": "t", "2": "z",
}
"""Obscenity's leetspeak dictionary (``resolve-leetspeak/dictionary.ts:1-12``), inverted to one
character each."""

_COLLAPSE_LIMIT: dict[str, int] = {"b": 2, "e": 2, "o": 2, "l": 2, "s": 2, "g": 2}
"""Consecutive copies kept per character; one for every other character
(``src/preset/english.ts:19-29``)."""

OBSCENITY: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("abo", ("|ab[b]o[s]|",), ()),
    ("abeed", ("ab[b]eed",), ()),
    ("africoon", ("africoon",), ()),
    ("anal",
     ("|anal", "danal", "eanal", "fanal", "ganal", "ianal", "janal", "kanal", "lanal", "oanal|",
      "panal", "qanal", "ranal", "sanal", "tanal", "uanal", "vanal", "wanal", "xanal", "yanal",
      "zanal"),
     ("analabos", "analagous", "analav", "analy", "analog", "an al", "fan al", "gan al", "ian al",
      "trojan al", "lan al", "pan al", "tan al", "uan al", "van al", "texan al")),
    ("anus", ("anus",), ("an us", "tetanus", "uranus", "janus", "manus")),
    ("arabush", ("arab[b]ush",), ()),
    ("arse", ("|ars[s]e",), ("arsen",)),
    ("ass",
     ("|ass",),
     ("45s", "assa", "assem", "assen", "asser", "assess", "asset", "assev", "assi", "assoc",
      "assoi", "assu")),
    ("bastard", ("bas[s]tard",), ()),
    ("bestiality", ("be[e][a]s[s]tial",), ()),
    ("bitch", ("bitch", "bich|"), ()),
    ("blowjob", ("b[b]l[l][o]wj[o]b",), ()),
    ("bollocks", ("bol[l]ock",), ()),
    ("boob", ("boob",), ()),
    ("boonga", ("boonga",), ("baboon ga",)),
    ("buttplug", ("buttplug",), ()),
    ("chingchong", ("chingchong",), ()),
    ("chink", ("chink",), ("chin k",)),
    ("cock", ("|cock|", "|cocks", "|cockp", "|cocke[e]|"), ("cockney",)),
    ("cuck", ("cuck",), ("cuckoo",)),
    ("cum", ("|cum",), ("cumu", "cumb")),
    ("cunt", ("|cunt", "cunt|"), ()),
    ("deepthroat", ("deepthro[o]at", "deepthro[o]t"), ()),
    ("dick", ("|dck|", "dick"), ("benedick", "dickens", "dickety", "dickory")),
    ("dildo", ("dildo",), ()),
    ("doggystyle", ("d[o]g[g]ys[s]t[y]l[l]",), ()),
    ("double penetration", ("double penetra",), ()),
    ("dyke", ("dyke",), ("van dyke",)),
    ("ejaculate", ("e[e]jacul", "e[e]jakul", "e[e]acul[l]ate"), ()),
    ("fag", ("|fag", "fggot"), ()),
    ("felch", ("fe[e]l[l]ch",), ()),
    ("fellatio", ("f[e][e]llat",), ()),
    ("finger bang", ("fingerbang",), ()),
    ("fisting", ("fistin",), ()),
    ("fuck", ("f[?]ck", "|fk", "|fu|", "|fuk"), ("fick", "kung-fu", "kung fu")),
    ("gangbang", ("g[?]ngbang",), ()),
    ("handjob", ("h[?]ndjob",), ()),
    ("hentai", ("h[e][e]ntai",), ()),
    ("hooker", ("hooker",), ()),
    ("incest", ("incest",), ()),
    ("jerk off", ("jerkoff",), ()),
    ("jizz", ("jizz",), ()),
    ("kike", ("kike",), ()),
    ("lubejob", ("lubejob",), ()),
    ("masturbate", ("m[?]sturbate", "masterbate"), ()),
    ("negro", ("negro",), ("montenegro", "negron", "stoneground", "winegrow")),
    ("nigger", ("n[i]gger", "n[i]gga", "|nig|", "|nigs|"), ("snigger",)),
    ("orgasm", ("[or]gasm",), ("gasma",)),
    ("orgy", ("orgy", "orgies"), ("porgy",)),
    ("penis", ("pe[e]nis", "|pnis"), ("pen is",)),
    ("piss", ("|piss",), ()),
    ("porn", ("|prn|", "porn"), ("p orna",)),
    ("prick", ("|prick[s]|",), ()),
    ("pussy", ("p[u]ssy",), ()),
    ("rape", ("|rape", "|rapis[s]t"), ("rapper",)),
    ("retard", ("retard",), ()),
    ("scat", ("|s[s]cat|",), ()),
    ("semen", ("|s[s]e[e]me[e]n",), ()),
    ("sex", ("|s[s]e[e]x|", "|s[s]e[e]xy|"), ()),
    ("shit", ("|shit", "shit|"), ("s hit", "sh it", "shi t", "shitake")),
    ("slut", ("s[s]lut",), ()),
    ("spastic", ("|spastic",), ()),
    ("tit", ("|tit|", "|tits|", "|titt", "|tiddies", "|tities"), ()),
    ("tranny", ("|trany",), ()),
    ("turd", ("|turd",), ("turducken",)),
    ("twat", ("|twat",), ("twattle",)),
    ("vagina", ("vagina", "|v[?]gina"), ()),
    ("wank", ("|wank",), ()),
    ("whore", ("|wh[o]re|", "|who[o]res[s]|"), ("who're",)),
)
"""``(word, patterns, whitelisted terms)`` for each of obscenity's 69 English phrases, in the
order of ``src/preset/english.ts:106-379``, pattern strings verbatim in their syntax."""

CUSS_RATED_2: tuple[str, ...] = (
    "abeed", "africoon", "alligator bait", "alligatorbait", "analannie", "arabush", "arabushs",
    "argie", "armo", "armos", "arse", "arse bandit", "arsehole", "ass", "assbagger", "assblaster",
    "assclown", "asscowboy", "asses", "assfuck", "assfucker", "asshat", "asshole", "assholes",
    "asshore", "assjockey", "asskiss", "asskisser", "assklown", "asslick", "asslicker", "asslover",
    "assman", "assmonkey", "assmunch", "assmuncher", "asspacker", "asspirate", "asspuppies",
    "assranger", "asswhore", "asswipe", "backdoorman", "badfuck", "balllicker", "barelylegal",
    "barf", "barface", "barfface", "batty boy", "bazongas", "bazooms", "beanbag", "beanbags",
    "beaner", "beaners", "beaney", "beaneys", "beatoff", "beatyourmeat", "biatch", "bigass",
    "bigbastard", "bigbutt", "bitcher", "bitchez", "bitchin", "bitching", "bitchslap", "bitchy",
    "biteme", "blowjob", "bluegum", "bluegums", "boang", "boche", "boches", "bogan", "bohunk",
    "bollick", "bollock", "bollocks", "boner", "bong", "boobies", "booby", "boody", "boong",
    "boonga", "boongas", "boongs", "boonie", "boonies", "bootlip", "bootlips", "booty", "bootycall",
    "bosche", "bosches", "boschs", "brea5t", "breastjob", "breastlover", "breastman", "buddhahead",
    "buddhaheads", "buffies", "bufter", "bufty", "bugger", "buggered", "buggery", "bule", "bules",
    "bullcrap", "bulldike", "bulldyke", "bullshit", "bum boy", "bum chum", "bum robber",
    "bumblefuck", "bumfuck", "bung", "bunga", "bungas", "bunghole", "burr head", "burr heads",
    "burrhead", "burrheads", "butchbabes", "butchdike", "butchdyke", "buttbang", "buttface",
    "buttfuck", "buttfucker", "buttfuckers", "butthead", "buttman", "buttmunch", "buttmuncher",
    "buttpirate", "buttstain", "byatch", "cacker", "camel jockey", "camel jockeys", "cameljockey",
    "cameltoe", "carpetmuncher", "carruth", "chav", "cheese eating surrender monkey",
    "cheese eating surrender monkies", "cheeseeating surrender monkey",
    "cheeseeating surrender monkies", "cheesehead", "cheeseheads", "cherrypopper", "chi chi man",
    "chickslick", "china swede", "china swedes", "chinaman", "chinamen", "chinaswede",
    "chinaswedes", "ching chong", "ching chongs", "chingchong", "chingchongs", "chink", "chinks",
    "chinky", "choad", "chode", "chonkies", "chonky", "chonkys", "christ killer", "christ killers",
    "chug", "chugs", "chunger", "chungers", "chunkies", "chunky", "chunkys", "clamdigger",
    "clamdiver", "clansman", "clansmen", "clanswoman", "clanswomen", "clogwog", "cockblock",
    "cockblocker", "cockcowboy", "cockfight", "cockhead", "cockknob", "cocklicker", "cocklover",
    "cocknob", "cockqueen", "cockrider", "cocksman", "cocksmith", "cocksmoker", "cocksucer",
    "cocksuck", "cocksucked", "cocksucker", "cocksucking", "cocktease", "cocky", "cohee", "commie",
    "coolie", "coolies", "cooly", "coon", "coon ass", "coon asses", "coonass", "coonasses",
    "coondog", "coons", "cornhole", "cracka", "crackwhore", "crap", "crapola", "crapper", "crappy",
    "crotchjockey", "crotchmonkey", "crotchrot", "cum", "cumbubble", "cumfest", "cumjockey", "cumm",
    "cummer", "cumming", "cumquat", "cumqueen", "cumshot", "cunn", "cunntt", "cunt", "cunteyed",
    "cuntfuck", "cuntfucker", "cuntlick", "cuntlicker", "cuntlicking", "cuntsucker",
    "curry muncher", "curry munchers", "currymuncher", "currymunchers", "cushi", "cushis",
    "cyberslimer", "dago", "dagos", "dahmer", "dammit", "damnit", "darkey", "darkeys", "darkie",
    "darkies", "darky", "datnigga", "deapthroat", "deepthroat", "dego", "degos", "diaper head",
    "diaper heads", "diaperhead", "diaperheads", "dickbrain", "dickforbrains", "dickhead",
    "dickless", "dicklick", "dicklicker", "dickman", "dickwad", "dickweed", "diddle", "dingleberry",
    "dink", "dinks", "dipshit", "dipstick", "dix", "dixiedike", "dixiedyke", "doggiestyle",
    "doggystyle", "dong", "doodoo", "dope", "dot head", "dot heads", "dothead", "dotheads",
    "dragqueen", "dragqween", "dripdick", "dumb", "dumbass", "dumbbitch", "dumbfuck", "dune coon",
    "dune coons", "dyefly", "easyslut", "eatballs", "eatme", "eatpussy", "eight ball",
    "eight balls", "ero", "esqua", "evl", "exkwew", "facefucker", "faeces", "fagging", "faggot",
    "fagot", "fannyfucker", "farty", "fastfuck", "fatah", "fatass", "fatfuck", "fatfucker", "fatso",
    "fckcum", "felch", "felcher", "felching", "fellatio", "feltch", "feltcher", "feltching",
    "fingerfuck", "fingerfucked", "fingerfucker", "fingerfuckers", "fingerfucking", "fister",
    "fistfuck", "fistfucked", "fistfucker", "fistfucking", "fisting", "flange", "floo", "flydie",
    "flydye", "fok", "footfuck", "footfucker", "footlicker", "footstar", "forni", "freakfuck",
    "freakyfucker", "freefuck", "fu", "fubar", "fuc", "fucck", "fuck", "fucka", "fuckable",
    "fuckbag", "fuckbook", "fuckbuddy", "fucked", "fuckedup", "fucker", "fuckers", "fuckface",
    "fuckfest", "fuckfreak", "fuckfriend", "fuckhead", "fuckher", "fuckin", "fuckina", "fucking",
    "fuckingbitch", "fuckinnuts", "fuckinright", "fuckit", "fuckknob", "fuckme", "fuckmehard",
    "fuckmonkey", "fuckoff", "fuckpig", "fucks", "fucktard", "fuckwhore", "fuckyou", "fudge packer",
    "fudgepacker", "fugly", "fuk", "fuks", "funfuck", "fuuck", "gables", "gangbang", "gangbanged",
    "gangbanger", "gangsta", "gator bait", "gatorbait", "gaymuthafuckinwhore", "gaysex", "geez",
    "geezer", "geni", "getiton", "ginzo", "ginzos", "gipp", "gippo", "gippos", "gipps", "givehead",
    "glazeddonut", "godammit", "goddamit", "goddammit", "goddamn", "goddamned", "goddamnes",
    "goddamnit", "goddamnmuthafucker", "goldenshower", "golliwog", "golliwogs", "gonorrehea",
    "gook", "gook eye", "gook eyes", "gookeye", "gookeyes", "gookies", "gooks", "gooky", "gora",
    "goras", "gotohell", "greaseball", "greaseballs", "greaser", "greasers", "gringo", "gringos",
    "groid", "groids", "gubba", "gubbas", "gubs", "gummer", "gwailo", "gwailos", "gweilo",
    "gweilos", "gyopo", "gyopos", "gyp", "gyped", "gypo", "gypos", "gypp", "gypped", "gyppie",
    "gyppies", "gyppo", "gyppos", "gyppy", "gyppys", "gypsies", "gypsy", "gypsys", "hadji",
    "hadjis", "hairyback", "hairybacks", "haji", "hajis", "hajji", "hajjis", "half breed",
    "half caste", "halfbreed", "halfcaste", "handjob", "haole", "haoles", "hapa", "hardon",
    "headfuck", "hebe", "hebes", "heeb", "heebs", "hillbillies", "hillbilly", "hindoo", "hiscock",
    "hitlerism", "hitlerist", "ho", "hobo", "hodgie", "hoes", "holestuffer", "homo", "homobangers",
    "honger", "honkers", "honkey", "honkeys", "honkie", "honkies", "honky", "hooker", "hookers",
    "hooters", "hore", "hori", "horis", "hork", "horney", "horniest", "horseshit", "hosejob",
    "hoser", "hotdamn", "hotpussy", "hottotrot", "hussy", "hymie", "hymies", "iblowu", "idiot",
    "ikeymo", "ikeymos", "ikwe", "indon", "indons", "injun", "injuns", "insest", "intheass",
    "inthebuff", "jackass", "jackoff", "jackshit", "jacktheripper", "jap", "japcrap", "japie",
    "japies", "japs", "jebus", "jeez", "jerkoff", "jewboy", "jewed", "jewess", "jig", "jiga",
    "jigaboo", "jigaboos", "jigarooni", "jigaroonis", "jigg", "jigga", "jiggabo", "jiggabos",
    "jiggas", "jigger", "jiggers", "jiggs", "jiggy", "jigs", "jijjiboo", "jijjiboos", "jimfish",
    "jism", "jiz", "jizim", "jizjuice", "jizm", "jizz", "jizzim", "jizzum", "juggalo",
    "jungle bunnies", "jungle bunny", "junglebunny", "kacap", "kacapas", "kacaps", "kaffer",
    "kaffir", "kaffre", "kafir", "kanake", "katsap", "katsaps", "khokhol", "khokhols", "kigger",
    "kike", "kikes", "kimchis", "kissass", "kkk", "klansman", "klansmen", "klanswoman",
    "klanswomen", "kondum", "koon", "krap", "krappy", "krauts", "kuffar", "kum", "kumbubble",
    "kumbullbe", "kummer", "kumming", "kumquat", "kums", "kunilingus", "kunnilingus", "kunt",
    "kushi", "kushis", "kwa", "kwai lo", "kwai los", "kyke", "kykes", "kyopo", "kyopos", "lebo",
    "lebos", "lesbain", "lesbayn", "lesbin", "lesbo", "lez", "lezbe", "lezbefriends", "lezbo",
    "lezz", "lezzo", "lickme", "limey", "limpdick", "limy", "livesex", "loadedgun", "looser",
    "loser", "lovebone", "lovegoo", "lovegun", "lovejuice", "lovemuscle", "lovepistol",
    "loverocket", "lowlife", "lubejob", "lubra", "luckycammeltoe", "lugan", "lugans", "mabuno",
    "mabunos", "macaca", "macacas", "magicwand", "mahbuno", "mahbunos", "mams", "manhater",
    "manpaste", "mastabate", "mastabater", "masterbate", "masterblaster", "mastrabator",
    "masturbate", "masturbating", "mattressprincess", "mau mau", "mau maus", "maumau", "maumaus",
    "meatbeatter", "meatrack", "mgger", "mggor", "mickeyfinn", "milf", "mockey", "mockie", "mocky",
    "mofo", "moky", "moneyshot", "moon cricket", "moon crickets", "mooncricket", "mooncrickets",
    "moron", "moskal", "moskals", "moslem", "mosshead", "mothafuck", "mothafucka", "mothafuckaz",
    "mothafucked", "mothafucker", "mothafuckin", "mothafucking", "mothafuckings", "motherfuck",
    "motherfucked", "motherfucker", "motherfuckin", "motherfucking", "motherfuckings",
    "motherlovebone", "muff", "muffdive", "muffdiver", "muffindiver", "mufflikcer", "mulatto",
    "muncher", "munt", "mzungu", "mzungus", "nastybitch", "nastyho", "nastyslut", "nastywhore",
    "negres", "negress", "negro", "negroes", "negroid", "negros", "nig", "nigar", "nigars",
    "nigers", "nigette", "nigettes", "nigg", "nigga", "niggah", "niggahs", "niggar", "niggaracci",
    "niggard", "niggarded", "niggarding", "niggardliness", "niggardlinesss", "niggards", "niggars",
    "niggas", "niggaz", "nigger", "niggerhead", "niggerhole", "niggers", "niggle", "niggled",
    "niggles", "niggling", "nigglings", "niggor", "niggress", "niggresses", "nigguh", "nigguhs",
    "niggur", "niggurs", "niglet", "nignog", "nigor", "nigors", "nigr", "nigra", "nigras", "nigre",
    "nigres", "nigress", "nigs", "nip", "nittit", "nlgger", "nlggor", "nofuckingway", "nookey",
    "nookie", "noonan", "nudger", "nutfucker", "ontherag", "orga", "orgasim", "paki", "pakis",
    "palesimian", "pancake face", "pancake faces", "pansies", "pansy", "panti", "payo",
    "peckerwood", "pedo", "peehole", "peepee", "peepshpw", "peni5", "perv", "phuk", "phuked",
    "phuking", "phukked", "phukking", "phungky", "phuq", "pi55", "picaninny", "piccaninny",
    "pickaninnies", "pickaninny", "piefke", "piefkes", "piker", "pikey", "piky", "pillow biter",
    "pimp", "pimped", "pimper", "pimpjuic", "pimpjuice", "pimpsimp", "pindick", "piss", "pissed",
    "pisser", "pisses", "pisshead", "pissin", "pissing", "pissoff", "pocha", "pochas", "pocho",
    "pochos", "pocketpool", "pohm", "pohms", "polack", "polacks", "pollock", "pollocks", "pom",
    "pommie", "pommie grant", "pommie grants", "pommies", "pommy", "poms", "poo", "poof", "poofta",
    "poofter", "poon", "poontang", "poop", "pooper", "pooperscooper", "pooping", "poorwhitetrash",
    "popimp", "porch monkey", "porch monkies", "porchmonkey", "pornking", "pornprincess",
    "prairie nigger", "prairie niggers", "pric", "prick", "prickhead", "pu55i", "pu55y",
    "pubiclice", "pud", "pudboy", "pudd", "puddboy", "puke", "puntang", "purinapricness", "puss",
    "pussie", "pussies", "pussyeater", "pussyfucker", "pussylicker", "pussylips", "pussylover",
    "pussypounder", "pusy", "quashie", "queef", "quickie", "quim", "ra8s", "raghead", "ragheads",
    "raper", "rearend", "rearentry", "redleg", "redlegs", "redneck", "rednecks", "redskin",
    "redskins", "reefer", "reestie", "rentafuck", "rere", "retard", "retarded", "rigger", "rimjob",
    "rimming", "round eyes", "roundeye", "russki", "russkie", "sadis", "sadom", "sambo", "sambos",
    "samckdaddy", "sand nigger", "sand niggers", "sandm", "sandnigger", "scallywag", "schlong",
    "schvartse", "schvartsen", "schwartze", "schwartzen", "screwyou", "seppo", "seppos", "sexed",
    "sexfarm", "sexhound", "sexing", "sexkitten", "sexpot", "sexslave", "sextogo", "sexwhore",
    "sexymoma", "sexyslim", "shaggin", "shagging", "shat", "shav", "shawtypimp", "sheeney", "shhit",
    "shiksa", "shitcan", "shitdick", "shite", "shiteater", "shited", "shitface", "shitfaced",
    "shitfit", "shitforbrains", "shitfuck", "shitfucker", "shitfull", "shithapens", "shithappens",
    "shithead", "shithouse", "shiting", "shitlist", "shitola", "shitoutofluck", "shits",
    "shitstain", "shitted", "shitter", "shitting", "shitty", "shortfuck", "shylock", "shylocks",
    "sissy", "sixsixsix", "sixtynine", "sixtyniner", "skank", "skankbitch", "skankfuck",
    "skankwhore", "skanky", "skankybitch", "skankywhore", "skinflute", "skum", "skumbag", "skwa",
    "skwe", "slanteye", "slanty", "slapper", "slave", "slavedriver", "sleezebag", "sleezeball",
    "slideitin", "slimeball", "slimebucket", "slopehead", "slopeheads", "sloper", "slopers",
    "slopey", "slopeys", "slopies", "slopy", "slut", "sluts", "slutt", "slutting", "slutty",
    "slutwear", "slutwhore", "smackthemonkey", "smut", "snatchpatch", "snowback", "snownigger",
    "sodomise", "sodomize", "sodomy", "sonofabitch", "sonofbitch", "sooties", "sooty",
    "spaghettibender", "spaghettinigger", "spankthemonkey", "spearchucker", "spearchuckers",
    "spermacide", "spermbag", "spermhearder", "spermherder", "spic", "spick", "spicks", "spics",
    "spig", "spigotty", "spik", "spit", "spitter", "splittail", "spooge", "spreadeagle", "spunk",
    "spunky", "sqeh", "squa", "squarehead", "squareheads", "squaw", "squinty", "stringer",
    "stripclub", "stuinties", "stupid", "stupidfuck", "stupidfucker", "suckdick", "sucker",
    "suckme", "suckmyass", "suckmydick", "suckmytit", "suckoff", "swallower", "swalow",
    "swamp guinea", "swamp guineas", "tacohead", "tacoheads", "taff", "tang", "tar babies",
    "tar baby", "tarbaby", "tard", "teste", "thicklip", "thicklips", "thirdeye", "thirdleg",
    "threeway", "timber nigger", "timber niggers", "timbernigger", "tinker", "tinkers",
    "titbitnipply", "titfuck", "titfucker", "titfuckin", "titjob", "titlicker", "titlover",
    "tittie", "titties", "titty", "tongethruster", "tonguethrust", "tonguetramp", "tortur",
    "tosser", "towel head", "towel heads", "towelhead", "trailertrash", "trannie", "tranny",
    "transvestite", "triplex", "tuckahoe", "tunneloflove", "turnon", "twat", "twink", "twinkie",
    "twobitwhore", "uck", "ukrop", "uncle tom", "unfuckable", "upskirt", "uptheass", "upthebutt",
    "usama", "vibr", "vibrater", "virginbreaker", "vomit", "wab", "wank", "wanker", "wanking",
    "waysted", "weenie", "weewee", "welcher", "welfare", "wetb", "wetback", "wetbacks", "wetspot",
    "whacker", "whash", "whigger", "whiggers", "whiskeydick", "whiskydick", "white trash",
    "whitenigger", "whitetrash", "whitey", "whiteys", "whities", "whiz", "whop", "whore",
    "whorefucker", "whorehouse", "wigga", "wiggas", "wigger", "wiggers", "willie", "williewanker",
    "wn", "wog", "wogs", "wop", "wtf", "wuss", "wuzzie", "xkwe", "yank", "yanks", "yarpie",
    "yarpies", "yellowman", "yid", "yids", "zigabo", "zigabos", "zipperhead", "zipperheads",
)
"""Every entry of ``words/cuss`` ``index.js`` rated 2 ("likely" a profanity or slur), sorted,
verbatim. Copyright (c) 2016 Titus Wormer, MIT — licence text in ``licenses/cuss-MIT.txt``."""

THREAT_PATTERNS: tuple[tuple[str, str], ...] = (
    ("kill_yourself", r"\bk+y+s+\b|\bkill (?:yo)?ur ?self\b"),
    ("wish_death", r"\bgo (?:and )?die\b|\bhope (?:you|u|they|she|he) (?:die|dies|get raped)\b"),
    ("call_for_violence",
     r"\b(?:should|ought to|needs? to|must|deserves? to|gonna|going to) (?:all )?(?:be |get )?"
     r"(?:shot|hanged|lynched|gassed|raped|exterminated)\b"),
    ("target_for_violence",
     r"\b(?:kill|shoot|hang|lynch|exterminate) (?:them all|all of them|these people|"
     r"those people)\b"),
    ("dehumanising",
     r"\b(?:they|these|those|them|you|u) (?:people |guys |ppl )?(?:are|r) "
     r"(?:all |just |nothing but )?(?:animals|vermin|rats|cockroaches|parasites|savages|"
     r"subhuman|scum|filth|pigs)\b"),
    ("expulsion",
     r"\bgo back to (?:your|ur|their|his|her) (?:own )?(?:country|countries|homeland|shithole)\b"),
    ("abusive_acronym", r"\b(?:stfu|gtfo|stfo)\b"),
)
"""ARGUS's own patterns, matched case-insensitively on whitespace-normalised text. Kept narrow on
purpose: "executed", "killed" and "beaten" were in the first draft and were taken out, because on
a trading desk an order *should be executed*, shorts *are going to get killed* and an estimate
*should be beaten*."""

MARKET_ALLOWLIST: tuple[str, ...] = (
    "gainers and losers", "gainers & losers", "gainers/losers", "winners and losers",
    "top losers", "biggest losers", "dumb money", "sucker rally", "sucker's rally",
    "suckers rally", "suckers' rally", "shitcoin", "shit coin", "rapeseed",
    "dick's sporting goods", "dicks sporting goods", "cockpit", "shiitake", "dumb it down",
    "dumb down", "dumbed down", "dumbing down",
)
"""Ordinary phrases in market talk that contain a listed word. Matched on lower-cased text; a hit
lying wholly inside one is ignored (plurals such as "shitcoins" are covered because the phrase is
a prefix of them). "Top ETF gainers & losers" and "let's dumb it down" were both withheld on the
desk's own snapshots of real X and Reddit posts on 2026-09-25 before these entries existed."""

_TYPOGRAPHIC_APOSTROPHE = chr(0x2019)

CUSS_RATED_2_SHA256 = "aa5b29a8161a2f23a5b92d50b33eba6d862ed6d615ba9efd19f1dcf1a5a982ab"
"""SHA-256 of :data:`CUSS_RATED_2` joined with newlines — changes if the copied list is edited."""


def _fold(ch: str) -> str:
    """NFKD compatibility decomposition with combining marks dropped: a full-width letter becomes
    its ASCII form and an accented one loses the accent."""
    decomposed = unicodedata.normalize("NFKD", ch)
    return "".join(c for c in decomposed if not unicodedata.combining(c)) or ch


def blacklist_form(text: str) -> tuple[str, tuple[int, ...]]:
    """The text as the matchers see it, and each character's index in the original.

    Fold, resolve leetspeak, lower-case ASCII, collapse repeats — obscenity's recommended chain.
    """
    chars: list[str] = []
    where: list[int] = []
    last = ""
    run = 0
    for position, original in enumerate(text):
        for folded in _fold(original):
            char = _LEET.get(folded, folded)
            if char.isascii():
                char = char.lower()
            if char == last:
                run += 1
                if run >= _COLLAPSE_LIMIT.get(char, 1):
                    continue
            else:
                last = char
                run = 0
            chars.append(char)
            where.append(position)
    return "".join(chars), tuple(where)


def _whitelist_form(text: str) -> tuple[str, tuple[int, ...]]:
    """Lower-case ASCII, runs of spaces collapsed to one — obscenity's whitelist chain
    (``src/preset/english.ts:36-42``) — plus one change of ours: the typographic apostrophe
    (U+2019) becomes ``'``, so "sucker's rally" typed on a phone still matches the term written
    with a plain apostrophe."""
    chars: list[str] = []
    where: list[int] = []
    for position, char in enumerate(text.replace(_TYPOGRAPHIC_APOSTROPHE, "'")):
        if char == " " and chars and chars[-1] == " ":
            continue
        chars.append(char.lower() if char.isascii() else char)
        where.append(position)
    return "".join(chars), tuple(where)


def _occurrences(form: str, where: tuple[int, ...], terms: Sequence[str]) -> list[tuple[int, int]]:
    """Original-text spans of every non-overlapping occurrence of each term, as obscenity's
    ``getWhitelistedIntervals`` walks them (``RegExpMatcher.ts:143-169``)."""
    spans: list[tuple[int, int]] = []
    for term in terms:
        start = form.find(term)
        while start != -1:
            spans.append((where[start], where[start + len(term) - 1]))
            start = form.find(term, start + len(term))
    return spans


def _covered(start: int, end: int, spans: Sequence[tuple[int, int]]) -> bool:
    """Obscenity's rule: a match is excused only if one span wholly contains it."""
    return any(low <= start and end <= high for low, high in spans)


def _compile(pattern: str) -> re.Pattern[str]:
    """Obscenity's pattern syntax to a Python regex (``src/pattern/Util.ts:8-37``)."""
    at_start = pattern.startswith("|")
    at_end = len(pattern) > 1 and pattern.endswith("|")
    core = pattern[1 if at_start else 0: -1 if at_end else None]
    body = ""
    index = 0
    while index < len(core):
        char = core[index]
        if char == "[":
            close = core.index("]", index)
            inner = "".join("." if c == "?" else re.escape(c) for c in core[index + 1:close])
            body += f"(?:{inner})?"
            index = close + 1
            continue
        body += "." if char == "?" else re.escape(char)
        index += 1
    return re.compile(("\\b" if at_start else "") + body + ("\\b" if at_end else ""),
                      re.ASCII | re.DOTALL)


@dataclass(frozen=True, slots=True)
class Verdict:
    """What the screen found. ``terms`` and ``threats`` name lexicon entries and pattern names —
    never the post's own text, so a verdict can be logged without repeating the abuse."""

    terms: tuple[str, ...] = ()
    threats: tuple[str, ...] = ()

    @property
    def abusive(self) -> bool:
        return bool(self.terms or self.threats)

    @property
    def reason(self) -> str | None:
        if self.threats:
            return REASON_THREAT
        return REASON_TERM if self.terms else None


@dataclass(frozen=True)
class _Lexicon:
    obscenity: tuple[tuple[str, tuple[re.Pattern[str], ...], tuple[str, ...]], ...]
    singles: frozenset[str]
    """Cuss entries after :func:`blacklist_form`, at least three characters long."""
    raw: frozenset[str]
    """Cuss entries whose transformed form is shorter than three characters ("kkk" collapses to
    "k"), matched only as the exact raw token so a lone letter never fires. The three two-letter
    entries ("fu", "ho", "wn") are not matched at all: a two-letter token is too often something
    else ("ho ho ho", "kung fu"), and obscenity's own ``|fu|`` pattern, with its "kung fu"
    whitelist, still covers the one that matters."""
    phrases: tuple[tuple[str, ...], ...]
    masked_targets: tuple[str, ...]
    """What a masked word may stand for, most specific first: words both lists carry, then the
    rest of obscenity's curated set, then the rest of cuss — so ``f***`` reads as the word
    everyone means rather than the first four-letter entry in alphabetical order."""
    whitelist: tuple[str, ...]
    """Every obscenity whitelisted term. Cuss hits are excused by them too: "45s" is a record
    format whichever list the ``ass`` hit came from."""
    threats: tuple[tuple[str, re.Pattern[str]], ...]


def _build() -> _Lexicon:
    singles: set[str] = set()
    raw: set[str] = set()
    phrases: list[tuple[str, ...]] = []
    for entry in CUSS_RATED_2:
        form = blacklist_form(entry)[0]
        if " " in form:
            phrases.append(tuple(form.split()))
        elif len(form) >= 3:
            singles.add(form)
        elif len(entry) >= 3:
            raw.add(entry.lower())
    obscenity = tuple((word, tuple(_compile(p) for p in patterns), whitelist)
                      for word, patterns, whitelist in OBSCENITY)
    core = [word for word, _, _ in OBSCENITY if " " not in word]
    targets = ([w for w in core if w in singles] + [w for w in core if w not in singles]
               + sorted(singles.difference(core)))
    whitelist = tuple(dict.fromkeys(term for _, _, terms in OBSCENITY for term in terms))
    threats = tuple((name, re.compile(rx, re.IGNORECASE)) for name, rx in THREAT_PATTERNS)
    return _Lexicon(obscenity=obscenity, singles=frozenset(singles), raw=frozenset(raw),
                    phrases=tuple(phrases), masked_targets=tuple(targets), whitelist=whitelist,
                    threats=threats)


_TOKEN = re.compile(r"[a-z]+")
_RAW_TOKEN = re.compile(r"[a-z]+", re.IGNORECASE)
_MASKED = re.compile(r"[a-z*]*[a-z][a-z*]*", re.IGNORECASE)
_SPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class AbuseScreen:
    """The screen, with each matcher switchable so its contribution can be measured.

    The production screen is :data:`DEFAULT_SCREEN` — every matcher on. The switches exist for
    `eval/sentiment_tweeteval.py`'s ablation and for nothing else.
    """

    obscenity: bool = True
    cuss: bool = True
    threats: bool = True
    allowlist: bool = True
    _lexicon: _Lexicon = field(default_factory=_build, repr=False, compare=False)

    @property
    def name(self) -> str:
        parts = [part for part, on in (("obscenity", self.obscenity), ("cuss2", self.cuss),
                                       ("threats", self.threats)) if on]
        return "+".join(parts) + ("" if self.allowlist else " (no allowlist)")

    def __call__(self, text: str) -> Verdict:
        form, where = blacklist_form(text)
        white = _whitelist_form(text)
        allowed = _occurrences(*white, MARKET_ALLOWLIST) if self.allowlist else []
        terms: list[str] = []
        if self.obscenity:
            terms.extend(self._obscenity_hits(white, form, where, allowed))
        if self.cuss:
            excused = allowed + _occurrences(*white, self._lexicon.whitelist)
            terms.extend(self._cuss_hits(text, form, where, excused))
        if (self.obscenity or self.cuss) and "*" in text:
            terms.extend(self._masked_hits(text, allowed))
        threats: list[str] = []
        if self.threats:
            flat = _SPACE.sub(" ", text)
            threats = [name for name, rx in self._lexicon.threats if rx.search(flat)]
        return Verdict(terms=tuple(dict.fromkeys(terms)), threats=tuple(threats))

    def _obscenity_hits(self, white: tuple[str, tuple[int, ...]], form: str,
                        where: tuple[int, ...], allowed: list[tuple[int, int]]) -> Iterator[str]:
        for word, patterns, whitelist in self._lexicon.obscenity:
            excused = allowed + _occurrences(*white, whitelist) if whitelist else allowed
            if any(not _covered(where[m.start()], where[m.end() - 1], excused)
                   for rx in patterns for m in rx.finditer(form)):
                yield word

    def _cuss_hits(self, text: str, form: str, where: tuple[int, ...],
                   allowed: list[tuple[int, int]]) -> Iterator[str]:
        lexicon = self._lexicon
        tokens = [(m.group(), where[m.start()], where[m.end() - 1])
                  for m in _TOKEN.finditer(form)]
        for token, start, end in tokens:
            if _covered(start, end, allowed):
                continue
            if token in lexicon.singles:
                yield token
            elif len(token) >= 5 and token.endswith("s") and token[:-1] in lexicon.singles:
                yield token[:-1]
        words = [token for token, _, _ in tokens]
        for phrase in lexicon.phrases:
            size = len(phrase)
            for index in range(len(words) - size + 1):
                if tuple(words[index:index + size]) == phrase and not _covered(
                        tokens[index][1], tokens[index + size - 1][2], allowed):
                    yield " ".join(phrase)
                    break
        for match in _RAW_TOKEN.finditer(text):
            token = match.group().lower()
            if token in lexicon.raw and not _covered(match.start(), match.end() - 1, allowed):
                yield token

    def _masked_hits(self, text: str, allowed: list[tuple[int, int]]) -> Iterator[str]:
        """``f*ck``, ``sh*t``, ``a**``: each run of ``*`` stands for that many letters. The token
        must start with a letter — masking keeps the first one — and be three characters or more,
        so markdown emphasis (``*very*``, ``**BTC**``) is never read as a mask."""
        for match in _MASKED.finditer(text):
            token = match.group().lower()
            if ("*" not in token or len(token) < 3 or not token[0].isalpha()
                    or _covered(match.start(), match.end() - 1, allowed)):
                continue
            rx = re.compile(re.sub(r"\*+", lambda run: f"[a-z]{{{len(run.group())}}}", token))
            hit = next((word for word in self._lexicon.masked_targets if rx.fullmatch(word)),
                       None)
            if hit is not None:
                yield hit


DEFAULT_SCREEN = AbuseScreen()


def screen(text: str) -> Verdict:
    """The production verdict for one piece of text."""
    return DEFAULT_SCREEN(text)


_CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z])|[_\-.\d]+")


def handle_words(handle: str) -> str:
    """An account name split into words: ``@fuck_the_fed`` and ``@FuckTheFed`` both become
    "fuck the fed". Handles run words together, so the screen's word-boundary patterns would
    miss them unsplit."""
    bare = handle.removeprefix("@").removeprefix("u/")
    return " ".join(part for part in _CAMEL.split(bare) if part)


def handle_is_abusive(handle: str, check: Callable[[str], Verdict] = screen) -> bool:
    """Whether an account name reads as abusive once split into words."""
    return check(handle_words(handle)).abusive


def lexicon_digest(entries: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()


__all__ = [
    "CUSS_RATED_2",
    "CUSS_RATED_2_SHA256",
    "DEFAULT_SCREEN",
    "MARKET_ALLOWLIST",
    "OBSCENITY",
    "REASON_TERM",
    "REASON_THREAT",
    "THREAT_PATTERNS",
    "AbuseScreen",
    "Verdict",
    "blacklist_form",
    "handle_is_abusive",
    "handle_words",
    "lexicon_digest",
    "screen",
]
