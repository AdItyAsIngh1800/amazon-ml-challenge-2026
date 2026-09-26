"""Normalisation dictionaries (plan Stage 0). Every entry traces to EDA evidence.

Sources: E7 / E10 / E11 in ``artifacts/eda/eda_report.txt`` (full data), plus
the dev-sample token counts quoted in comments (name_norm after
transliteration: the last token / last two tokens of each name). Keys are
normalised tokens (lowercase ASCII, as in ``name_norm`` / ``addr_norm``).
All dictionaries apply to every record; none is conditional on country.
"""

from __future__ import annotations

# ---------------------------------------------------------------- names

# Legal-form token -> canonical form. Used only inside the trailing (or, if
# there is none, leading) run of legal tokens of a name, so words such as
# "public" or "co" elsewhere in a name are never touched.
LEGAL_FORMS: dict[str, str] = {
    # India, E10 top name tokens: limited 1.76M, private 1.64M, ltd 0.80M, pvt 0.53M, llp 0.18M
    # (dict order = order of forms inside legal_suffix: "private limited", "public limited")
    "private": "private", "pvt": "private", "public": "public",
    "limited": "limited", "ltd": "limited", "llp": "llp",
    # transliterated Indic spellings (dev sample, E10 लिमिटेड / प्राइवेट / எல்எல்பி ...)
    "praaivet": "private",  # 27,612 (Devanagari प्राइवेट)
    "praivet": "private",  # 8,751 (Telugu/Kannada)
    "praaibhet": "private",  # 4,432 (Bengali/Oriya প্রাইভেট)
    "piraivet": "private",  # 3,918 (Tamil பிரைவேட்)
    "praivatt": "private",  # 2,231 (Malayalam പ്രൈവറ്റ്)
    "praaeevet": "private",  # 593 (Gurmukhi ਪ੍ਰਾਈਵੇਟ)
    "limitet": "limited",  # 4,476 (Tamil லிமிடெட்)
    "limittad": "limited",  # 2,537 (Malayalam ലിമിറ്റഡ്)
    "limatid": "limited",  # 741 (Gurmukhi ਲਿਮਟਿਡ)
    "elaelapee": "llp",  # 2,208 (Devanagari एलएलपी)
    "elelpi": "llp",  # 757 (Kannada/Telugu)
    # US, E10: llc 1.46M, inc 1.09M, corp 0.28M, ltd 0.18M, lp 0.11M, pc 0.10M,
    # corporation 90k, pllc 71k, incorporated 49k; E11 "Amber Loan, Inc." / "AMBER LOAN,"
    "llc": "llc", "inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp",
    "lp": "lp", "pc": "pc", "pllc": "pllc", "pa": "pa",  # "p a" 580 (P.A.)
    "co": "co", "company": "co",  # India co 110k, company 49k; "and co" 2,873
    # France (test only), E10: sarl 372k, sas 253k, eurl 101k, sa 82k, sasu 74k,
    # sci 64k, cie 42k, ei 18.5k, snc 14.8k
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "sa": "sa",
    "sci": "sci", "snc": "snc", "ei": "ei", "cie": "cie",
}

# Multi-word legal forms rewritten before tokenising (regex on name_norm).
LEGAL_PHRASES: tuple[tuple[str, str], ...] = (
    (r"(?:^| )praa li(?= |$)", " private limited"),  # 6,340: Gujarati પ્રા. લિ. = Pvt. Ltd.
    (r"(?:^| )(?:and|et) (co|cie)(?= |$)", r" \1"),  # "and co" 2,873; French "et cie"
)

# Words injected after a legal form (E11 "Tvh Exports (India) Care",
# "Bansal LLP Center"; dev last-two tokens "limited center" 6,422, "llc center"
# 4,450, "inc services" 2,150). Allowed inside the trailing legal run so the
# suffix is still found; kept in name_core, except "com" (from domain-style
# names, E11 "héartcenter.com"; last token com 17k India / 32k US), which is dropped.
SUFFIX_NOISE: frozenset[str] = frozenset({"center", "centre", "services", "service", "partners", "care", "com"})
DROP_IN_SUFFIX: frozenset[str] = frozenset({"com"})

# Honorific prefixes, dropped when leading. E10 India: dr 55k, shri/sri 55k,
# mr 55k, smt 55k (near-identical counts = injected noise); E11 "Smt Pamtl
# Holdings", "Dr Classic LLP Developers", "Mr Malhotra (Índia) World Ltd".
# "ms" also covers "M/s" (Messrs), which normalises to "m s" -> "ms".
HONORIFICS: frozenset[str] = frozenset({"mr", "mrs", "ms", "dr", "smt"})

# Other name-token spellings -> one form (E10 India: sri 55k, shri 55k, shree 34k).
NAME_ABBREVIATIONS: dict[str, str] = {"shri": "sri", "shree": "sri"}

# ---------------------------------------------------------------- addresses

# Address token -> short canonical form. Long -> short so no short token
# gets two meanings (US "st" street and French "st" saint both stay "st").
ADDR_ABBREVIATIONS: dict[str, str] = {
    # street types, E10 US: street 840k / st 651k, road 768k / rd 604k, drive 676k / dr 548k,
    # avenue 558k / ave 451k, lane 290k / ln 232k, court 181k / ct 201k, circle 88k / cir 67k,
    # place 84k / pl 63k, boulevard 67k / blvd 52k; E11 "2652 50th Street" / "50th St"
    "street": "st", "road": "rd", "drive": "dr", "avenue": "ave", "lane": "ln", "court": "ct",
    "circle": "cir", "place": "pl", "boulevard": "blvd", "apartment": "apt",
    "township": "twp",  # E11 "Ashtabula" / "ASHTABULA TWP"; township 98k
    "saint": "st", "sainte": "ste",  # US saint 125k; France saint 141k / st 25k
    # France, E10: rue 729k / r 357k, avenue 126k / av 57k, boulevard 38k / bd 25k, allee 52k / all 26k
    "rue": "r", "av": "ave", "bd": "blvd", "allee": "all",
    # India, E10: near 259k / nr 90k, opp 137k; E7(d) "Opp Kamakya Theatre", "Near Punjab National Bank"
    "near": "nr", "opposite": "opp", "number": "no",
    # US states: E11 true pairs mix codes and names ("OK" / "Oklahoma", "IL" / "Illinois",
    # "TX" / "Texas", "MA" / "Massachusetts", "NM" / "New Mexico"); E10 US lists both forms.
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
    "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "ohio": "oh", "oklahoma": "ok", "oregon": "or",
    "pennsylvania": "pa", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "wisconsin": "wi", "wyoming": "wy",
    # India states: E10 India lists both (maharashtra 537k / mh 325k, delhi / dl 202k,
    # tamil nadu / tn 105k, gujarat / gj 95k, west bengal / wb 95k); E11 "Telangana" / "TG",
    # "Karnataka" / "KA", "Rajasthan" / "RJ", "Madhya Pradesh" / "MP", E7(d) "Haryana" / "HR".
    # Transliterated spellings are the dev-sample Indic address parts (महाराष्ट्र 21k, ...).
    "maharashtra": "mh", "mahaaraashtra": "mh", "delhi": "dl", "dillee": "dl",
    "karnataka": "ka", "karnaatak": "ka", "gujarat": "gj", "gujaraat": "gj",
    "telangana": "tg", "telangaan": "tg", "haryana": "hr", "hariyaanaa": "hr",
    "rajasthan": "rj", "raajasthaan": "rj", "kerala": "kl", "keralam": "kl",
    "bihar": "br", "bihaar": "br", "punjab": "pb", "panjaab": "pb",
    "odisha": "od", "orissa": "od", "orishaa": "od", "tamilnaatu": "tn", "pashchimabang": "wb",
    "aandhrapradesh": "ap",
}

# Multi-word address names -> code (regex on addr_norm, applied before tokens).
ADDR_PHRASES: tuple[tuple[str, str], ...] = (
    (r"(?:^| )new york(?= |$)", " ny"), (r"(?:^| )new jersey(?= |$)", " nj"),
    (r"(?:^| )new mexico(?= |$)", " nm"), (r"(?:^| )new hampshire(?= |$)", " nh"),
    (r"(?:^| )north carolina(?= |$)", " nc"), (r"(?:^| )south carolina(?= |$)", " sc"),
    (r"(?:^| )north dakota(?= |$)", " nd"), (r"(?:^| )south dakota(?= |$)", " sd"),
    (r"(?:^| )west virginia(?= |$)", " wv"), (r"(?:^| )rhode island(?= |$)", " ri"),
    (r"(?:^| )uttar pradesh(?= |$)", " up"), (r"(?:^| )madhya pradesh(?= |$)", " mp"),
    (r"(?:^| )andhra pradesh(?= |$)", " ap"), (r"(?:^| )tamil nadu(?= |$)", " tn"),
    (r"(?:^| )west bengal(?= |$)", " wb"),
)

# Landmark words (canonical forms, after ADDR_ABBREVIATIONS). E10 India near /
# nr / opp; E7(d) "Behind R-Ma"; E11 "Above Marss Herbals".
LANDMARK_WORDS: tuple[str, ...] = ("nr", "opp", "behind", "beside", "above")
