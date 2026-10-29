"""
Name check on the proposed (Direct, Good) pairs – runs AFTER Steps 1-3.

1. Name cleaning (same for both sides)
   - leading number blocks with >= 5 digits removed ("40624139 - 09082026 APS" -> "APS",
     "123 ACME" keeps 123)
   - lowercase, accents removed, dots removed ("B.V." -> "bv"), split into words
   - legal forms removed using legal_form_mapping.csv (regex pre-expanded to word sequences):
       * longer than 3 letters (counted without spaces/punctuation)  -> removed anywhere
       * 3 letters or fewer, or PRIVATE -> follow the CSV anchor:
             ^ only -> start, $ only -> end, both -> start or end, none -> anywhere
     removed forms are kept as LegalForms (CSV rows merged when equal without dots/spaces)
   - stopwords removed
   - legal-form + stopword removal run twice ("AG in Liquidation")
   - typo fix on the LAST word only: 1 edit away from a one-word legal form,
     word >= 6 letters, same first letter, not in PROTECT_WORDS
   - if nothing is left, the original letters are kept
2. Features: Compact (no spaces), Numbers (digits, Roman II-XX, number words 2-20 / 1 where
   unambiguous, EN/DE/FR/IT/ES/PT/NL) – read after legal-form removal
3. MatchScore = max(fuzz.ratio(Compact), token_sort_ratio(clean), JaroWinkler*100 (Compact))
4. Pass = MatchScore >= THRESHOLD  AND  Numbers equal  AND
          (one/both sides without legal form  OR  at least one shared legal form group)
"""
import re
from collections import defaultdict

import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from unidecode import unidecode

# ─── WORD LISTS ──────────────────────────────────────────────────────────────

STOPWORDS = {
    "the", "of", "and", "&", "for", "an",                        # EN
    "der", "die", "das", "und", "fur", "von", "des", "dem", "den",  # DE
    "le", "la", "les", "de", "du", "et", "aux", "au",            # FR
    "het", "en", "van", "voor",                                  # NL
    "el", "los", "las", "del",                                   # ES
    "il", "lo", "gli", "di", "della", "dei", "delle",            # IT
    "os", "da", "do", "dos", "das",                              # PT
    # status words, so "AG in Liquidation" -> "AG" -> removed on the 2nd pass
    "in", "liquidation", "liquidazione", "liquidacion", "liquidatie", "liquidacao",
}

# Legal-form groups treated as the SAME (group names = CSV LegalForm without spaces/dots).
# Everything not listed stays strict (CORP != INC, PTE LTD != LTD, CO LTD != LTD).
LEGAL_EQUIV = [
    {"COMPANY", "CORP"},       # CO. = CORP.
    {"PTYLTD", "LIMITED"},     # PTY LTD = LTD
]

PROTECT_WORDS = {
    "parts", "partner", "partners", "operation", "operations", "corporate", "corporates",
    "limits", "limiter", "limiting", "limitless", "general", "generali",
}

_ROMAN = {"ii": 2, "iii": 3, "iv": 4, "vi": 6, "vii": 7, "viii": 8, "ix": 9, "xi": 11,
          "xii": 12, "xiii": 13, "xiv": 14, "xv": 15, "xvi": 16, "xvii": 17, "xviii": 18,
          "xix": 19, "xx": 20}          # i, v, x skipped (ambiguous)

_NUMWORDS_RAW = {
    # EN
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    # DE  (ein/eine skipped)
    "eins": 1, "zwei": 2, "drei": 3, "vier": 4, "funf": 5, "sechs": 6, "sieben": 7, "neun": 9,
    "zehn": 10, "zwolf": 12, "dreizehn": 13, "vierzehn": 14, "funfzehn": 15, "sechzehn": 16,
    "siebzehn": 17, "achtzehn": 18, "neunzehn": 19, "zwanzig": 20,
    # FR  (un/une skipped)
    "deux": 2, "trois": 3, "quatre": 4, "cinq": 5, "sept": 7, "huit": 8, "neuf": 9, "dix": 10,
    "onze": 11, "douze": 12, "treize": 13, "quatorze": 14, "quinze": 15,
    "dixsept": 17, "dixhuit": 18, "dixneuf": 19, "vingt": 20,
    # IT  (uno/una skipped)
    "due": 2, "tre": 3, "quattro": 4, "cinque": 5, "sei": 6, "sette": 7, "otto": 8, "nove": 9,
    "dieci": 10, "undici": 11, "dodici": 12, "tredici": 13, "quattordici": 14, "quindici": 15,
    "sedici": 16, "diciassette": 17, "diciotto": 18, "diciannove": 19, "venti": 20,
    # ES  (uno/una skipped)
    "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7, "ocho": 8, "nueve": 9,
    "diez": 10, "doce": 12, "trece": 13, "catorce": 14, "quince": 15, "dieciseis": 16,
    "diecisiete": 17, "dieciocho": 18, "diecinueve": 19, "veinte": 20,
    # PT  (um/uma skipped)
    "dois": 2, "duas": 2, "quatro": 4, "sete": 7, "oito": 8, "dez": 10, "dezesseis": 16,
    "dezessete": 17, "dezoito": 18, "dezenove": 19, "vinte": 20, "catorze": 14, "quinze": 15,
    # NL  (een skipped)
    "twee": 2, "drie": 3, "vijf": 5, "zes": 6, "zeven": 7, "acht": 8, "negen": 9, "tien": 10,
    "twaalf": 12, "dertien": 13, "veertien": 14, "vijftien": 15, "zestien": 16, "zeventien": 17,
    "achttien": 18, "negentien": 19, "twintig": 20,
}
NUMBER_WORDS = {k: str(v) for k, v in _NUMWORDS_RAW.items()}

# ─── NORMALISATION ───────────────────────────────────────────────────────────

_LEADING_NUM = re.compile(r"^\s*([0-9][0-9 \-]*[0-9]|[0-9])(-?[A-Za-z])?(?=[\s\-\u2013\u2014:,./()]|$)")
_LEAD_SEP = re.compile(r"^[\s\-\u2013\u2014:,./()]+")


def strip_leading_numbers(name: str, min_digits: int = 5) -> str:
    """Remove number blocks in front of the name if they have >= min_digits digits."""
    s = str(name or "")
    for _ in range(5):
        m = _LEADING_NUM.match(s)
        if not m or sum(ch.isdigit() for ch in m.group(1)) < min_digits:
            break
        s = _LEAD_SEP.sub("", s[m.end():])
    return s.strip()


def tokens(text: str) -> list:
    s = unidecode(str(text or "")).lower()
    s = s.replace(".", "").replace("'", "")
    s = s.replace("&", " & ")
    s = re.sub(r"[^a-z0-9&]+", " ", s)
    return s.split()

# ─── LEGAL FORM DICTIONARY (from CSV) ────────────────────────────────────────

_OPT_SEP = "\x00"   # marker: "" or " "


def _expand(pattern: str) -> list:
    """Expand one regex alternative into plain strings (only constructs used in the CSV)."""
    p = pattern
    p = re.sub(r"\[\\s\\\.\]\*", _OPT_SEP, p)             # [\s\.]*
    p = re.sub(r"\((?:\\[./,]\|?)+\)\?", _OPT_SEP, p)      # (\.|\/)?
    p = re.sub(r"\\[s./,]\?", _OPT_SEP, p)                 # \s?  \.?  \,?  \/?
    p = p.replace("\\b", "").replace("^", "").replace("$", "")
    p = p.replace("\\s", " ").replace("\\.", "").replace("\\,", " ").replace("\\/", " ")
    p = p.replace("-", " ")

    outs = [""]
    i = 0
    while i < len(p):
        ch = p[i]
        if ch == "(":
            j = p.index(")", i)
            inner = _expand(p[i + 1:j])
            opt = j + 1 < len(p) and p[j + 1] == "?"
            choices = inner + ([""] if opt else [])
            outs = [o + c for o in outs for c in choices]
            i = j + (2 if opt else 1)
            continue
        if ch == _OPT_SEP:
            outs = [o + c for o in outs for c in ("", " ")]
        else:
            outs = [o + ch for o in outs]
        i += 1
    return outs


def _split_top(regex: str) -> list:
    """Split on '|' that is not inside (...) or [...]."""
    parts, depth, cur = [], 0, ""
    for ch in regex:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "|" and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return parts


def load_legal_forms(csv_path: str) -> dict:
    """
    Returns {"variants": {token_tuple: {"groups": set, "start": bool, "end": bool, "any": bool}},
             "max_len": int, "single": {first_letter: [variant_word, ...]}}
    """
    df = pd.read_csv(csv_path)
    variants = {}

    def add(tok, group, start, end):
        if not tok:
            return
        v = variants.setdefault(tok, {"groups": set(), "start": False, "end": False, "any": False})
        v["groups"].add(group)
        compact = "".join(tok)
        short = len(compact) <= 3 or compact == "private"
        if not short or (not start and not end):
            v["any"] = True
        else:
            v["start"] |= start
            v["end"] |= end

    for _, row in df.iterrows():
        group = re.sub(r"[^A-Z0-9]", "", unidecode(str(row["LegalForm"])).upper())
        covered = set()
        for alt in _split_top(str(row["Regex"])):
            start = "^" in alt
            end = "$" in alt
            for s in _expand(alt):
                tok = tuple(tokens(s))
                add(tok, group, start, end)
                covered.add(tok)
        # forms listed only in RelatedLegalForms (not in the regex) -> anywhere if long, else end
        try:
            related = eval(row["RelatedLegalForms"]) if isinstance(row["RelatedLegalForms"], str) else []
        except Exception:
            related = []
        for r in related:
            tok = tuple(tokens(r))
            if tok and tok not in covered:
                add(tok, group, False, True)

    single = defaultdict(list)
    for tok in variants:
        if len(tok) == 1 and len(tok[0]) >= 5:
            single[tok[0][0]].append(tok[0])
    return {"variants": variants, "max_len": max(len(t) for t in variants), "single": single}

# ─── CLEANING ────────────────────────────────────────────────────────────────

def _typo_fix_last(words, lf):
    """Last word, >= 6 letters: 1 edit from a one-word legal form with same first letter."""
    if not words:
        return words
    w = words[-1]
    if (len(w) < 6 or w in PROTECT_WORDS or (w,) in lf["variants"]
            or w in STOPWORDS or not w.isalpha()):
        return words
    for cand in lf["single"].get(w[0], ()):
        if abs(len(cand) - len(w)) <= 1 and Levenshtein.distance(w, cand, score_cutoff=1) == 1:
            return words[:-1] + [cand]
    return words


def _remove_legal(words, lf, found):
    variants, max_len = lf["variants"], lf["max_len"]
    out = list(words)
    i = 0
    while i < len(out):
        hit = False
        for n in range(min(max_len, len(out) - i), 0, -1):
            v = variants.get(tuple(out[i:i + n]))
            if not v:
                continue
            at_start, at_end = i == 0, i + n == len(out)
            if v["any"] or (v["start"] and at_start) or (v["end"] and at_end):
                found |= v["groups"]
                del out[i:i + n]
                hit = True
                break
        if not hit:
            i += 1
    return out


def _drop_acronym(words):
    """Last word = first letters of the words before it -> drop
    ("icaza trust corporation itc" -> "icaza trust corporation")."""
    if len(words) < 3:
        return words
    last = words[-1]
    if not (2 <= len(last) <= 5 and last.isalpha()):
        return words
    before = words[:-1]
    initials_all = "".join(w[0] for w in before)
    initials_nostop = "".join(w[0] for w in before if w not in STOPWORDS)
    if last in (initials_all, initials_nostop):
        return before
    return words


def _canon(groups: set) -> set:
    out = set()
    for g in groups:
        for eq in LEGAL_EQUIV:
            if g in eq:
                g = min(eq)
                break
        out.add(g)
    return out


def clean_name(name: str, lf: dict):
    """Returns (clean_words, legal_form_groups)."""
    words = tokens(strip_leading_numbers(name))
    words = _drop_acronym(words)
    original = list(words)
    found = set()
    for _ in range(2):
        words = _typo_fix_last(words, lf)
        words = _remove_legal(words, lf, found)
        words = [w for w in words if w not in STOPWORDS]
    if not words:                       # name was only a legal form -> keep original letters
        words = [w for w in original if w != "&"] or original
    return words, found


def numbers_of(words) -> frozenset:
    out = set()
    for w in words:
        for d in re.findall(r"[0-9]+", w):
            out.add(d.lstrip("0") or "0")
        if w in _ROMAN:
            out.add(str(_ROMAN[w]))
        elif w in NUMBER_WORDS:
            out.add(NUMBER_WORDS[w])
    return frozenset(out)

# ─── PAIR CHECK ──────────────────────────────────────────────────────────────

def check_pair(direct_name, good_name, lf, threshold, legal_check=False):
    dw, dlf = clean_name(direct_name, lf)
    gw, glf = clean_name(good_name, lf)
    dc, gc = "".join(dw), "".join(gw)
    dclean, gclean = " ".join(dw), " ".join(gw)

    score = max(fuzz.ratio(dc, gc),
                fuzz.token_sort_ratio(dclean, gclean),
                JaroWinkler.normalized_similarity(dc, gc) * 100)
    nums_ok = numbers_of(dw) == numbers_of(gw)
    lf_ok = (not legal_check) or (not dlf) or (not glf) or bool(_canon(dlf) & _canon(glf))

    reasons = []
    if score < threshold:
        reasons.append(f"Score<{threshold:g}")
    if not nums_ok:
        reasons.append("Numbers differ")
    if not lf_ok:
        reasons.append("Legal form differs")

    return {
        "Direct_Clean": dclean, "Good_Clean": gclean,
        "Direct_LegalForms": ", ".join(sorted(dlf)), "Good_LegalForms": ", ".join(sorted(glf)),
        "Name_Score": round(score, 2),
        "Name_Pass": not reasons,
        "Fail_Reason": "; ".join(reasons),
    }


def check_pairs(pdf: pd.DataFrame, csv_path: str, threshold: float,
                legal_check: bool = False) -> pd.DataFrame:
    """pdf needs Direct_Name, Good_Name. Returns pdf + name-check columns."""
    lf = load_legal_forms(csv_path)
    res = [check_pair(d, g, lf, threshold, legal_check) for d, g in zip(pdf["Direct_Name"], pdf["Good_Name"])]
    return pd.concat([pdf.reset_index(drop=True), pd.DataFrame(res)], axis=1)
