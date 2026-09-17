"""Turn enwik9 (the first 10^9 bytes of the 2006 English Wikipedia XML dump) into fil9: lowercase a-z and single

    python3 scripts/make_fil9.py data/fil9/enwik9 data/fil9/fil9 --check data/text8/text8"""
import argparse
import re
import sys

DIGITS = {"0": " zero ", "1": " one ", "2": " two ", "3": " three ", "4": " four ",
          "5": " five ", "6": " six ", "7": " seven ", "8": " eight ", "9": " nine "}
RULES = [
    (re.compile(r"<.*>"), ""),                                  # xml tags (greedy within the record)
    (re.compile(r"&amp;"), "&"), (re.compile(r"&lt;"), "<"), (re.compile(r"&gt;"), ">"),
    (re.compile(r"<ref[^<]*<\/ref>"), ""),                     # references
    (re.compile(r"<[^>]*>"), ""),                               # xhtml tags
    (re.compile(r"\[http:[^] ]*"), "["),                        # urls: keep visible text
    (re.compile(r"\|thumb", re.I), ""), (re.compile(r"\|left", re.I), ""), (re.compile(r"\|right", re.I), ""),
    (re.compile(r"\|\d+px", re.I), ""),
    (re.compile(r"\[\[image:[^\[\]]*\|", re.I), ""),
    (re.compile(r"\[\[category:([^|\]]*)[^]]*\]\]", re.I), r"[[\1]]"),
    (re.compile(r"\[\[[a-z\-]*:[^\]]*\]\]"), ""),              # interlanguage links
    (re.compile(r"\[\[[^\|\]]*\|"), "[["),                      # wiki url: keep visible text
    (re.compile(r"\{\{[^\}]*\}\}"), ""), (re.compile(r"\{[^\}]*\}"), ""),   # templates, tables
    (re.compile(r"\["), ""), (re.compile(r"\]"), ""),
    (re.compile(r"&[^;]*;"), " "),                              # remaining entities
]
NON_AZ = re.compile(r"[^a-z]+")
REDIRECT = re.compile(r"#redirect", re.I)
UPPER_TO_LOWER = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")


def clean_record(rec: str) -> str:
    """- Apply the markup-stripping rules to one record and return its plain-text form, space-separated.
        - Perl's order (markup, case fold, digits, collapse) since each stage assumes the last ran"""
    for rx, sub in RULES:
        rec = rx.sub(sub, rec)
    rec = " " + rec + " "
    rec = rec.translate(UPPER_TO_LOWER)       # tr/A-Z/a-z/ : ASCII only, as Perl does on bytes (no Unicode casefolding)
    rec = "".join(DIGITS.get(c, c) for c in rec)
    rec = NON_AZ.sub(" ", rec)          # tr/a-z/ /cs : complement + squeeze
    return rec[:-1]                     # chop


def convert(src: str, dst: str, chunk: int = 1 << 24) -> None:
    """- Stream the 1 GB dump in chunks (records end at '>'), so peak memory stays under ~100 MB.
    - latin-1 keeps it byte-oriented like the Perl original; multi-byte UTF-8 becomes spaces, matching text8.
    - Output is tokenized by dgpt/data.py straight from the raw bytes."""
    out = open(dst, "w", encoding="latin-1")
    text = False
    n = 0
    tail = ""
    with open(src, "r", encoding="latin-1") as f:      # latin-1: one char per byte, exactly how the Perl original sees it
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            parts = (tail + buf).split(">")
            tail = parts.pop()                          # incomplete last record: carry into the next chunk
            for rec in parts:
                rec = rec + ">"
                if "<text " in rec:
                    text = True
                if REDIRECT.search(rec):
                    text = False
                if text:
                    if "</text>" in rec:
                        text = False
                    out.write(clean_record(rec))
                    n += 1
    if tail:                                            # final record without a closing '>' (Perl would emit it too)
        if "<text " in tail:
            text = True
        if REDIRECT.search(tail):
            text = False
        if text:
            out.write(clean_record(tail))
            n += 1
    out.close()
    print(f"wrote {dst} from {n} text records")


def main():
    """- Convert src to dst and, with --check, verify the result's first 100M bytes equal text8."""
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--check", help="path to text8: verify fil9[:1e8] == text8")
    a = ap.parse_args()
    convert(a.src, a.dst)
    if a.check:
        t8 = open(a.check, "rb").read()
        f9 = open(a.dst, "rb").read(len(t8))
        if f9 == t8:
            print("check OK: fil9 prefix is byte-identical to text8")
        else:
            i = next(i for i in range(min(len(t8), len(f9))) if t8[i] != f9[i]) if len(f9) else 0
            print(f"check FAILED: first difference at byte {i}: text8={t8[i-40:i+40]!r} fil9={f9[i-40:i+40]!r}")
            sys.exit(1)


if __name__ == "__main__":
    main()
