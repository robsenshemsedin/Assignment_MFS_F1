"""
MFS vs WordNet-S1 baselines, run over every available English WSD test set.

Purpose: find out which dataset the 65.2 figure in the EMNLP 2025 paper
actually refers to. Run it on ALL and ALLamended together and compare.

Both the training corpus (SemCor) and the test sets use the same
Raganato format, so a single parser handles everything:

    <instance id="..." lemma="..." pos="NOUN">surface</instance>
    d000.s000.t000 lemma%1:04:00::

Usage
-----
    python compare_baselines.py                  # auto-discovers everything
    python compare_baselines.py --root ..        # if run from src/
    python compare_baselines.py --backoff none   # abstain instead of S1
"""

import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict, Counter
from pathlib import Path

from nltk.corpus import wordnet as wn

POS_MAP = {"NOUN": "n", "VERB": "v", "ADJ": "a", "ADV": "r"}


# ----------------------------------------------------------------- parsing
def find_pair(directory):
    """Locate the (data.xml, gold.key.txt) pair inside a directory."""
    directory = Path(directory)
    xmls = sorted(directory.glob("*.data.xml"))
    keys = sorted(directory.glob("*.gold.key.txt"))
    if not xmls or not keys:
        return None
    return xmls[0], keys[0]


def read_instances(xml_path):
    """-> [(instance_id, lemma, wn_pos)] for content words only."""
    out = []
    for _, elem in ET.iterparse(xml_path, events=("end",)):
        if elem.tag != "instance":
            continue
        pos = POS_MAP.get(elem.get("pos"))
        if pos:
            out.append((elem.get("id"), elem.get("lemma").lower(), pos))
        elem.clear()
    return out


def read_gold(path):
    """-> {instance_id: {sense_key, ...}}  (an instance may have several)."""
    gold = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()      # .strip() also kills the \r
            if len(parts) >= 2:
                gold[parts[0]] = set(parts[1:])
    return gold


# ------------------------------------------------------------- WordNet S1
def wn_first(lemma, pos, restrict=None):
    """The sense WordNet lists first for this lemma+pos."""
    for syn in wn.synsets(lemma, pos):
        for lem in syn.lemmas():
            if lem.name().lower() != lemma.lower():
                continue
            if restrict is None or lem.key() in restrict:
                return lem.key()
    return None


# ------------------------------------------------------- stage 1: counting
def count_training(xml_path, key_path):
    """(lemma, pos) -> Counter(sense_key -> n), counted over SemCor."""
    counts = defaultdict(Counter)
    gold = read_gold(key_path)
    seen = skipped = 0
    for iid, lemma, pos in read_instances(xml_path):
        senses = gold.get(iid)
        if not senses:
            skipped += 1
            continue
        for key in senses:
            counts[(lemma, pos)][key] += 1
            seen += 1
    return counts, seen, skipped


def build_mfs(counts):
    """One sense per (lemma, pos); ties broken by WordNet sense order."""
    mfs, ties = {}, 0
    for (lemma, pos), dist in counts.items():
        top = max(dist.values())
        tied = [k for k, c in dist.items() if c == top]
        if len(tied) == 1:
            mfs[(lemma, pos)] = tied[0]
        else:
            ties += 1
            mfs[(lemma, pos)] = wn_first(lemma, pos, restrict=tied) or sorted(tied)[0]
    return mfs, ties


# ------------------------------------------------------ stages 3+4: scoring
def score(instances, gold, predict):
    correct = attempted = 0
    for iid, lemma, pos in instances:
        pred = predict(lemma, pos)
        if pred is None:
            continue                       # abstained
        attempted += 1
        if pred in gold.get(iid, ()):
            correct += 1
    n = len(instances)
    p = correct / attempted if attempted else 0.0
    r = correct / n if n else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return dict(n=n, attempted=attempted, correct=correct,
                P=100 * p, R=100 * r, F1=100 * f1)


def per_pos(instances, gold, predict):
    tally = defaultdict(lambda: [0, 0])
    for iid, lemma, pos in instances:
        tally[pos][1] += 1
        pred = predict(lemma, pos)
        if pred and pred in gold.get(iid, ()):
            tally[pos][0] += 1
    return {p: (100 * c / t if t else 0.0, t) for p, (c, t) in tally.items()}


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".",
                    help="project root (the folder holding ALLamended.data.xml)")
    ap.add_argument("--backoff", choices=["s1", "none"], default="s1")
    args = ap.parse_args()
    root = Path(args.root).resolve()

    # -- locate the framework, however deeply it nested itself ------------
    fw = None
    for cand in root.rglob("Evaluation_Datasets"):
        fw = cand.parent
        break
    if fw is None:
        raise SystemExit(f"could not find Evaluation_Datasets under {root}")
    print(f"framework: {fw}")

    # -- locate SemCor ----------------------------------------------------
    semcor_dir = None
    for cand in (fw / "Training_Corpora").rglob("*"):
        if cand.is_dir() and find_pair(cand) and "semcor" in cand.name.lower():
            semcor_dir = cand
            break
    if semcor_dir is None:
        raise SystemExit("could not find SemCor under Training_Corpora")
    sc_xml, sc_key = find_pair(semcor_dir)
    print(f"training : {semcor_dir.name}\n")

    # -- stage 1 + 2 ------------------------------------------------------
    counts, seen, skipped = count_training(sc_xml, sc_key)
    mfs, ties = build_mfs(counts)
    print(f"SemCor: {seen:,} sense occurrences, "
          f"{len(counts):,} (lemma,pos) entries, {ties:,} ties broken")
    if skipped:
        print(f"        {skipped} instances had no gold entry (skipped)")

    # -- assemble the test sets ------------------------------------------
    datasets = {}
    for d in sorted((fw / "Evaluation_Datasets").iterdir()):
        if d.is_dir() and find_pair(d):
            datasets[d.name] = find_pair(d)
    for name in ("ALLamended", "hardEN", "42D"):
        pair = (root / f"{name}.data.xml", root / f"{name}.gold.key.txt")
        if pair[0].exists() and pair[1].exists():
            datasets[name] = pair
        else:
            hit = find_pair(root / name) if (root / name).is_dir() else None
            if hit:
                datasets[name] = hit

    # -- predictors -------------------------------------------------------
    def predict_mfs(lemma, pos):
        p = mfs.get((lemma, pos))
        if p is None and args.backoff == "s1":
            p = wn_first(lemma, pos)
        return p

    def predict_s1(lemma, pos):
        return wn_first(lemma, pos)

    # -- run ---------------------------------------------------------------
    print(f"\nbackoff = {args.backoff}\n")
    hdr = (f"{'dataset':<16}{'N':>7}{'MFS_F1':>9}{'S1_F1':>8}"
           f"{'unseen':>8}{'agree%':>8}")
    print(hdr)
    print("-" * len(hdr))

    results = {}
    for name, (xml_p, key_p) in datasets.items():
        insts = read_instances(xml_p)
        gold = read_gold(key_p)
        if not insts:
            continue
        m = score(insts, gold, predict_mfs)
        s = score(insts, gold, predict_s1)
        unseen = sum(1 for _, l, p in insts if (l, p) not in mfs)
        same = sum(1 for _, l, p in insts
                   if predict_mfs(l, p) == predict_s1(l, p))
        results[name] = (m, s, insts, gold)
        print(f"{name:<16}{m['n']:>7}{m['F1']:>9.1f}{s['F1']:>8.1f}"
              f"{unseen:>8}{100 * same / len(insts):>8.1f}")

    # -- detail on the two datasets in question ---------------------------
    for name in ("ALL", "ALLamended"):
        if name not in results:
            continue
        m, s, insts, gold = results[name]
        print(f"\n--- {name} ---")
        print(f"  MFS   P={m['P']:.1f}  R={m['R']:.1f}  F1={m['F1']:.1f}  "
              f"({m['correct']}/{m['n']})")
        print(f"  S1    P={s['P']:.1f}  R={s['R']:.1f}  F1={s['F1']:.1f}  "
              f"({s['correct']}/{s['n']})")
        print("  per-POS (MFS):", end=" ")
        for p, (acc, n) in sorted(per_pos(insts, gold, predict_mfs).items()):
            print(f"{p}={acc:.1f}(n={n})", end="  ")
        print()


if __name__ == "__main__":
    main()