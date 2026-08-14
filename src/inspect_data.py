"""
Session 1 diagnostic: inspect a Raganato-format WSD dataset.

Answers every question in roadmap steps 1.2 and 1.3 without touching the
MFS algorithm.

Usage (from the src folder):

    python inspect_data.py                 # SemCor + ALL  (the default)
    python inspect_data.py semcor          # just SemCor
    python inspect_data.py ALL senseval2   # named eval sets
    python inspect_data.py all_sets        # every configured dataset
"""

import sys
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

import config

SS_TYPE = {"1": "noun", "2": "verb", "3": "adj(head)",
           "4": "adv", "5": "adj(satellite)"}


def inspect(xml_path, key_path, label=""):
    xml_path, key_path = Path(xml_path), Path(key_path)

    if not xml_path.exists():
        print(f"  MISSING: {xml_path}")
        return None
    if not key_path.exists():
        print(f"  MISSING: {key_path}")
        return None

    # ---------- 1. the XML ----------
    # iterparse streams the file rather than loading it whole: SemCor's
    # XML is large, and el.clear() frees each element after use.
    n_instances = n_wf = n_sentences = n_texts = 0
    pos_tags = Counter()
    sample, mwe, upper, ids = [], [], [], []

    for _, el in ET.iterparse(xml_path, events=("end",)):
        if el.tag == "instance":
            n_instances += 1
            lemma, pos = el.get("lemma"), el.get("pos")
            pos_tags[pos] += 1
            ids.append(el.get("id"))
            if n_instances <= 5:
                sample.append((el.get("id"), lemma, pos, el.text))
            if lemma and "_" in lemma and len(mwe) < 5:
                mwe.append(lemma)
            if lemma and lemma != lemma.lower() and len(upper) < 5:
                upper.append(lemma)
            el.clear()
        elif el.tag == "wf":
            n_wf += 1
            el.clear()
        elif el.tag == "sentence":
            n_sentences += 1
            el.clear()
        elif el.tag == "text":
            n_texts += 1
            el.clear()

    # ---------- 2. the gold key ----------
    gold, n_lines, multi, multi_example = {}, 0, 0, None
    key_pos, weird = Counter(), []

    with open(key_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            parts = line.split()
            gold[parts[0]] = parts[1:]
            if len(parts) > 2:
                multi += 1
                if multi_example is None:
                    multi_example = line
            for k in parts[1:]:
                try:
                    key_pos[k.split("%")[1][0]] += 1
                except IndexError:
                    if len(weird) < 5:
                        weird.append(k)

    # ---------- 3. cross-check ----------
    id_set, gold_set = set(ids), set(gold)
    missing_gold = id_set - gold_set
    orphan_gold = gold_set - id_set
    dup_ids = n_instances - len(id_set)

    # ---------- report ----------
    print("=" * 64)
    print(f"{label or xml_path.name}")
    print(f"  {xml_path}")
    print("=" * 64)

    print("\n--- STRUCTURE ---")
    print(f"  <text>      {n_texts:>9,}")
    print(f"  <sentence>  {n_sentences:>9,}")
    print(f"  <wf>        {n_wf:>9,}   (not annotated)")
    print(f"  <instance>  {n_instances:>9,}   <-- your data points")

    print("\n--- POS SCHEME (XML attribute) ---")
    for p, c in pos_tags.most_common():
        print(f"  {p:<10} {c:>9,}")

    print("\n--- POS (ss_type digit in sense keys) ---")
    for d, c in sorted(key_pos.items()):
        print(f"  {d} = {SS_TYPE.get(d,'?'):<16} {c:>9,}")

    print("\n--- FIRST 5 INSTANCES ---")
    for iid, lem, pos, txt in sample:
        print(f"  {iid:<20} lemma={lem:<14} pos={pos:<6} text={txt!r}")
        print(f"  {'':<20} gold ={gold.get(iid)}")

    print("\n--- GOLD KEY ---")
    print(f"  lines                 {n_lines:>9,}")
    pct = 100 * multi / max(n_lines, 1)
    print(f"  lines with 2+ senses  {multi:>9,}   ({pct:.2f}%)")
    if multi_example:
        print(f"  example: {multi_example}")

    print("\n--- FORMAT HAZARDS ---")
    print(f"  multi-word lemmas : {mwe or 'none in first 5'}")
    print(f"  capitalised lemmas: {upper or 'none in first 5'}")
    if weird:
        print(f"  UNPARSEABLE KEYS  : {weird}")

    print("\n--- CONSISTENCY (all must be zero) ---")
    print(f"  duplicate instance ids         {dup_ids:>6}")
    print(f"  instances with no gold entry   {len(missing_gold):>6}")
    print(f"  gold entries with no instance  {len(orphan_gold):>6}")
    if missing_gold:
        print(f"    e.g. {sorted(missing_gold)[:3]}")
    if orphan_gold:
        print(f"    e.g. {sorted(orphan_gold)[:3]}")

    ok = (n_instances == n_lines and not missing_gold
          and not orphan_gold and not dup_ids)
    print("\n  " + ("OK - counts match, safe to proceed"
                    if ok else
                    "MISMATCH - investigate before writing the baseline"))
    print()
    return n_instances


def main():
    args = sys.argv[1:]

    # explicit pair of paths
    if len(args) == 2 and args[0].endswith(".xml"):
        inspect(args[0], args[1])
        return

    if not args:
        targets = [("SemCor", config.SEMCOR_XML, config.SEMCOR_KEY),
                   ("ALL", config.EVAL_SETS["ALL"],
                    config.key_for(config.EVAL_SETS["ALL"]))]
    elif args == ["all_sets"]:
        targets = [("SemCor", config.SEMCOR_XML, config.SEMCOR_KEY)]
        targets += [(n, p, config.key_for(p))
                    for n, p in config.EVAL_SETS.items()]
    else:
        targets = []
        for a in args:
            if a.lower() == "semcor":
                targets.append(("SemCor", config.SEMCOR_XML,
                                config.SEMCOR_KEY))
            elif a in config.EVAL_SETS:
                p = config.EVAL_SETS[a]
                targets.append((a, p, config.key_for(p)))
            else:
                print(f"unknown dataset: {a}")
                print(f"available: semcor, {', '.join(config.EVAL_SETS)}")
                return

    counts = {}
    for label, xml, key in targets:
        n = inspect(xml, key, label)
        if n is not None:
            counts[label] = n

    if len(counts) > 1:
        print("=" * 64)
        print("SUMMARY")
        for k, v in counts.items():
            print(f"  {k:<14} {v:>9,} instances")
        if "SemCor" in counts:
            print(f"\n  SemCor expected ~226,036 -> "
                  f"{'looks right' if abs(counts['SemCor']-226036) < 500 else 'CHECK THIS'}")


if __name__ == "__main__":
    main()