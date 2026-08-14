"""
All file paths for the MFS assignment live here.

"""

from pathlib import Path

# ---------------------------------------------------------------------
# Root of the extracted download.
# NOTE: the folder name really is doubled - that is how the zip extracts.
# The r"..." prefix makes Python treat backslashes literally (Windows).
# ---------------------------------------------------------------------
FRAMEWORK = Path(
    r"C:\Users\shems\Study Materials\NLP\Assignment_MFS_F1"
    r"\WSD_Evaluation_Framework\WSD_Evaluation_Framework"
)

TRAIN_DIR = FRAMEWORK / "Training_Corpora"
EVAL_DIR = FRAMEWORK / "Evaluation_Datasets"

# ---------------------------------------------------------------------
# Training corpus: SemCor only.
# Ignore SemCor+OMSTI - OMSTI is automatically tagged, and MFS is
# defined over SemCor alone.
# ---------------------------------------------------------------------
SEMCOR_XML = TRAIN_DIR / "SemCor" / "semcor.data.xml"
SEMCOR_KEY = TRAIN_DIR / "SemCor" / "semcor.gold.key.txt"

# ---------------------------------------------------------------------
# Evaluation sets. ALL = the five competition sets concatenated
# (Raganato et al. 2017). The Maru et al. 2022 set is a manually
# corrected version of this same data - add it here when you get it.
# ---------------------------------------------------------------------
EVAL_SETS = {
    "ALL":          EVAL_DIR / "ALL"          / "ALL.data.xml",
    "senseval2":    EVAL_DIR / "senseval2"    / "senseval2.data.xml",
    "senseval3":    EVAL_DIR / "senseval3"    / "senseval3.data.xml",
    "semeval2007":  EVAL_DIR / "semeval2007"  / "semeval2007.data.xml",
    "semeval2013":  EVAL_DIR / "semeval2013"  / "semeval2013.data.xml",
    "semeval2015":  EVAL_DIR / "semeval2015"  / "semeval2015.data.xml",
    # "maru2022":   Path(r"...\maru2022.data.xml"),
    # "hardEN":     Path(r"...\hardEN.data.xml"),
    # "42D":        Path(r"...\42D.data.xml"),
}

RESULTS_DIR = FRAMEWORK.parent.parent / "results"


def key_for(xml_path: Path) -> Path:
    """
    Gold key path for a given data file.
    Convention: foo.data.xml -> foo.gold.key.txt
    """
    return xml_path.with_name(xml_path.name.replace(".data.xml",
                                                    ".gold.key.txt"))


def check() -> bool:
    """Verify every configured path exists. Run this first."""
    problems = []

    print(f"framework root: {FRAMEWORK}")
    print(f"  exists: {FRAMEWORK.exists()}\n")
    if not FRAMEWORK.exists():
        print("  Root not found. Check the path - remember the doubled")
        print("  WSD_Evaluation_Framework folder.")
        return False

    for label, xml in [("SemCor", SEMCOR_XML)] + list(EVAL_SETS.items()):
        gold = SEMCOR_KEY if label == "SemCor" else key_for(xml)
        x, g = xml.exists(), gold.exists()
        mark = "ok  " if (x and g) else "MISS"
        print(f"  [{mark}] {label:<14} xml={x!s:<5} key={g!s}")
        if not x:
            problems.append(str(xml))
        if not g:
            problems.append(str(gold))

    if problems:
        print("\nMissing files:")
        for p in problems:
            print("   ", p)
        print("\nList the actual folder contents to find the real names:")
        print(f"   dir \"{EVAL_DIR}\"")
        return False

    print("\nAll paths resolve.")
    return True


if __name__ == "__main__":
    check()