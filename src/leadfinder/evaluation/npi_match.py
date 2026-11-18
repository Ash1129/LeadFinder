"""Practice matching on the real NPI practice table: does a company-name hint find the right registry record, and
does a practice that is not in the table stay unmatched?

    python -m leadfinder.evaluation.npi_match            # writes docs/npi-match-report.md and .json

The table is split once (seeded): 70% becomes the directory, 30% is held out. Each practice gets the same hint
variants a visit-identity source plausibly produces (the registry name, the name without its legal suffix, a
casual form, and a one-letter typo), each with and without a "City, ST" location hint. Hints for directory practices
should match their own record; hints for held-out practices should match nothing, so any match there is a false
match onto a real, similarly named practice. The pipeline is the one the research agent uses without its model:
TablePracticeDirectory.lookup -> research.rank_candidates -> the threshold-and-margin rule. As in match_eval.py,
this is the best case for the agent, which can only confirm a match these safeguards allow.

The hints are generated from registry names, not observed from real visitor tracking, so the rates say how the
matcher behaves on real practice names, not how often a real identity source supplies a usable name.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from unittest import mock

from .. import npi
from .. import research as rs

VARIANTS = ("registry_name", "no_legal_suffix", "casual", "typo")
THRESHOLDS = (0.75, 0.80, 0.85, 0.90, 0.95)
MARGINS = (0.05, 0.10, 0.15)
DOCS = Path(__file__).resolve().parents[3] / "docs"


def variant(name: str, kind: str, rng: random.Random) -> str:
    words = name.replace(",", " ").split()
    plain = [w for w in words if rs._normalized(w) not in rs.LEGAL_WORDS] or words
    if kind == "registry_name":
        return name
    if kind == "no_legal_suffix":
        return " ".join(plain)
    if kind == "casual":
        casual = " ".join(plain).lower().replace(" and ", " & ")
        return casual.replace(" associates", "").replace(" group", "") or casual
    long_words = [i for i, w in enumerate(plain) if len(w) >= 5]
    if not long_words:
        return " ".join(plain)
    i = rng.choice(long_words)
    w = plain[i]
    j = rng.randrange(1, len(w) - 1)
    plain[i] = w[:j - 1] + w[j] + w[j - 1] + w[j + 1:]
    return " ".join(plain)


def decide(ranked, threshold: float, margin: float) -> tuple[str, str | None]:
    """ResearchAgent._result's rule for a model that picks the top candidate, with the cutoffs as parameters."""
    if not ranked:
        return "not_found", None
    top, second = ranked[0], (ranked[1] if len(ranked) > 1 else None)
    if top.confidence >= threshold and (second is None or top.confidence - second.confidence >= margin):
        return "matched", top.practice_id
    if top.confidence >= rs.AMBIGUOUS_THRESHOLD:
        return "ambiguous", None
    return "not_found", None


def queries(records, seed: int):
    """(record, held_out, variant, with_location, request) for every practice and hint form."""
    rng = random.Random(seed)
    order = sorted(records, key=lambda r: r.npi)
    rng.shuffle(order)
    cut = int(0.7 * len(order))
    directory, held_out = order[:cut], order[cut:]
    out = []
    for held, group in ((False, directory), (True, held_out)):
        for r in group:
            for kind in VARIANTS:
                hint = variant(r.name, kind, rng)
                for with_location in (True, False):
                    out.append((r, held, kind, with_location,
                                rs.ResearchRequest(hint, None, r.location if with_location else None)))
    return directory, out


def run(records, seed: int, normalizer=None) -> list[dict]:
    """Rank every query once and keep the top three, so cutoffs can be swept without re-ranking."""
    patches = [mock.patch.object(rs, "normalize_name", normalizer), mock.patch.object(npi, "normalize_name", normalizer)] \
        if normalizer else []
    for p in patches:
        p.start()
    try:
        directory_records, qs = queries(records, seed)
        directory = npi.TablePracticeDirectory(directory_records)
        rows = []
        for record, held, kind, with_location, request in qs:
            strategy = "name_domain_location" if with_location else "name_only"
            found = directory.lookup(rs.PracticeLookupRequest(request.practice_name, None, request.location_hint,
                                                              strategy))
            ranked = rs.rank_candidates(request, found)[:3]
            rows.append({"truth": record.practice_id, "truth_name": rs.normalize_name(record.name), "held_out": held, "variant": kind,
                         "with_location": with_location, "hint": request.practice_name, "ranked": ranked})
        return rows
    finally:
        for p in patches:
            p.stop()


def score(rows, threshold: float, margin: float) -> dict:
    pos, neg, false_kind = Counter(), Counter(), Counter()
    by_variant: dict[str, Counter] = {}
    for row in rows:
        status, chosen = decide(row["ranked"], threshold, margin)
        if row["held_out"]:
            outcome = "false_match" if status == "matched" else status
            neg[outcome] += 1
            if status == "matched":
                same = rs.normalize_name(row["ranked"][0].name) == row["truth_name"]
                false_kind["same_name_other_site" if same else "different_practice"] += 1
        else:
            if status == "matched":
                outcome = "correct" if chosen == row["truth"] else "wrong_match"
            elif status == "ambiguous":
                outcome = "ambiguous_truth_in_top3" if any(c.practice_id == row["truth"] for c in row["ranked"]) \
                    else "ambiguous_truth_missing"
            else:
                outcome = "not_found"
            pos[outcome] += 1
            key = f"{row['variant']}{'' if row['with_location'] else ' (no location)'}"
            by_variant.setdefault(key, Counter())[outcome] += 1
    n_pos, n_neg = sum(pos.values()), sum(neg.values())
    return {
        "threshold": threshold, "margin": margin,
        "positives": n_pos, "negatives": n_neg,
        "correct_rate": pos["correct"] / n_pos,
        "wrong_match_rate": pos["wrong_match"] / n_pos,
        "false_match_rate": neg["false_match"] / n_neg,
        "false_match_different_rate": false_kind["different_practice"] / n_neg,
        "positive_outcomes": dict(pos), "negative_outcomes": dict(neg), "false_match_kinds": dict(false_kind),
        "by_variant": {k: dict(v) for k, v in sorted(by_variant.items())},
    }


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def report(records, seed: int) -> tuple[str, dict]:
    v1_rows = run(records, seed, normalizer=rs._normalized)
    v2_rows = run(records, seed)
    current = (rs.MATCH_THRESHOLD, rs.MATCH_MARGIN)
    v1, v2 = score(v1_rows, *current), score(v2_rows, *current)
    sweep = [score(v2_rows, t, m) for t in THRESHOLDS for m in MARGINS]
    examples = [r for r in v2_rows if r["held_out"] and decide(r["ranked"], *current)[0] == "matched"][:5]

    lines = [
        "# Practice matching on the real NPI table",
        "",
        f"Generated by `python -m leadfinder.evaluation.npi_match --seed {seed}` from `data/npi_practices.json` "
        f"({len(records)} practices). Directory: 70% of the table; held out: 30%. Each practice is queried with "
        f"{len(VARIANTS)} hint forms, each with and without a location hint "
        f"({v2['positives']} positive and {v2['negatives']} held-out queries). Hints are generated from registry names; "
        "they are not observed visitor data.",
        "",
        f"## Name normalization: before and after (threshold {current[0]}, margin {current[1]})",
        "",
        "| normalization | correct match | wrong match | false match on held-out | not found or ambiguous |",
        "|---|---|---|---|---|",
    ]
    for label, s in (("v1: punctuation only", v1), ("v2: legal suffixes, initials, 'and' removed", v2)):
        rest = 1 - s["correct_rate"] - s["wrong_match_rate"]
        lines.append(f"| {label} | {_pct(s['correct_rate'])} | {_pct(s['wrong_match_rate'])} | "
                     f"{_pct(s['false_match_rate'])} | {_pct(rest)} |")
    lines += ["", "## By hint form (v2, current cutoffs)", "",
              "| hint | n | correct | wrong | ambiguous, truth in top 3 | ambiguous, truth missing | not found |",
              "|---|---|---|---|---|---|---|"]
    for key, c in v2["by_variant"].items():
        n = sum(c.values())
        lines.append(f"| {key} | {n} | " + " | ".join(_pct(c.get(k, 0) / n) for k in (
            "correct", "wrong_match", "ambiguous_truth_in_top3", "ambiguous_truth_missing", "not_found")) + " |")
    lines += ["", "## Cutoff sweep (v2)", "",
              "| threshold | margin | correct match | wrong match | false match on held-out | of which a different practice |",
              "|---|---|---|---|---|---|"]
    for s in sweep:
        mark = " **(current)**" if (s["threshold"], s["margin"]) == current else ""
        lines.append(f"| {s['threshold']}{mark} | {s['margin']} | {_pct(s['correct_rate'])} | "
                     f"{_pct(s['wrong_match_rate'])} | {_pct(s['false_match_rate'])} | "
                     f"{_pct(s['false_match_different_rate'])} |")
    lines += ["", "A held-out practice that shares its exact name with a practice in another city (another site of the same "
              "organization, or an unrelated practice with the same name) can match that record when the hint has no "
              "location; the last column counts only false matches onto a differently named practice.", ""]
    if examples:
        lines += ["## Held-out hints that matched another record (current cutoffs)", "",
                  "| hint | matched | confidence |", "|---|---|---|"]
        for r in examples:
            top = r["ranked"][0]
            lines.append(f"| {r['hint']} | {top.name} ({top.location}) | {top.confidence} |")
    data = {"seed": seed, "practices": len(records), "v1": v1, "v2": v2, "sweep": sweep}
    return "\n".join(lines) + "\n", data


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m leadfinder.evaluation.npi_match")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--table", type=Path, default=npi.TABLE_PATH)
    parser.add_argument("--out", type=Path, default=DOCS / "npi-match-report.md")
    args = parser.parse_args(argv)
    text, data = report(npi.load_table(args.table), args.seed)
    args.out.write_text(text)
    args.out.with_suffix(".json").write_text(json.dumps(data, indent=1, default=str) + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
