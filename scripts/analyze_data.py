import csv, sys
from collections import Counter

with open("data/amp_dataset_clean.csv", "r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

results = []
results.append(f"Total rows: {len(rows)}")
results.append(f"Labels: {set(r['label'] for r in rows)}")

lens = [len(r["sequence"]) for r in rows]
results.append(f"Seq length range: {min(lens)}-{max(lens)}")
results.append(f"Avg length: {sum(lens)/len(lens):.1f}")

lowercase_count = sum(1 for r in rows if any(c.islower() for c in r["sequence"]))
results.append(f"Lowercase count: {lowercase_count}")

STD_AA = set("ACDEFGHIKLMNPQRSTVWY")
non_std = []
for r in rows:
    for c in r["sequence"]:
        if c.upper() not in STD_AA:
            non_std.append((r["sequence"], c))
results.append(f"Non-std AA count: {len(non_std)}")
if non_std:
    results.append(f"Non-std examples: {non_std[:10]}")

unique_seqs = set(r["sequence"] for r in rows)
results.append(f"Unique seqs: {len(unique_seqs)}")

short_seqs = [r["sequence"] for r in rows if len(r["sequence"]) < 5]
results.append(f"Seqs < 5 AA: {len(short_seqs)}")
if short_seqs:
    results.append(f"Short examples: {short_seqs[:10]}")

with open("data_analysis_report.txt", "w") as f:
    f.write("\n".join(results))
print("\n".join(results))
print("\nResults saved to data_analysis_report.txt")
