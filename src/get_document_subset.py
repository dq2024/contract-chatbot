"""
1. Groups data/contracts/ PDFs by shared 5-digit prefix.
2. Prints groups sorted by total size, smallest first, stopping once the
   cumulative file count across listed groups reaches or exceeds 130.
3. Copies selected PDFs into data/selected_contracts/ and writes data/selected_contracts.csv.

Run from project root:
    python3 data/get_document_subset.py
"""

import csv
import os
import re
import shutil

CONTRACTS_DIR = "data/contracts"
SELECTED_DIR  = "data/selected_contracts"
OUTPUT_CSV    = "data/selected_contracts.csv"


# Match a 5-digit prefix only when not surrounded by other digits
def contains_prefix(text, p):
    return bool(re.search(r'(?<!\d)' + p + r'(?!\d)', text))


def group_size(file_list):
    return sum(os.path.getsize(os.path.join(CONTRACTS_DIR, f)) for f in file_list)


# Step 1: unique 5-digit prefixes from filenames that start with 5 digits
ac_files = [f for f in os.listdir(CONTRACTS_DIR) if os.path.isfile(os.path.join(CONTRACTS_DIR, f))]
prefixes  = sorted({m.group(1) for f in ac_files for m in [re.match(r'^(\d{5})', f)] if m})
print(f"Found {len(prefixes)} unique 5-digit prefixes.\n")

# Step 2: group contracts/ files by prefix
groups = {p: set(f for f in ac_files if contains_prefix(f, p)) for p in prefixes}

# Step 3: compute total group sizes, sort ascending
ranked = sorted(
    [(p, sorted(groups[p]), group_size(groups[p])) for p in groups if groups[p]],
    key=lambda x: x[2]
)

# Step 4: print until cumulative file count >= 130 (finish current group)
print(f"{'Group':>7}  {'Files':>5}  {'Group Size':>12}  {'Cumul Files':>12}")
print("-" * 50)
cumul = 0
included = []
for p, files, size in ranked:
    cumul += len(files)
    print(f"{p:>7}  {len(files):>5}  {size / 1024:>10.1f} KB  {cumul:>12}")
    included.extend(files)
    if cumul >= 130:
        print(f"\nStopped: {cumul} cumulative files across {ranked.index((p, files, size)) + 1} groups.")
        break

print(f"\n--- All files ({len(included)} total) ---")
for f in included:
    print(f)

# Step 5: copy selected PDFs into data/selected_contracts/
if os.path.exists(SELECTED_DIR):
    shutil.rmtree(SELECTED_DIR)
os.makedirs(SELECTED_DIR)
for f in included:
    shutil.copy2(os.path.join(CONTRACTS_DIR, f), os.path.join(SELECTED_DIR, f))
print(f"\nCopied {len(included)} PDFs → {SELECTED_DIR}/")

# Step 6: write selected_contracts.csv with filenames for extract_contracts.py
prefix_for = {f: p for p, files, _ in ranked for f in files}
with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as fh:
    writer = csv.DictWriter(fh, fieldnames=["filename", "group_prefix"])
    writer.writeheader()
    writer.writerows({"filename": f, "group_prefix": prefix_for[f]} for f in included)
print(f"Wrote {len(included)} entries → {OUTPUT_CSV}")
