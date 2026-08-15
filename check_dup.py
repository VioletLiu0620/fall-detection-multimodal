import hashlib
from pathlib import Path
from collections import defaultdict

def file_hash(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()

all_csvs = list(Path("data").glob("**/*.csv"))
hash_to_files = defaultdict(list)
for csv in all_csvs:
    hash_to_files[file_hash(csv)].append(str(csv))

unique = len(hash_to_files)
total = len(all_csvs)
print(f"total files: {total}, unique files: {unique}, duplicates: {total - unique}")

# show some examples of duplicates
dupes = {h: files for h, files in hash_to_files.items() if len(files) > 1}
print(f"files that have duplicates: {len(dupes)}")

# total files: 5440, unique files: 1539, duplicates: 3901
# files that have duplicates: 1538