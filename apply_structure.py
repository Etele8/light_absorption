import csv
import shutil
from pathlib import Path
from unittest import skip
from unittest.util import unorderable_list_difference

# ============================================================
# CONFIG
# ============================================================

CONFIG = {
    "csv_path": "file_audit_osszes/filename_audit_all.csv",
    "root_path": "osszes",
    "dry_run": False,
}

# ============================================================
# MAIN
# ============================================================

def main():
    csv_path = Path(CONFIG["csv_path"])
    root = Path(CONFIG["root_path"]) if CONFIG["root_path"] else None

    moved = 0
    skipped = 0

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            replicate_source = row.get("replicate_source")
            processable = row.get("processable")

            # Skip unprocessable files
            if processable != "YES":
                continue

            # Skip files already structured by folder
            if replicate_source == "folder_explicit":
                skipped += 1
                continue

            src = Path(row["full_path"])

            # Target directory is relative; anchor it at current parent
            target_subpath = Path(row["target_subpath"])
            target_dir = src.parent / target_subpath.name if target_subpath.is_absolute() is False else target_subpath

            dst = target_dir / src.name

            # Skip if already in place
            if src.resolve() == dst.resolve():
                skipped += 1
                continue

            if CONFIG["dry_run"]:
                print(f"[DRY RUN] Move: {src} -> {dst}")
            else:
                target_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
                moved += 1

    print("\nRestructuring complete")
    print(f"Moved files  : {moved}")
    print(f"Skipped files, because explicit folders: {skipped}")


if __name__ == "__main__":
    main()
