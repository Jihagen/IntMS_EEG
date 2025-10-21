# backfill_kind_plateau.py
from pathlib import Path
import argparse
import numpy as np
import shutil

def backfill_kind(npy_path: Path, dry_run=False, backup=True):
    try:
        d = np.load(npy_path, allow_pickle=True).item()
    except Exception as e:
        print(f"[SKIP] {npy_path.name}: cannot load ({e})")
        return 0, 0

    cuts = d.get("cuts", None)
    if not cuts or not isinstance(cuts, list):
        print(f"[OK]   {npy_path.name}: no cuts -> nothing to change")
        return 0, 0

    changed = 0
    total = 0
    for i, entry in enumerate(cuts):
        total += 1
        if isinstance(entry, dict):
            if "kind" not in entry:
                entry["kind"] = "plateau"
                changed += 1
        else:
            # If somehow stored as a non-dict (unexpected), try to coerce
            try:
                dd = dict(entry)
                if "kind" not in dd:
                    dd["kind"] = "plateau"
                    cuts[i] = dd
                    changed += 1
            except Exception:
                print(f"[WARN] {npy_path.name}: cuts[{i}] not a dict; left as-is")

    if changed == 0:
        print(f"[OK]   {npy_path.name}: all cuts already have 'kind'")
        return 0, total

    if dry_run:
        print(f"[DRY]  {npy_path.name}: would set 'kind=\"plateau\"' on {changed}/{total} cuts")
        return changed, total

    if backup:
        bak = npy_path.with_suffix(npy_path.suffix + ".bak")
        try:
            shutil.copy2(npy_path, bak)
        except Exception as e:
            print(f"[WARN] {npy_path.name}: backup failed ({e})")

    # atomic write
    tmp = npy_path.with_suffix(".tmp.npy")
    try:
        np.save(tmp, d, allow_pickle=True)
        tmp.replace(npy_path)
        print(f"[SAVE] {npy_path.name}: updated {changed}/{total} cuts with kind='plateau'")
    except Exception as e:
        print(f"[ERR ] {npy_path.name}: write failed ({e})")
        # try to clean tmp
        try: tmp.unlink(missing_ok=True)
        except Exception: pass
        return 0, total

    return changed, total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="Folder/file/glob for *_combined.npy")
    ap.add_argument("--dry-run", action="store_true", help="Only report changes")
    ap.add_argument("--no-backup", action="store_true", help="Do not write .bak files")
    args = ap.parse_args()

    p = Path(args.path)
    files = []
    if p.is_file():
        files = [p]
    elif p.is_dir():
        files = sorted(p.glob("*_combined.npy"))
    else:
        # treat as glob
        files = sorted(Path().glob(args.path))

    if not files:
        print(f"No files matched: {args.path}")
        return

    total_changed = 0
    total_entries = 0
    for f in files:
        ch, tot = backfill_kind(f, dry_run=args.dry_run, backup=not args.no_backup)
        total_changed += ch
        total_entries += tot

    print(f"\nDone. Cuts updated: {total_changed}/{total_entries} across {len(files)} file(s).")

if __name__ == "__main__":
    main()
