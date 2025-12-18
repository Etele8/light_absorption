import re
import csv
import unicodedata
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple
from collections import defaultdict

# ============================================================
# CONFIG
# ============================================================

CONFIG = {
    "root_path": r"osszes",
    "output_dir": r"file_audit_osszes",
    "dry_run_delete_duplicates": True,
}


# Always ignored (failed / setup)
IGNORE_ALWAYS = {
    "probe", "prob",
    "pr", "pr2", "proba",
    "bekalencse",
    "zoldlevel", "zoldlevell", "zoldlevellel",
}

# Ignore ONLY if no distance is found
IGNORE_IF_NO_DISTANCE = {
    "nap", "napos", "hinaralatt", "elmentanap", "napkisut", "sutanap", "napfeny", "hullam", "r", "_otatlagolassal", "om"
}

# Explicit surface variants ONLY (no fuzzy matching)
SEMANTIC_SURFACE = {
    "fsz",
    "felszin", "felstin", "felsin", "fleszin", "felzsin",
    "felszi", "felszinb", "felsi", "f", "felsz", "feleszin_egyatlag", "felstozin", "felzin" 
}

# Explicit semantic distances (value in CM)
SEMANTIC_DISTANCE_CM = {
    "felcm": 0.5,
    "masfelcm": 1.5,
    "maseflcm": 1.5
}

SEMANTIC_REPLICATE = {
    "vissza": 2, "3meres": 3, "2meres": 2
}


# numbers (unit optional); decimal comma normalized before regex
NUMERIC_RE = re.compile(r"(\d+(?:\.\d+)?)(mm|cm|m)?", re.IGNORECASE)

# compound: 1cm5mm (also works if separated by underscores after tokenization normalization)
COMPOUND_RE = re.compile(r"(\d+(?:\.\d+)?)cm(\d+(?:\.\d+)?)mm", re.IGNORECASE)

# ============================================================
# HUNGARIAN TEXT NUMBER PARSER (restored)
# ============================================================

ONES = {
    "egy": 1, "ket": 2, "ketto": 2, "keto": 2, "harom": 3, "negy": 4, "ot": 5, "öt": 5,
    "hat": 6, "het": 7, "hezt": 7, "nyolc": 8, "kilenc": 9,
}

TENS = {
    "tiz": 10,
    "husz": 20,
    "harminc": 30,
    "negyven": 40,
    "otven": 50,
    "hatvan": 60,
    "hetven": 70,
    "hetnen": 70
}

SPECIAL_PREFIX = {
    "tizen": 10,   # tizenket -> 12
    "tzen": 10,
    "huszon": 20,  # huszonot -> 25
    "hiuszon": 20,
    "hosuon": 20
}

UNITS = ("mm", "cm", "m")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text.lower()


def parse_hungarian_text_distance(stem: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Parses forms like:
      hetesfelcm -> 7.5 cm
      harminchetesfelcm -> 37.5 cm
      tizenketcm -> 12 cm
      huszonotcm -> 25 cm
      otvencm -> 50 cm
    Also supports missing unit: harminchetesfel -> 37.5 (unit None)
    """
    unit = None
    core = stem
    
    core = re.sub(r"[^a-z0-9]+", "", core)  # removes '_' and any other separators


    for u in UNITS:
        if core.endswith(u):
            unit = u
            core = core[: -len(u)]
            break

    half = 0.0
    if core.endswith("esfel"):
        half = 0.5
        core = core[:-5]
    if core.endswith("esfell"):
        half = 0.5
        core = core[:-6]
    elif core.endswith("efel"):
        half = 0.5
        core = core[:-4]
    elif core.endswith("fel"):
        half = 0.5
        core = core[:-3]

    if not core:
        return None, None
    
    
    # --- NEW: handle quarter fractions (negyed, haromnegyed) ---
    QUARTERS = {
        "negyed": 0.25,
        "haromnegyed": 0.75,
    }

    for qword, qval in QUARTERS.items():
        if core == qword:
            return qval + half, unit

        if core.endswith(qword):
            left = core[:-len(qword)]

            if left.endswith("es"):
                left = left[:-2]

            if left in ONES:
                return ONES[left] + qval + half, unit




    # --- NEW: handle "egesz" decimal construction ---
    # e.g. egyegeszhuszonot -> 1.25
    if ("egesz" in core) or ("egesu" in core) or ("egsz" in core):
        if "egesz" in core:
            left, right = core.split("egesz", 1)
        elif "egsz" in core:
            left, right = core.split("egsz", 1)
        else:
            left, right = core.split("egesu", 1)

        left_val = None

        # SPECIAL_PREFIX + ONES (tizenegy, huszonket, etc.)
        for pref, base in SPECIAL_PREFIX.items():
            if left.startswith(pref):
                rest = left[len(pref):]
                if rest in ONES:
                    left_val = base + ONES[rest]
                    break

        # fallback: pure tens or pure ones
        if left_val is None:
            if left in ONES:
                left_val = ONES[left]
            elif left in TENS:
                left_val = TENS[left]


        if left_val is None:
            return None, None

        # parse fractional (hundredths) part
        right_val = None

        # try special prefixes first (huszonot = 25, tizenket = 12, etc.)
        for pref, base in SPECIAL_PREFIX.items():
            if right.startswith(pref):
                rest = right[len(pref):]
                if rest in ONES:
                    right_val = base + ONES[rest]
                    break

        # fallback: tens + ones (e.g. hetvenot = 75)
        if right_val is None:
            for tw, tv in sorted(TENS.items(), key=lambda x: -len(x[0])):
                if right.startswith(tw):
                    rest = right[len(tw):]
                    if rest in ONES:
                        right_val = tv + ONES[rest]
                        break

        # fallback: pure tens or pure ones
        if right_val is None:
            if right in TENS:
                right_val = TENS[right]
            elif right in ONES:
                right_val = ONES[right]


        if right_val is None:
            return None, None

        return left_val + right_val / 100.0 + half, unit


    val = 0
    matched_any = False

    # special prefixes
    for pref, base in SPECIAL_PREFIX.items():
        if core.startswith(pref):
            matched_any = True
            val += base
            rest = core[len(pref):]
            if rest in ONES:
                val += ONES[rest]
                return val + half, unit
            # if no ones after tizen/huszon, it's invalid
            return None, None

    # tens
    for tw, tv in sorted(TENS.items(), key=lambda x: -len(x[0])):
        if core.startswith(tw):
            matched_any = True
            val += tv
            core = core[len(tw):]
            break

    # ones
    if core == "":
        return (val + half, unit) if matched_any else (None, None)

    if core in ONES:
        matched_any = True
        val += ONES[core]
        return val + half, unit

    # pure ones (no tens matched)
    if not matched_any and core in ONES:
        return ONES[core] + half, unit

    return None, None


# ============================================================
# DATA STRUCTURE
# ============================================================

@dataclass
class AuditRecord:
    full_path: str
    folder_path: str
    filename: str
    stem: str

    location: Optional[str]
    location_confidence: str
    
    target_subpath: str

    distance_value_numeric: Optional[float]
    distance_unit_raw: Optional[str]
    distance_confidence: str
    distance_source: Optional[str]

    replicate_id: Optional[int]
    replicate_source: Optional[str]

    processable: str
    failure_reason: Optional[str]


# ============================================================
# CORE PROCESSOR
# ============================================================

def process_file(path: Path) -> Optional[AuditRecord]:
    raw_stem = normalize(path.stem)
    stem = re.sub(r"(\d),(\d)", r"\1.\2", raw_stem)  # decimal comma -> dot
    stem = re.sub(r"(\d)_(\d)", r"\1.\2", stem)
    
    # remove trailing time token like _36msec / _44msec / _250ms / _2sec
    stem = re.sub(
        r"(?:[_\W]+)\d+(?:\.\d+)?(?:msec|ms|sec)\b$",
        "",
        stem,
        flags=re.IGNORECASE
    )
    
    # Half suffix only applies when the WHOLE filename is like "37esfelcm" or "37felmm"
    m_half = re.fullmatch(r"\d+(?:\.\d+)?(esfell|esfel|fel)(mm|cm|m)", stem)
    numeric_half = 0.5 if m_half else 0.0


    # underscore-aware tokenization (so fsz_vissza splits)
    tokens = re.split(r"[_\W]+", stem)

    # ---------- HARD IGNORE ----------
    if any(tok in IGNORE_ALWAYS for tok in tokens):
        return None

    # LOCATION
    # Rule: if filename starts with 2–3 letters immediately followed by a digit (or ends up followed by digits),
    # that prefix is the location code — unless it's "fsz".
    m_loc = re.match(r"^([a-z]{2,4})(?=\d)", stem)  # e.g. za50 -> za, ke150v -> ke
    if m_loc and m_loc.group(1) != "fsz":
        location = m_loc.group(1)
        loc_conf = "FILENAME_PREFIX"
    else:
        # fallback: old token rule (covers cases like "za_50")
        if tokens and re.fullmatch(r"[a-z]{2,3}", tokens[0]) and tokens[0] != "fsz":
            location = tokens[0]
            loc_conf = "FILENAME_CODE"
        else:
            location = path.parent.name
            loc_conf = "FOLDER"


    # ---------- REPLICATE / EXPERIMENT ----------
    replicate_id = None
    replicate_source = None

    # 1. semantic tokens in filename
    for tok in tokens:
        if tok in SEMANTIC_REPLICATE:
            replicate_id = SEMANTIC_REPLICATE[tok]
            replicate_source = "semantic_token"
            break

    # 2. glued "vissza" in filename
    if replicate_id is None and "vissza" in stem:
        replicate_id = 2
        replicate_source = "filename_vissza_glued"

    # 3. suffix-based: ...v means experiment 2
    if replicate_id is None and stem.endswith("v"):
        replicate_id = 2
        replicate_source = "suffix_v"

    # 4. folder-based experiment (explicit list, authoritative only if filename gave nothing)
    if replicate_id is None:
        folder_name = normalize(path.parent.name)

        FOLDER_EXP_1 = {
            "1",
            "1meres",
            "1 meres",
            "1. meres",
            "1.meres",
            "1meres_jomeretekkel",
            "1meres_1cmigjomeretekkel",
        }

        FOLDER_EXP_2 = {
            "2",
            "vissza",
            "2meres",
            "2 meres",
            "2. meres",
            "2meres_kezdesni_elborulva",
            "2meres_jomeretekkel",
        }
        
        FOLDER_EXP_3 = {
            "3meres"
        }

        if folder_name in FOLDER_EXP_1:
            replicate_id = 1
            replicate_source = "folder_explicit"

        elif folder_name in FOLDER_EXP_2:
            replicate_id = 2
            replicate_source = "folder_explicit"
            
        elif folder_name in FOLDER_EXP_3:
            replicate_id = 3
            replicate_source = "folder_explicit"

    # 5. default
    if replicate_id is None:
        replicate_id = 1
        replicate_source = "default_single_experiment"



    # ---------- DISTANCE ----------
    distance_value = None
    distance_unit = None
    distance_conf = "NONE"
    distance_source = None

    # semantic surface (0 cm)
    if any(tok in SEMANTIC_SURFACE for tok in tokens):
        distance_value = 0.0
        distance_unit = "CM"
        distance_conf = "HIGH"
        distance_source = "semantic_surface"

    # semantic fixed distances (felcm/masfelcm)
    if distance_value is None:
        for tok in tokens:
            if tok in SEMANTIC_DISTANCE_CM:
                distance_value = SEMANTIC_DISTANCE_CM[tok]
                distance_unit = "CM"
                distance_conf = "HIGH"
                distance_source = "semantic_distance"
                break

    # mixed form like "2cmhetesfelmm" = 2cm + (hetesfel mm)
    m_mix = re.fullmatch(r"(\d+(?:\.\d+)?)(cm|mm|m)([a-z0-9_]+)", stem)
    if distance_value is None and m_mix:
        left_val = float(m_mix.group(1))
        left_unit = m_mix.group(2).upper()
        right_part = m_mix.group(3)

        hv, hu = parse_hungarian_text_distance(right_part)
        if hv is not None and hu is not None:
            hu = hu.upper()

            # convert both to CM and sum
            def to_cm(val, unit):
                if unit == "CM": return val
                if unit == "MM": return val / 10.0
                if unit == "M":  return val * 100.0
                return None

            total_cm = to_cm(left_val, left_unit) + to_cm(hv, hu)
            distance_value = total_cm
            distance_unit = "CM"
            distance_conf = "HIGH"
            distance_source = "mixed_numeric_plus_hungarian"


    # compound numeric: XcmYmm
    if distance_value is None:
        m = COMPOUND_RE.search(stem)
        if m:
            distance_value = float(m.group(1)) + float(m.group(2)) / 10.0
            distance_unit = "CM"
            distance_conf = "HIGH"
            distance_source = "compound_numeric"

    # numeric (unit optional) — supports: 200, 1.3cm, 1.3, etc.
    if distance_value is None:
        hits = NUMERIC_RE.findall(stem)

        # remove time-ish units from distance consideration (you had msec/sec patterns earlier)
        # since we only allow mm/cm/m here, it's already safe; just keep as-is.

        # filter out empty matches produced by regex on non-numeric tokens
        hits = [(v, u) for v, u in hits if v != ""]

        # If there are multiple numbers (e.g., "50cm_250msec"), we need to decide:
        # - if exactly one of them has a distance unit -> use it
        # - else if only one number exists -> use it (unit may be missing)
        with_unit = [(v, u) for v, u in hits if u]
        unique_numbers = [v for v, _ in hits]

        if len(with_unit) == 1:
            v, u = with_unit[0]
            distance_value = float(v) + numeric_half
            distance_unit = u.upper()
            distance_conf = "HIGH"
            distance_source = "numeric_with_unit"
            
        elif len(with_unit) == 0:
            # accept only if there's exactly one numeric value in the whole stem
            # otherwise ambiguous (e.g., 50_250)
            # de-dup identical captures
            uniq = []
            for v in unique_numbers:
                if v not in uniq:
                    uniq.append(v)
            if len(uniq) == 1:
                distance_value = float(uniq[0]) + numeric_half
                distance_unit = "UNKNOWN"
                distance_conf = "MEDIUM"
                distance_source = "numeric_no_unit"
            elif len(uniq) > 1:
                # might be a distance + time etc. but without units, can't decide safely
                                # keep as failure so you can add rules
                if replicate_source == "folder_explicit":
                    target_subpath = location
                else:
                    target_subpath = f"{location}/{replicate_id}"
                return AuditRecord(
                    full_path=str(path),
                    folder_path=str(path.parent),
                    filename=path.name,
                    stem=stem,
                    location=location,
                    location_confidence=loc_conf,
                    target_subpath=target_subpath,
                    distance_value_numeric=None,
                    distance_unit_raw=None,
                    distance_confidence="NONE",
                    distance_source=None,
                    replicate_id=replicate_id,
                    replicate_source=replicate_source,
                    processable="NO",
                    failure_reason="AMBIGUOUS_NUMERIC",
                )

    # hungarian text distance (with or without unit)
    if distance_value is None:
        hv, hu = parse_hungarian_text_distance(stem)
        if hv is not None:
            distance_value = hv
            distance_unit = hu.upper() if hu else "UNKNOWN"
            distance_conf = "LOW" if hu else "LOW"
            distance_source = "hungarian_text"

    # conditional ignore: nap only if no distance found
    if distance_value is None and any(tok in IGNORE_IF_NO_DISTANCE for tok in tokens):
        return None

    # ---------- FINAL STATUS ----------
    if distance_value is None:
        processable = "NO"
        failure_reason = "NO_DISTANCE"
    elif distance_unit is None or distance_unit == "UNKNOWN":
        processable = "PARTIAL"
        failure_reason = "UNKNOWN_UNIT"
    else:
        processable = "YES"
        failure_reason = None

    if replicate_source == "folder_explicit":
        target_subpath = location
    else:
        target_subpath = f"{location}/{replicate_id}"
        
    return AuditRecord(
        full_path=str(path),
        folder_path=str(path.parent),
        filename=path.name,
        stem=stem,
        location=location,
        location_confidence=loc_conf,
        target_subpath=target_subpath,
        distance_value_numeric=distance_value,
        distance_unit_raw=distance_unit,
        distance_confidence=distance_conf,
        distance_source=distance_source,
        replicate_id=replicate_id,
        replicate_source=replicate_source,
        processable=processable,
        failure_reason=failure_reason,
        
    )


# ============================================================
# UNIT INFERENCE (SECOND PASS) — restored
# ============================================================

def infer_units(records: List[AuditRecord]) -> None:
    by_folder = defaultdict(list)
    for r in records:
        by_folder[r.folder_path].append(r)

    for recs in by_folder.values():
        vals = [r.distance_value_numeric for r in recs if r.distance_value_numeric is not None]
        if not vals:
            continue

        max_val = max(vals)

        # Your invariant:
        # <=2 => meters, <=200 => cm, <=2000 => mm
        if max_val <= 2:
            inferred = "M"
        elif max_val <= 200:
            inferred = "CM"
        elif max_val <= 2000:
            inferred = "MM"
        else:
            continue

        for r in recs:
            if r.distance_unit_raw == "UNKNOWN":
                r.distance_unit_raw = inferred
                r.distance_confidence = "INFERRED"
                if r.processable == "PARTIAL":
                    r.processable = "YES"
                    r.failure_reason = None


# ============================================================
# MAIN
# ============================================================

def main():
    root = Path(CONFIG["root_path"])
    out = Path(CONFIG["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    records: List[AuditRecord] = []

    seen = set()
    deleted = 0

    for file in root.rglob("*.txt"):
        
        norm_stem = normalize(file.stem)
        norm_stem = re.sub(r"(\d),(\d)", r"\1.\2", norm_stem)
        norm_stem = re.sub(r"(\d)_(\d)", r"\1.\2", norm_stem)  # 2_5 -> 2.5

        # remove trailing time token like _36msec / _44msec / _250ms / _2sec
        dedup_stem = re.sub(
            r"(?:[_\W]+)\d+(?:\.\d+)?(?:msec|ms|sec)\b$",
            "",
            norm_stem,
            flags=re.IGNORECASE
        )

        dedup_key = (str(file.parent), dedup_stem)
        if dedup_key in seen:
            # delete duplicates in place
            if CONFIG.get("dry_run_delete_duplicates", True):
                print(f"[DRY RUN] Would delete duplicate: {file}")
            else:
                try:
                    file.unlink()
                    deleted += 1
                except Exception as e:
                    print(f"[WARN] Failed to delete {file}: {e}")
            continue

        seen.add(dedup_key)


        try:
            rec = process_file(file)
            if rec:
                records.append(rec)
        except Exception as e:
            records.append(AuditRecord(
                full_path=str(file),
                folder_path=str(file.parent),
                filename=file.name,
                stem=file.stem,
                location=None,
                location_confidence="EXCEPTION",
                distance_value_numeric=None,
                distance_unit_raw=None,
                distance_confidence="NONE",
                distance_source=None,
                replicate_id=None,
                replicate_source=None,
                processable="NO",
                failure_reason=f"EXCEPTION: {e}",
            ))

    # infer unknown units after initial parse
    infer_units(records)

    all_csv = out / "filename_audit_all.csv"
    fail_csv = out / "filename_audit_failures.csv"

    fields = asdict(records[0]).keys()

    with all_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))

    with fail_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            if r.processable != "YES":
                writer.writerow(asdict(r))

    print(f"Audit complete: {len(records)} files")
    print(f"All files  → {all_csv}")
    print(f"Failures   → {fail_csv}")
    print(f"Deleted duplicates: {deleted}")



if __name__ == "__main__":
    main()
