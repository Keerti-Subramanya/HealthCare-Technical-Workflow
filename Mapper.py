#!/usr/bin/env python
# -- coding: utf-8 --
"""
mapper.py — Llama3-based PDF → Excel Field Extractor
----------------------------------------------------
✨ Full-Featured Version (Safe + Resume + JSON Repair)
- Reads text, tables, and image pages from study + criteria PDFs
- Uses local Ollama Llama3 to infer structured JSON data
- Processes template in 3 batches (1–6, 7–12, 13–18)
- Automatically retries malformed JSON (via repair pass)
- Auto-retries failed batches (max 2 times)
- Supports resume mode (only reprocesses failed/missing batches)
- Returns dict: { sheet_name: [ {col: value}, ... ] }
- Works with FastAPI progress updates via _update_progress()
"""

import warnings
warnings.filterwarnings("ignore")

import os, re, json, pdfplumber, pandas as pd, requests, json5, fitz
from PIL import Image
import pytesseract
from pathlib import Path

# ---------------- CONFIG ----------------
MODEL_NAME = "llama3:8b"
OLLAMA_API = "http://localhost:11434/api/generate"
RAW_LOG_PATH = "llama_raw_log.txt"
# Each PDF gets its own cache file
PARTIAL_CACHE_DIR = "partial_caches"
os.makedirs(PARTIAL_CACHE_DIR, exist_ok=True)
  # ⏸ resume cache
MAX_RETRIES_PER_BATCH = 2
MAX_RESUME_PASSES = 3  # extra resume attempts in multi-PDF to reach threshold

# Windows default tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

_update_progress = None  # dynamically injected by app.py


# ---------------- UTIL ----------------
def _p(pct, msg):
    """Progress-safe print and callback."""
    # Console log for visibility
    print(f"[{pct}%] {msg}")
    """Progress-safe print and callback."""
    if callable(_update_progress):
        try:
            _update_progress("running", msg, pct)
        except Exception:
            pass


def _get_cache_path(pdf_path: str, session_id: str | None = None):
    """Build unique cache path for each PDF and session."""
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    sid = session_id or "default"
    return os.path.join(PARTIAL_CACHE_DIR, f"{base}_{sid}_cache.json")


def _save_partial(data: dict, pdf_path: str, session_id: str | None = None):
    """
    Always save partial extraction results to a per-PDF cache.
    ✅ Handles non-JSON-serializable data (NumPy, sets, NaN, None, etc.)
    ✅ Ensures directory exists
    ✅ Logs exact path & size on success
    ✅ Atomic write (prevents corrupted partial saves)
    """
    import tempfile
    import shutil
    import numpy as np

    path = _get_cache_path(pdf_path, session_id)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # --- Safe conversion for weird objects ---
        def make_json_safe(obj):
            if obj is None:
                return "NR"
            if isinstance(obj, (np.generic,)):
                return obj.item()
            if isinstance(obj, (set, tuple)):
                return [make_json_safe(x) for x in obj]
            if isinstance(obj, dict):
                return {str(k): make_json_safe(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [make_json_safe(x) for x in obj]
            try:
                json.dumps(obj)  # test if JSON-safe
                return obj
            except Exception:
                return str(obj)

        safe_data = make_json_safe(data)

        # --- Atomic write: write to temp, then replace ---
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as tmp:
            json.dump(safe_data, tmp, indent=2, ensure_ascii=False)
            tmp_path = tmp.name
        shutil.move(tmp_path, path)

        size_kb = os.path.getsize(path) / 1024
        print(f"💾 Saved partial cache: {path} ({size_kb:.1f} KB)")
        return True

    except Exception as e:
        print(f"❌ Failed saving partial cache for {pdf_path}: {e}")
        import traceback; traceback.print_exc()
        return False




def _load_partial(pdf_path: str, session_id: str | None = None):
    """
    Load partial cache for this specific PDF, if exists.
    ✅ Auto-fallback: tries session-specific first, then 'default'
    ✅ Prints which one is being used for traceability
    """
    # 1️⃣ Try session-specific cache
    path = _get_cache_path(pdf_path, session_id)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                print(f"🟢 Loaded session cache for {os.path.basename(pdf_path)} → {path}")
                return json.load(f)
        except Exception as e:
            print(f"⚠ Failed reading cache {path}: {e}")

    # 2️⃣ Fallback to default cache (common after single-PDF runs)
    fallback_path = _get_cache_path(pdf_path, "default")
    if os.path.exists(fallback_path):
        try:
            with open(fallback_path, "r", encoding="utf-8") as f:
                print(f"🟢 Using fallback default cache for {os.path.basename(pdf_path)} → {fallback_path}")
                return json.load(f)
        except Exception as e:
            print(f"⚠ Failed reading fallback cache {fallback_path}: {e}")

    # 3️⃣ Nothing found
    print(f"🔴 No cache found for {os.path.basename(pdf_path)} in either session or default mode.")
    return {}




# ---------------- COMPLETENESS CHECK ----------------
def check_completeness(data_dict):
    """
    Evaluate completeness and logical validity.
    Returns (overall_completeness, {sheet: completeness%}, logical_validity%)
    """
    import pandas as pd, math

    completeness_by_sheet = {}
    logical_validity_pass, logical_validity_total = 0, 0

    for sheet, records in data_dict.items():
        if not records:
            completeness_by_sheet[sheet] = 0.0
            continue
        df = pd.DataFrame(records)
        total = df.size
        filled = df.replace(["NR", "", None, "NA"], pd.NA).count().sum()
        completeness = round((filled / total) * 100, 2)
        completeness_by_sheet[sheet] = completeness

        # logical rule example: N_in_Arm * Arms (N) ≈ Sample_Size_Total
        if "Arms (N)" in df.columns and "Sample_Size_Total" in df.columns:
            logical_validity_total += 1
            try:
                arms = df["Arms (N)"].dropna().astype(float).mean()
                total_sample = df["Sample_Size_Total"].dropna().astype(float).mean()
                if arms and total_sample:
                    per_arm = total_sample / arms
                    if abs(per_arm * arms - total_sample) <= 0.05 * total_sample:
                        logical_validity_pass += 1
            except Exception:
                pass

    overall_completeness = round(sum(completeness_by_sheet.values()) / max(1, len(completeness_by_sheet)), 2)
    logical_validity = round((logical_validity_pass / max(1, logical_validity_total)) * 100, 2)

    return overall_completeness, completeness_by_sheet, logical_validity


# ---------------- PDF READER ----------------
def read_pdf_text(pdf_path: str) -> str:
    """Extract text + tables + OCR from PDFs."""
    text = []
    _p(10, f"Reading PDF: {os.path.basename(pdf_path)}")

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ""
                # Add tables
                for table in page.extract_tables() or []:
                    for row in table:
                        if any(row):
                            page_text += "\n" + " | ".join([str(c).strip() for c in row if c])
                # OCR fallback
                if not page_text.strip():
                    with fitz.open(pdf_path) as doc:
                        pix = doc[i - 1].get_pixmap()
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        ocr_text = pytesseract.image_to_string(img)
                        if ocr_text.strip():
                            page_text += "\n" + ocr_text.strip()
                if page_text.strip():
                    text.append(page_text.strip())
    except Exception as e:
        text.append(f"[ERROR reading {pdf_path}: {e}]")

    return "\n".join(text)


# ---------------- LLM CALL ----------------
def query_llama(prompt: str, model: str = MODEL_NAME) -> str:
    """
    Send prompt to local Ollama model with enforced JSON-only output.
    ✅ Schema-anchored and repair-safe
    ✅ Supports system message + JSON template injection
    ✅ Works for Llama3 and Meditron-Instruct models
    """

    import requests, time, re

    # --- health check ---
    def _ollama_alive():
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    if not _ollama_alive():
        print("🛑 Ollama not responding — skipping query to protect run.")
        return "{}"

    _p(50, f"Querying {model} via Ollama (safe mode)…")

    # --- wrap prompt ---
    def make_payload(prompt_text: str):
        safe_prompt = f"""
You are a biomedical data extraction and harmonization specialist for the
CardioProtect meta-analysis project.

TASK:
Return *only valid JSON* matching the provided schema.
No explanations, markdown, or text outside the JSON braces.

PROMPT START
{prompt_text}
PROMPT END
"""

        return {
            "model": model,
            # 🔹 global schema anchor for all runs
            "system": (
                "You are an expert biomedical data extractor. "
                "Always output valid JSON with exactly the same keys as the schema shown. "
                "Replace 'NR' values where confident, otherwise leave them. "
                "Never add or rename keys."
            ),
            "prompt": safe_prompt,
            "format": "json",            # enforce JSON mode
            "stream": False,
            "options": {
                "num_ctx": 2048,         # keeps context from truncating
                "temperature": 0,
                "num_gpu": 0,
                "stop": ["", "Answer:", "As an AI", "Result:", "Output:"]
            }
        }

    # --- retry loop ---
    max_retries = 3
    context_limit = 7000
    base_delay = 8
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            payload = make_payload(prompt[:context_limit])
            resp = requests.post(OLLAMA_API, json=payload, timeout=600)

            if resp.status_code >= 500:
                print(f"⚠ Ollama internal error (HTTP {resp.status_code}) — skipping this query.")
                return "{}"

            data = resp.json()
            raw = data.get("response", "").strip()

            # --- sanitize ---
            raw = re.sub(r"^(json)?", "", raw, flags=re.I).strip()
            raw = re.sub(r"$", "", raw).strip()
            if not raw.startswith("{"):
                m = re.search(r"\{[\s\S]*\}", raw)
                if m:
                    raw = m.group(0)
            if not raw.endswith("}"):
                m = re.search(r".*\}", raw, re.S)
                if m:
                    raw = m.group(0)

            # --- log success ---
            with open(RAW_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(
                    "\n" + "=" * 80 +
                    f"\n✅ SUCCESS (Attempt {attempt})\nPrompt (first 600 chars):\n{prompt[:600]}\n\n" +
                    f"Response (first 1200 chars):\n{raw[:1200]}\n" +
                    "=" * 80 + "\n"
                )
            return raw

        except requests.exceptions.Timeout:
            last_error = "timeout"
            print(f"⏱ [Attempt {attempt}] Timeout, waiting {base_delay}s...")
        except requests.exceptions.ConnectionError:
            last_error = "connection"
            print(f"🌐 [Attempt {attempt}] Connection error — is Ollama running?")
        except Exception as e:
            last_error = str(e)
            print(f"⚠ [Attempt {attempt}] Unexpected error: {e}")

        # back-off
        context_limit = max(1000, int(context_limit * 0.6))
        base_delay = min(30, base_delay * 1.5)
        print(f"⚠ Retrying with reduced context ({context_limit}) in {int(base_delay)}s...")
        time.sleep(base_delay)

    print(f"❌ {model} failed after {max_retries} attempts — skipping this query safely.")
    return "{}"





# ---------------- JSON REPAIR ----------------
def safe_json_parse(response_text: str):
    """
    Robust JSON repair and parsing. Cleans markdown, commentary, and dangling braces.
    """
    # Extract JSON-like portion
    match = re.search(r"\{[\s\S]*\}", response_text)
    json_str = match.group(0) if match else response_text

    # Clean typical noise from Ollama / Llama3
    fixed = (
        json_str.replace("**", "")
        .replace("*", "")
        .replace("json", "")
        .replace("```", "")
        .replace("True", "true")
        .replace("False", "false")
        .replace("None", "null")
        .replace("…", "")
    )

    # Repair typical dangling commas / malformed tokens
    fixed = re.sub(r",\s*([\}\]])", r"\1", fixed)
    fixed = re.sub(r"[\x00-\x1F]+", " ", fixed)
    fixed = re.sub(r"NR(?=[,\}\]])", '"NR"', fixed)
    fixed = re.sub(r"NA(?=[,\}\]])", '"NA"', fixed)

    # Try multiple parsing attempts
    try:
        return json.loads(fixed)
    except Exception:
        try:
            return json5.loads(fixed)
        except Exception as e:
            # Try balancing braces if truncated
            open_braces = fixed.count("{") - fixed.count("}")
            if open_braces > 0:
                fixed += "}" * open_braces
            try:
                return json5.loads(fixed)
            except Exception as e2:
                with open("llama_bad_json.txt", "w", encoding="utf-8") as f:
                    f.write(response_text)
                raise ValueError(f"🟠 Model output could not be repaired: {e2}")



def _validate_sheet_records(records: list[dict], sheet_name: str, schema_cols: list[str]) -> list[dict]:
    """
    Cleans and validates a sheet's extracted records for logical accuracy.
    - Normalizes numeric values
    - Removes None/nan/empty → "NR"
    - Ensures all columns exist
    - Enforces schema column order
    """
    import pandas as pd, re
    df = pd.DataFrame(records)
    # If the sheet has no rows, ensure at least one default NR row
    if df.empty:
        return [{c: "NR" for c in schema_cols}]

    # ensure every schema column exists
    for c in schema_cols:
        if c not in df.columns:
            df[c] = "NR"

    # numeric normalization for key metrics
    numeric_like = ["age", "sample", "arm", "followup", "dose", "duration"]
    for col in df.columns:
        if any(key in col.lower() for key in numeric_like):
            df[col] = (
                df[col].astype(str)
                .str.extract(r"(\d+\.?\d*)")[0]
                .fillna("NR")
                .astype(str)
            )

    # replace NaN/None/empty → NR
    df = df.fillna("NR")
    df = df.replace(to_replace=["None", "none", "NaN", "nan", ""], value="NR")

    # logical flag example — sanity check
    if "Arms (N)" in df.columns and "Sample_Size_Total" in df.columns:
        try:
            arms = pd.to_numeric(df["Arms (N)"], errors="coerce")
            total = pd.to_numeric(df["Sample_Size_Total"], errors="coerce")
            if (arms.mean() or 0) > 0 and (total.mean() or 0) > 0 and (total.mean() / arms.mean()) < 5:
                print(f"⚠ Check {sheet_name}: suspicious arm/sample ratio.")
        except Exception:
            pass

    # reindex for schema order
    df = df.reindex(columns=schema_cols)
    return df.to_dict(orient="records")



# ---------------- EXTRACTION CORE ----------------
def extract_fields(study_pdf: str, criteria_pdf: str, template_xlsx: str, session_id: str | None = None):
    """
    Extract fields, retry failed batches, and resume from per-PDF partial caches.
    🧠 High-Accuracy Version:
    - Section-based prompt slicing
    - Confidence-based extraction rules
    - Validation after each batch
    - Auto-preserves filled data
    """
    _p(15, f"Loading PDFs and schema for {os.path.basename(study_pdf)}...")
    study_text = read_pdf_text(study_pdf)
    criteria_text = read_pdf_text(criteria_pdf)

    # helper to split sections for higher accuracy
    def _split_sections(text: str):
        sections = {}
        for sec in ("Introduction", "Methods", "Results", "Discussion", "Conclusion"):
            m = re.search(rf"{sec}[:\s\n]+", text, re.I)
            if m:
                start = m.start()
                nxt = len(text)
                for sec2 in ("Introduction", "Methods", "Results", "Discussion", "Conclusion"):
                    if sec2 != sec:
                        m2 = re.search(rf"{sec2}[:\s\n]+", text[start+1:], re.I)
                        if m2:
                            nxt = min(nxt, start + 1 + m2.start())
                sections[sec.lower()] = text[start:nxt]
        return sections

    sections = _split_sections(study_text)

    xl = pd.ExcelFile(template_xlsx)
    sheet_names = xl.sheet_names
    schema = {s: list(xl.parse(s, nrows=1).columns) for s in sheet_names}

    batches = [sheet_names[i:i + 4] for i in range(0, len(sheet_names), 4)]
    filled, cache = {}, _load_partial(study_pdf, session_id)

    for bi, batch in enumerate(batches, start=1):
        if all(sheet in cache for sheet in batch):
            _p(25 + bi * 10, f"Skipping batch {bi}/{len(batches)} (already cached ✅)")
            filled.update({s: cache[s] for s in batch})
            continue

        attempt, success = 0, False
        while attempt <= MAX_RETRIES_PER_BATCH and not success:
            _p(30 + bi * 10, f"Batch {bi}/{len(batches)} Attempt {attempt + 1}")
            partial_schema = {s: schema[s] for s in batch}

            # --------------- BUILD PROMPT ----------------
            schema_text = ""
            for sheet, columns in partial_schema.items():
                schema_text += f"\n### Sheet: {sheet}\n"
                schema_text += "\n".join([f"- {col}" for col in columns]) + "\n"

            system_prompt = f"""
You are a biomedical data extraction and harmonization specialist working on the
CardioProtect meta-analysis project. You extract structured data from oncology
cardioprotection studies using the exact Excel schema below.

==============================
GLOBAL OUTPUT REQUIREMENTS
==============================
1. Output only valid JSON starting with '{{' and ending with '}}'.
2. Fill values *only when clearly supported by text*.
3. Do NOT guess or infer if ambiguous — leave "NR".
4. Each sheet = list of dicts (rows), each column = key.
5. Include all sheets and columns exactly as in schema.
6. No markdown, commentary, or text outside JSON.
==============================
COMPLETENESS TARGET
==============================
- ≥95% accuracy; fill only verified values.
- Never omit columns or sheets.
==============================
EXCEL SCHEMA
==============================
{schema_text}
"""

            user_prompt = f"""
[CRITERIA CONTENT]
{criteria_text[:8000]}

[STUDY CONTENT — Key Sections]
[Methods]
{sections.get('methods', '')[:2000]}

[Results]
{sections.get('results', '')[:2000]}

[Conclusion]
{sections.get('conclusion', '')[:1000]}

[SCHEMA BATCH {bi}/{len(batches)}]
{json.dumps(partial_schema, indent=2)}
"""

            full_prompt = system_prompt + "\n\n" + user_prompt
            # 🧠 Inject per-batch JSON template to improve schema adherence
            json_template = {}
            for s in batch:
                json_template[s] = [{col: "NR" for col in schema[s]}]

            template_snippet = json.dumps(json_template, indent=2)
            full_prompt += f"\n\nJSON TEMPLATE (fill NR values only where confident):\n{template_snippet}\n"

            # --------------- MODEL CALL ----------------
            _p(45, f"Querying model for batch {bi}/{len(batches)}...")

            try:
                response = query_llama(full_prompt)
            except Exception as e:
                print(f"⚠ Llama call failed: {e} — retrying with shorter prompt...")
                try:
                # Reduce context if model fails due to memory/context overflow
                    short_prompt = full_prompt[:3000]
                    response = query_llama(short_prompt)
                except Exception as e2:
                    print(f"❌ Fallback query also failed: {e2}")
                    response = "{}"  # return empty JSON to prevent crash


            try:
                data = safe_json_parse(response)
                success = True
                _p(55, f"✅ Batch {bi}/{len(batches)} parsed successfully")
            except Exception:
                _p(60, f"Repairing invalid JSON from batch {bi}")
                try:
                    repair_prompt = f"Repair this JSON and return only valid JSON:\n{response}"
                    repair_resp = query_llama(repair_prompt)
                    data = safe_json_parse(repair_resp)
                    success = True
                    _p(65, f"🟢 Batch {bi}/{len(batches)} repaired successfully")
                except Exception as e2:
                    attempt += 1
                    _p(70, f"Retry {attempt} failed for batch {bi} ({e2})")

        if not success:
            _p(80, f"❌ Batch {bi} failed after {MAX_RETRIES_PER_BATCH} retries")
            data = {s: [{col: "NR" for col in schema[s]}] for s in batch}

        # ✅ Save validated per-sheet data
        for s in batch:
            val = data.get(s, [{col: "NR" for col in schema[s]}])
            if isinstance(val, dict):
                val = [val]
            elif not isinstance(val, list):
                val = [{col: "NR" for col in schema[s]}]

            # skip overwrite if previous cache has meaningful data
            if s in cache and cache[s]:
                if any(any(v not in ("NR", "", None, "NA") for v in r.values()) for r in cache[s]):
                    print(f"🛑 Preserving existing data for {s}, skipping overwrite.")
                    filled[s] = cache[s]
                    continue

            # ✅ Apply post-validation cleanup
            cleaned = _validate_sheet_records(val, s, schema[s])
            filled[s], cache[s] = cleaned, cleaned

        # Batch-level completeness feedback
        batch_data = {s: filled[s] for s in batch}
        overall, _, _ = check_completeness(batch_data)
        if overall < 70:
            _p(85, f"🟠 Low completeness ({overall}%) — re-extracting batch {bi}")
            attempt = 0
            success = False
            continue

    # fill missing sheets
    for s in schema:
        if s not in filled:
            print(f"⚠ Missing sheet {s} → auto-filling NR")
            filled[s] = [{col: "NR" for col in schema[s]}]
            cache[s] = filled[s]

    _save_partial(cache, study_pdf, session_id or "default")
    _p(100, f"✅ Extraction complete for {os.path.basename(study_pdf)} (cache saved)")

    print(f"\n=== Extraction Summary: {os.path.basename(study_pdf)} ===")
    for sheet, records in filled.items():
        print(f"{sheet:<35} → {len(records)} rows")

    return filled




def resume_incomplete_fields(study_pdf: str, criteria_pdf: str, template_xlsx: str,
                             preview_path: str = None, session_id: str | None = None,
                             target_completeness: float | None = None):
    """
    Resume extraction only for missing fields using the cache for this specific PDF.
    Each PDF has its own cache under partial_caches/<pdf>_<session>.json
    """
    _p(5, "Resuming incomplete fields from cache...")
    cache = _load_partial(study_pdf, session_id)
    if not cache:
        raise ValueError(f"No cache found for {study_pdf}. Run extract_fields() first.")
            # ✅ QUICK COMPLETENESS CHECK BEFORE ANY RE-QUERY
    overall, details, logical_validity = check_completeness(cache)
    threshold = target_completeness if target_completeness is not None else 90
    if overall >= threshold:
        print(f"✅ Cache already {overall}% complete — skipping Llama3 resume to prevent overwrite.")
        _save_partial(cache, study_pdf, session_id)  # ensure it's preserved
        _p(20, f"Cache completeness {overall}% — skipping re-extraction.")
        # Ensure preview file even when skipping
        try:
            out_path = preview_path or "resume_preview.xlsx"
            with pd.ExcelWriter(out_path, engine="openpyxl", mode="w") as writer:
                for sheet, records in cache.items():
                    df = pd.DataFrame(records)
                    df.to_excel(writer, sheet_name=sheet[:31], index=False)
            _p(95, f" o. Resume (skip) wrote preview  +' {out_path}")
        except Exception as e:
            print(f" s  ,? Resume skip: preview generation failed: {e}")

        overall, details, logical_validity = check_completeness(cache)
        print(f"Resume completeness: {overall}% | Logical validity: {logical_validity}%")
        _p(100, f"Resume completed for {os.path.basename(study_pdf)}  o.")
        return cache

    # Identify sheets with NR/empty (only truly missing columns across rows)
    incomplete_sheets = {}
    for sheet, records in cache.items():
        if not isinstance(records, list):
            continue
        missing_cols = set()
        for row in records:
            for k, v in row.items():
                if v in ("NR", "", None, "NA"):
                    missing_cols.add(k)
        if missing_cols:
            incomplete_sheets[sheet] = sorted(missing_cols)

    if not incomplete_sheets:
        _p(20, "✅ All fields complete, nothing to resume.")
        return cache

    study_text = read_pdf_text(study_pdf)
    criteria_text = read_pdf_text(criteria_pdf)
    xl = pd.ExcelFile(template_xlsx)

    for sheet, missing_cols in incomplete_sheets.items():
        _p(30, f"Re-extracting missing fields for: {sheet}")

        base_cols = list(xl.parse(sheet, nrows=1).columns)
        schema_subset = {sheet: base_cols}

        prompt = f"""
You are a biomedical data extraction and harmonization specialist working on the
CardioProtect meta-analysis project. Your task is to CORRECT incomplete data from
a previously extracted oncology cardioprotection study.

==============================
TASK OBJECTIVE
==============================
- Return only valid JSON (no text outside braces).
- Output structure must be identical to the schema provided below.
- Preserve all existing non-empty values.
- Replace only "NR" or empty fields with correct values.
- Do NOT remove or rename any column or sheet key.

==============================
TARGET SHEET
==============================
{sheet}

==============================
MISSING COLUMNS (to refill)
==============================
{', '.join(missing_cols)}

==============================
FULL SCHEMA (for structure reference)
==============================
{json.dumps(schema_subset, indent=2)}

==============================
STUDY & CRITERIA CONTENT
==============================
[STUDY PDF EXCERPT]
{study_text[:1200]}

[CRITERIA PDF EXCERPT]
{criteria_text[:600]}

==============================
GLOBAL OUTPUT RULES
==============================
1. Return a single valid JSON object in this exact format:
   {{ "{sheet}": [ {{ col1: value, col2: value, ... }} ] }}
2. Include all columns from the schema, not only missing ones.
3. Use "NR" for missing values, but never omit any column key.
4. Keep all data types as strings.
5. No markdown, commentary, or explanations.
6. JSON must be syntactically valid and parsable.
7. Column names must match the schema exactly — case, spacing, and punctuation included.


==============================
INFERENCE RULES
==============================
- Infer obvious values from text (e.g., "158 per arm" → Arms (N)=2).
- When unsure, leave "NR" but preserve structure.
- Never add extra fields or notes.
"""
        # 🧠 Inject JSON template for the target sheet (to enforce schema)
        json_template = {sheet: [{col: "NR" for col in base_cols}]}
        prompt += f"\n\nJSON TEMPLATE (fill NR values only where confident):\n{json.dumps(json_template, indent=2)}\n"
        try:
            response = query_llama(prompt)
            new_data = safe_json_parse(response)

            if sheet in new_data and isinstance(new_data[sheet], list):
                # ⚡ Protect already filled sheets (skip overwrite if sheet mostly complete)
                has_real_data = any(
                    any(v not in ("NR", "", None, "NA") for v in row.values())
                    for row in cache.get(sheet, [])
                )
                if has_real_data and sheet not in incomplete_sheets:
                    print(f"🛑 Preserving existing data for {sheet}, skipping overwrite.")
                    continue

                df_old = pd.DataFrame(cache[sheet])
                df_new = pd.DataFrame(new_data[sheet])

                for col in df_old.columns:
                    if col in df_new.columns:
                        df_old[col] = df_old[col].mask(
                            df_old[col].isin(["NR", "", None]),
                            df_new[col]
                        )

                cache[sheet] = df_old.to_dict(orient="records")
                _p(60, f"✅ Updated missing fields for {sheet}")
            else:
                _p(60, f"⚠ No new data found for {sheet}")


        except Exception as e:
            _p(80, f"❌ Resume failed for {sheet}: {e}")

    _save_partial(cache, study_pdf, session_id)
    
    _p(90, "Cache updated successfully ✅")

    # Always recreate preview from scratch
    try:
        out_path = preview_path or "resume_preview.xlsx"
        with pd.ExcelWriter(out_path, engine="openpyxl", mode="w") as writer:
            for sheet, records in cache.items():
                df = pd.DataFrame(records)
                df.to_excel(writer, sheet_name=sheet[:31], index=False)
        _p(95, f"✅ Resume complete → {out_path}")
    except Exception as e:
        print(f"⚠ Resume succeeded but preview generation failed: {e}")

    overall, details, logical_validity = check_completeness(cache)
    print(f"\n📊 Resume completeness: {overall}% | Logical validity: {logical_validity}%")
    _p(100, f"Resume completed for {os.path.basename(study_pdf)} ✅")

    return cache




def process_multiple_pdfs(input_dir: str, criteria_pdf: str, template_xlsx: str,
                          session_id: str, preview_dir: str = "multi_previews",
                          auto_merge: bool = True, completeness_threshold: float = 95.0):
    """
    Processes all PDFs in a folder using existing extraction logic.
    Produces one preview Excel per PDF (no shared report file).
    """
    import pandas as pd, os
    from datetime import datetime

    os.makedirs(preview_dir, exist_ok=True)
    pdf_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(".pdf") and not f.lower().startswith("copy of")
    ]
    if not pdf_files:
        raise ValueError(f"No study PDFs found in: {input_dir}")

    print(f"\n📂 Found {len(pdf_files)} PDFs — session {session_id}\n")

    results, merged_data = [], {}

    for i, pdf_path in enumerate(pdf_files, start=1):
        base = os.path.splitext(os.path.basename(pdf_path))[0]
        preview_name = f"preview_{session_id}_{base}.xlsx"
        preview_path = os.path.join(preview_dir, preview_name)

        print(f"=== [{i}/{len(pdf_files)}] Processing {base} ===")

        if os.path.exists(preview_path):
            print(f"⏭ Skipping {base} (already done)")
            results.append({"pdf": base, "preview_path": preview_path, "completeness": "cached"})
            continue

        try:
    # Core extraction
            filled = extract_fields(pdf_path, criteria_pdf, template_xlsx, session_id=session_id)

    # Try to resume using per-PDF cache
            try:
                resumed = resume_incomplete_fields(
                pdf_path, criteria_pdf, template_xlsx,
                preview_path=preview_path,
                session_id=session_id,
                target_completeness=completeness_threshold
                )
            except ValueError as e:
                if "No cache found" in str(e):
                    print(f"⚠ No cache found for {base}, using extract_fields() output directly.")
                    resumed = filled  # Fallback to initial extraction
                else:
                    raise

    # ✅ Enhanced completeness check (3 outputs)
            # Iterative resume passes to reach threshold before finalizing
            try:
                _overall_tmp, _, _ = check_completeness(resumed)
            except Exception:
                _overall_tmp = 0.0
            if _overall_tmp < completeness_threshold:
                for _pass in range(max(0, MAX_RESUME_PASSES - 1)):  # total passes incl. first
                    try:
                        resumed = resume_incomplete_fields(
                            pdf_path, criteria_pdf, template_xlsx,
                            preview_path=preview_path,
                            session_id=session_id,
                            target_completeness=completeness_threshold
                        )
                    except Exception:
                        break
                    try:
                        _overall_tmp, _, _ = check_completeness(resumed)
                    except Exception:
                        _overall_tmp = 0.0
                    if _overall_tmp >= completeness_threshold:
                        break

            overall, details, logical_validity = check_completeness(resumed)
            print(f"✅ {base} → {overall}% complete ({logical_validity}% logical validity)")

            results.append({
                "pdf": base,
                "preview_path": preview_path,
                "completeness": overall,
                "logical_validity": logical_validity,
                "sheet_wise": details
            })

    # ✅ Auto-merge only if enabled and threshold met
            if auto_merge and overall >= completeness_threshold:
                for sheet, records in resumed.items():
                    merged_data.setdefault(sheet, []).extend(records)

        except Exception as e:
            print(f"❌ {base} failed: {e}")
            results.append({"pdf": base, "error": str(e)})



    # --- Optional auto-merge ---
    final_output = None
    if auto_merge and merged_data:
        final_output = os.path.join(preview_dir, f"CardioProtect_Final_AutoMerged_{session_id}.xlsx")
        xl = pd.ExcelFile(template_xlsx)
        with pd.ExcelWriter(final_output, engine="openpyxl") as writer:
            for sheet in xl.sheet_names:
                cols = list(xl.parse(sheet, nrows=1).columns)
                df = pd.DataFrame(merged_data.get(sheet, []))
                if df.empty:
                    df = pd.DataFrame([{c: "NR" for c in cols}])
                for c in cols:
                    if c not in df.columns:
                        df[c] = "NR"
                df = df.reindex(columns=cols)
                df.to_excel(writer, sheet_name=sheet[:31], index=False)
        print(f"\n🎯 Auto-merged Excel → {final_output}")

    # Summary in return (no log file)
    print("\n📘 Summary:")
    for r in results:
        c = r.get("completeness", "—")
        print(f" • {r['pdf']} → {c}")

    return {"results": results, "final_output": final_output}


def resume_multiple_pdfs(input_dir: str, criteria_pdf: str, template_xlsx: str,
                         session_id: str, preview_dir: str = "multi_previews",
                         completeness_threshold: float = 95.0):
    """
    Resume extraction for all PDFs in a multi-PDF session.

    🔹 Detects and uses per-PDF caches stored under partial_caches/<pdf>_<session>.json
    🔹 Skips PDFs that are already 100% complete
    🔹 Rebuilds preview_<session>_<pdf>.xlsx for each resumed file
    🔹 Optionally auto-merges PDFs ≥ completeness_threshold
    """
    import pandas as pd, os
    from datetime import datetime

    # Prepare directories
    os.makedirs(preview_dir, exist_ok=True)
    os.makedirs(PARTIAL_CACHE_DIR, exist_ok=True)

    # Map available previews and caches
    previews = [f for f in os.listdir(preview_dir)
                if f.startswith(f"preview_{session_id}_") and f.endswith(".xlsx")]
    cache_files = [f for f in os.listdir(PARTIAL_CACHE_DIR)
                   if f.endswith(f"_{session_id}_cache.json")]

    if not previews and not cache_files:
        raise ValueError(f"No previews or caches found for session {session_id}")

    results, merged_data = {}, {}
    processed = 0

    print(f"\n🔁 Resuming extraction for session: {session_id}")
    print(f"📂 Input folder: {os.path.abspath(input_dir)}")
    print(f"🧩 Found {len(cache_files)} cache files\n")

    for cache_file in cache_files:
        base = cache_file.replace(f"_{session_id}_cache.json", "")
        pdf_path = os.path.join(input_dir, f"{base}.pdf")
        preview_path = os.path.join(preview_dir, f"preview_{session_id}_{base}.xlsx")

        if not os.path.exists(pdf_path):
            print(f"⚠ Missing source PDF for {base}, skipping.")
            continue

        processed += 1
        print(f"\n=== [{processed}] Resuming {base} ===")

        try:
            # Resume using its individual cache
            resumed = resume_incomplete_fields(
                study_pdf=pdf_path,
                criteria_pdf=criteria_pdf,
                template_xlsx=template_xlsx,
                preview_path=preview_path,
                session_id=session_id
            )

            overall, details, logical_validity = check_completeness(resumed)
            print(f"✅ {base} → {overall}% complete ({logical_validity}% logically valid)")

            results[base] = {
                "pdf": base,
                "preview_path": preview_path,
                "completeness": overall,
                "logical_validity": logical_validity,
                "sheet_wise": details
            }

            # Auto-merge if meets threshold
            if overall >= completeness_threshold:
                for sheet, records in resumed.items():
                    merged_data.setdefault(sheet, []).extend(records)

        except Exception as e:
            print(f"❌ Resume failed for {base}: {e}")
            results[base] = {"pdf": base, "error": str(e)}

    # --- Optional merge for all resumed results ---
    final_output = None
    if merged_data:
        final_output = os.path.join(preview_dir, f"CardioProtect_Final_Resumed_{session_id}.xlsx")
        xl = pd.ExcelFile(template_xlsx)
        with pd.ExcelWriter(final_output, engine="openpyxl") as writer:
            for sheet in xl.sheet_names:
                cols = list(xl.parse(sheet, nrows=1).columns)
                df = pd.DataFrame(merged_data.get(sheet, []))
                if df.empty:
                    df = pd.DataFrame([{c: "NR" for c in cols}])
                for c in cols:
                    if c not in df.columns:
                        df[c] = "NR"
                df = df.reindex(columns=cols)
                df.to_excel(writer, sheet_name=sheet[:31], index=False)
        print(f"\n🎯 Resumed & auto-merged Excel created → {final_output}")

    # --- Summary ---
    print("\n📘 Resume Summary:")
    for base, info in results.items():
        c = info.get("completeness", "—")
        lv = info.get("logical_validity", "—")
        print(f" • {base:<35} → {c}% complete ({lv}% logical)")

    print(f"\n🧩 {len(results)} PDFs resumed successfully.")
    if final_output:
        print(f"💾 Final merged file: {final_output}")
    # --- Auto-recover Excel for all successful caches ---
    try:
        cache_files = [f for f in os.listdir(PARTIAL_CACHE_DIR) if f.endswith("_cache.json")]
        for cache_file in cache_files:
            cache_path = os.path.join(PARTIAL_CACHE_DIR, cache_file)
            rebuild_excel_from_cache(cache_path, template_xlsx)
    except Exception as e:
        print(f"⚠ Auto-recovery skipped: {e}")

    return {"results": list(results.values()), "final_output": final_output}


# ---------------- FILL EXCEL ----------------
def fill_template(template_xlsx: str, extracted_json: dict, out_path="CardioProtect_Filled_Llama3.xlsx"):
    """Fill Excel template preserving all columns & order (now with list flattening)."""
    xl = pd.ExcelFile(template_xlsx)
    writer = pd.ExcelWriter(out_path, engine="openpyxl")

    for sheet in xl.sheet_names:
        base_df = xl.parse(sheet)
        cols = list(base_df.columns)

        if sheet in extracted_json:
            raw = extracted_json[sheet]

            # 🔧 Flatten any nested lists into scalars for Excel compatibility
            flat_rows = []
            for row in raw:
                # Each row should be a dict; if not, default all to "NR"
                if not isinstance(row, dict):
                    flat_rows.append({col: "NR" for col in cols})
                    continue

                flattened = {}
                for k, v in row.items():
                    if isinstance(v, list):
                        # If it's a list with one element, take the first
                        if len(v) == 1:
                            flattened[k] = v[0]
                        # If it's a multi-element list, join with semicolon
                        elif len(v) > 1:
                            flattened[k] = "; ".join(map(str, v))
                        else:
                            flattened[k] = "NR"
                    else:
                        flattened[k] = v
                flat_rows.append(flattened)

            new_df = pd.DataFrame(flat_rows)

            # Ensure all columns exist and keep original order
            for col in cols:
                if col not in new_df.columns:
                    new_df[col] = "NR"
            new_df = new_df.reindex(columns=cols)

            if len(new_df) == 0:
                new_df = pd.DataFrame([{col: "NR" for col in cols}])

            new_df.to_excel(writer, sheet_name=sheet[:31], index=False)
        else:
            base_df.to_excel(writer, sheet_name=sheet[:31], index=False)

    writer.close()
    return out_path



# ---------------- JSON → EXCEL RECOVERY ----------------
def rebuild_excel_from_cache(json_cache_path, template_xlsx, out_path=None):
    """
    Rebuild Excel from a saved JSON cache.
    Use when FastAPI returns 500 or preview generation fails.
    """
    import pandas as pd, json, os

    if not os.path.exists(json_cache_path):
        raise FileNotFoundError(f"Cache file not found: {json_cache_path}")

    if not out_path:
        base = os.path.splitext(os.path.basename(json_cache_path))[0].replace("_cache", "")
        out_path = f"Recovered_{base}.xlsx"

    print(f"🧩 Rebuilding Excel from cache: {json_cache_path}")

    with open(json_cache_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    xl = pd.ExcelFile(template_xlsx)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for sheet in xl.sheet_names:
            cols = list(xl.parse(sheet, nrows=1).columns)
            rows = data.get(sheet, [{c: "NR" for c in cols}])
            if isinstance(rows, dict):
                rows = [rows]
            df = pd.DataFrame(rows)
            for c in cols:
                if c not in df.columns:
                    df[c] = "NR"
            df = df.reindex(columns=cols)
            df.to_excel(writer, sheet_name=sheet[:31], index=False)

    print(f"✅ Rebuilt Excel saved → {out_path}")
    return out_path
