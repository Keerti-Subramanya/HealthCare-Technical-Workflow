#!/usr/bin/env python
# -- coding: utf-8 --
"""
CardioProtect Extractor API (Local Llama3)
------------------------------------------
FastAPI wrapper for mapper.py that fills Excel templates
offline using Llama3 reasoning (via Ollama API).
Includes progress + preview support + live progress stream + resume mode + auto cache cleanup.
"""

import sys, os, threading, time, uuid, warnings, json
warnings.filterwarnings("ignore")

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse, StreamingResponse
import pandas as pd
from dotenv import load_dotenv
import mapper

# --- Load environment ---
load_dotenv()

MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3:8b")
OLLAMA_API = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")

app = FastAPI(title="CardioProtect Extractor (Local Llama3)", version="3.4")

# ------------------ GLOBAL STATE ------------------
_STATE = {}      # session_id → sheet_name → DataFrame
_PROGRESS = {}   # session_id → {progress, stage, status}
CACHE_FILE = getattr(mapper, "PARTIAL_SAVE_PATH", "partial_extraction_cache.json")


# ------------------ REQUEST MODEL ------------------

class ExtractRequest(BaseModel):
    # For single PDF mode
    pdf_path: str | None = None

    # For multi-PDF mode
    study_pdf: str | None = None   # folder containing all PDFs

    # Common for both
    template_path: str
    criteria_pdf: str
    session_id: str | None = None
    output_xlsx_path: str | None = None



# ------------------ UTILITIES ------------------
def make_json_safe(obj):
    """Ensure pandas/numpy types are JSON-safe."""
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(i) for i in obj]
    else:
        try:
            import numpy as np
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
        except Exception:
            pass
        return obj


def update_progress(sid: str, pct: int, stage: str, status: str = "running"):
    """Update session progress safely."""
    _PROGRESS[sid] = {
        "progress": pct,
        "stage": stage,
        "status": status,
    }


def cleanup_cache():
    """Remove the partial cache file if it exists."""
    try:
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
            print(f"🧹 Deleted cache file: {CACHE_FILE}")
    except Exception as e:
        print(f"⚠ Cache cleanup failed: {e}")


# Session cache cleanup (per-PDF caches for a session)
def cleanup_session_caches(session_id: str):
    """Delete all partial cache files for the given session under partial_caches/."""
    try:
        cache_dir = getattr(mapper, "PARTIAL_CACHE_DIR", "partial_caches")
        deleted = 0
        if os.path.isdir(cache_dir):
            for name in os.listdir(cache_dir):
                if name.endswith(f"_{session_id}_cache.json"):
                    path = os.path.join(cache_dir, name)
                    try:
                        os.remove(path)
                        deleted += 1
                    except Exception as e:
                        print(f"Warning: failed to delete {path}: {e}")
        print(f"Deleted {deleted} cache file(s) for session {session_id} in {cache_dir}")
    except Exception as e:
        print(f"Session cache cleanup failed for {session_id}: {e}")

# Connect progress callback to mapper
def mapper_progress(status, step, percent):
    """Callback passed into mapper for live updates."""
    for sid, prog in _PROGRESS.items():
        if prog.get("status") in ("running", "starting"):
            _PROGRESS[sid] = {"progress": percent, "stage": step, "status": status}


mapper._update_progress = mapper_progress


# ------------------ ROUTES ------------------
@app.get("/health")
def health():
    """Check service and model status."""
    model_name = getattr(mapper, "MODEL_NAME", MODEL_NAME)
    return {"status": "ok", "model": model_name}


@app.post("/extract")
def extract(req: ExtractRequest):
    """Run field extraction and return preview Excel."""
    sid = req.session_id or str(uuid.uuid4())
    update_progress(sid, 0, "Starting extraction")

    def run_extraction():
        try:
            update_progress(sid, 10, "Reading PDFs")
            preview = mapper.extract_fields(
                study_pdf=req.pdf_path,
                criteria_pdf=req.criteria_pdf,
                template_xlsx=req.template_path,
                session_id=sid
            )

            update_progress(sid, 80, "Formatting extracted data")

            sheet_dfs = {sheet: pd.DataFrame(records) for sheet, records in preview.items()}
            _STATE[sid] = sheet_dfs

            preview_path = f"preview_{sid}.xlsx"
            with pd.ExcelWriter(preview_path, engine="openpyxl") as writer:
                for sheet, df in sheet_dfs.items():
                    try:
                        base_df = pd.read_excel(req.template_path, sheet_name=sheet)
                        cols = list(base_df.columns)
                        df = df.reindex(columns=cols, fill_value="NR")
                    except Exception:
                        pass
                    df.to_excel(writer, index=False, sheet_name=sheet[:31])

            update_progress(sid, 100, "Extraction complete ✅", status="done")

        except ValueError as e:
            msg = str(e)
            if "repaired" in msg.lower():
                update_progress(sid, 70, "🟢 JSON repaired successfully", status="running")
            else:
                update_progress(sid, 0, f"Error: {msg}", status="failed")

        except Exception as e:
            update_progress(sid, 0, f"Error: {str(e)}", status="failed")

    thread = threading.Thread(target=run_extraction)
    thread.start()

    return {
        "session_id": sid,
        "message": "Extraction started. Use /status, /progress/{session_id}, or /live_progress/{session_id} to track progress."
    }



import threading

@app.post("/extract/multi_pdf")
def extract_multi(req: ExtractRequest):
    """Batch extraction for all PDFs in a specified folder (async)."""
    sid = req.session_id or str(uuid.uuid4())
    pdf_dir = req.study_pdf
    criteria = req.criteria_pdf
    template = req.template_path

    if not os.path.isdir(pdf_dir):
        return JSONResponse(status_code=400, content={"error": f"{pdf_dir} is not a valid folder"})

    preview_dir = os.path.join("multi_previews", sid)
    os.makedirs(preview_dir, exist_ok=True)

    def run_batch():
        try:
            from mapper import process_multiple_pdfs
            update_progress(sid, 5, f"📁 Found PDFs — starting batch...", status="running")
            results = process_multiple_pdfs(
                pdf_dir, criteria, template, sid,
                preview_dir=preview_dir,
                auto_merge=True, completeness_threshold=95.0
            )
            update_progress(sid, 100, "✅ Multi-PDF extraction complete", status="saved")
        except Exception as e:
            update_progress(sid, 0, f"❌ Error: {e}", status="failed")

    # Run in background so Swagger returns immediately
    thread = threading.Thread(target=run_batch)
    thread.start()

    return {
    "session_id": sid,
    "status": "started",
    "message": (
        f"✅ Multi-PDF extraction started for session: {sid}\n"
        f"Use these URLs to track progress:\n"
        f"- /progress/{sid}\n"
        f"- /live_progress/{sid}\n"
        f"- /preview/{sid}"
    )
    }




@app.post("/resume/multi_pdf")
def resume_multi(req: ExtractRequest):
    """Batch resume for all existing preview files."""
    import uuid
    sid = req.session_id or str(uuid.uuid4())
    pdf_dir = req.study_pdf
    criteria = req.criteria_pdf
    template = req.template_path
    preview_dir = os.path.join("multi_previews", sid)

    if not os.path.isdir(preview_dir):
        return JSONResponse(status_code=400, content={"error": f"No previews found for session {sid}"})

    try:
        from mapper import resume_multiple_pdfs
        update_progress(sid, 0, "Starting multi-PDF resume…", status="running")
        results = resume_multiple_pdfs(
            pdf_dir, criteria, template, sid,
            preview_dir=preview_dir,
            completeness_threshold=95.0
        )
        update_progress(sid, 100, "✅ Resume pass complete", status="saved")
        return {
            "session_id": sid,
            "preview_dir": os.path.abspath(preview_dir),
            "results": results["results"],
            "final_output": results["final_output"]
        }
    except Exception as e:
        update_progress(sid, 0, f"❌ Resume failed: {e}", status="failed")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/resume")
def resume(req: ExtractRequest):
    """
    Resume extraction only for missing ('NR' or empty) fields using existing cache.
    Automatically updates cache and preview Excel.
    """
    sid = req.session_id or "resume_manual"
    update_progress(sid, 0, "Starting resume process...")

    def run_resume():
        try:
            from mapper import resume_incomplete_fields

            preview_path = req.output_xlsx_path or f"preview_{sid}.xlsx"
            updated_cache = resume_incomplete_fields(
                study_pdf=req.pdf_path,
                criteria_pdf=req.criteria_pdf,
                template_xlsx=req.template_path,
                preview_path=preview_path,
                session_id=sid
            )

            update_progress(sid, 100, "Resume completed ✅", status="done")
            _STATE[sid] = {sheet: pd.DataFrame(records) for sheet, records in updated_cache.items()}

        except Exception as e:
            update_progress(sid, 0, f"Resume failed: {str(e)}", status="failed")

    thread = threading.Thread(target=run_resume)
    thread.start()

    return {
        "session_id": sid,
        "message": "Resume started. Use /progress/{session_id} or /live_progress/{session_id} to track."
    }


@app.post("/resume/{session_id}")
def resume(session_id: str, req: ExtractRequest):
    """
    Resume a previously interrupted extraction using mapper resume mode.
    Re-processes only missing batches from partial_extraction_cache.json,
    then deletes cache automatically if successful.
    """
    update_progress(session_id, 5, "Resuming extraction from cache...")

    def run_resume():
        try:
            update_progress(session_id, 10, "Loading cached batches...")
            preview = mapper.extract_fields(
                study_pdf=req.pdf_path,
                criteria_pdf=req.criteria_pdf,
                template_xlsx=req.template_path,
                session_id=session_id
            )

            update_progress(session_id, 85, "Merging cached + new batches")

            sheet_dfs = {sheet: pd.DataFrame(records) for sheet, records in preview.items()}
            _STATE[session_id] = sheet_dfs

            preview_path = f"preview_resume_{session_id}.xlsx"
            with pd.ExcelWriter(preview_path, engine="openpyxl") as writer:
                for sheet, df in sheet_dfs.items():
                    try:
                        base_df = pd.read_excel(req.template_path, sheet_name=sheet)
                        cols = list(base_df.columns)
                        df = df.reindex(columns=cols, fill_value="NR")
                    except Exception:
                        pass
                    df.to_excel(writer, index=False, sheet_name=sheet[:31])

            # ✅ Cleanup cache after successful resume
            cleanup_session_caches(session_id)

            update_progress(session_id, 100, "Resume complete ✅ (cache cleared)", status="done")

        except Exception as e:
            update_progress(session_id, 0, f"Resume failed: {str(e)}", status="failed")

    thread = threading.Thread(target=run_resume)
    thread.start()

    return {
        "session_id": session_id,
        "message": "Resumed extraction started. Use /live_progress/{session_id} or /preview/{session_id} to track progress."
    }


@app.get("/status")
def status():
    """Return progress for all sessions."""
    return _PROGRESS


@app.get("/progress/{session_id}")
def progress(session_id: str):
    """Return progress for a single session."""
    if session_id not in _PROGRESS:
        return {"session_id": session_id, "status": "not_found"}
    return {"session_id": session_id, **_PROGRESS[session_id]}


@app.get("/live_progress/{session_id}")
def live_progress(session_id: str, interval: float = 2.0):
    """Stream live progress updates via SSE (auto-refresh)."""
    def event_stream():
        last = None
        while True:
            payload = _PROGRESS.get(session_id, {"progress": 0, "stage": "waiting", "status": "waiting"})
            if payload != last:
                payload_with_id = {"session_id": session_id, **payload}
                yield f"data: {json.dumps(payload_with_id)}\n\n"
                last = payload
            if payload.get("status") in ("done", "saved", "failed"):
                break
            time.sleep(interval)
    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/preview/{session_id}")
def preview(session_id: str):
    """Return preview of extracted data (sheet-by-sheet)."""
    if session_id not in _STATE:
        return JSONResponse(status_code=404, content={"error": "Session not found or still running"})

    try:
        sheet_dfs = _STATE[session_id]
        preview_dict = {sheet: df.head(5).to_dict(orient="records") for sheet, df in sheet_dfs.items()}
        return {
            "session_id": session_id,
            "sheets": list(sheet_dfs.keys()),
            "preview_rows_per_sheet": preview_dict,
            "status": _PROGRESS.get(session_id, {"status": "unknown"})
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Preview failed: {str(e)}"})


@app.post("/save")
def save(req: ExtractRequest):
    """Save extracted data to Excel and report completeness."""
    sid = req.session_id
    if sid not in _STATE:
        return JSONResponse(status_code=404, content={"error": "Session not found"})

    output_path = req.output_xlsx_path or f"CardioProtect_Filled_{sid}.xlsx"
    writer = pd.ExcelWriter(output_path, engine="openpyxl")

    for sheet, df in _STATE[sid].items():
        df.to_excel(writer, index=False, sheet_name=sheet[:31])

    writer.close()
    update_progress(sid, 100, "Data saved to Excel", status="saved")

# ✅ Add completeness + logical validity
    from mapper import check_completeness
    overall, details, logical_validity = check_completeness(_STATE[sid])

# ✅ Return richer Swagger response
    return {
        "status": "saved",
        "path": os.path.abspath(output_path),
        "completeness_overall": overall,
        "logical_validity": logical_validity,
        "sheet_wise_completeness": details
    }



@app.get("/")
def root():
    """Root endpoint with all routes listed."""
    return {
        "message": "✅ CardioProtect Extractor API running (Llama3)",
        "docs_url": "http://127.0.0.1:8000/docs",
        "endpoints": [
            "/health",
            "/extract",
            "/resume/{session_id}",
            "/save",
            "/status",
            "/progress/{session_id}",
            "/preview/{session_id}",
            "/live_progress/{session_id}"
        ]
    }


# ------------------ MAIN ------------------
if _name_ == "_main_":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)



