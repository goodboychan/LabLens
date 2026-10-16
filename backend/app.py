"""LabLens v1.1 FastAPI Application Server.

Serves the same-origin web UI and authenticated REST API implementing:
- Operator session + CSRF + Origin verification (Gate G11)
- Distinct 4-counter catalog & lazy per-episode validation (Gate G01)
- Exact PyAV frame decoding & HTTP Range MP4 video streaming (Gate G03)
- Deterministic T1 (query_signal_window) & T3 (get_frame_bundle) endpoints (Gates G02, G04)
- Bounded 2-call Gemini runtime via Google Cloud ADC (Gates G05, G06, G07, G09, G10)
- Sanitized JSON evidence bundle export (Gate G11)
"""
from __future__ import annotations

import datetime
import json
import os
import secrets
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from urllib.parse import urlparse

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend.gemini_runtime import (
    DEFAULT_MODEL_ID,
    InvestigationManager,
    check_gemini_auth_status,
)
from backend.registry import EpisodeRegistry
from backend.schemas import (
    CatalogResponse,
    EpisodeManifest,
    FinalReport,
    FrameRef,
    GetFrameBundleArgs,
    Investigation,
    InvestigationCreateRequest,
    QuerySignalWindowArgs,
)
from backend.tools import EvidenceToolkit

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"

# Initialize singleton registry, toolkit, and investigation manager
registry = EpisodeRegistry()
toolkit = EvidenceToolkit(registry=registry)
manager = InvestigationManager(registry=registry, toolkit=toolkit)

# Operator session security state (Gate G11)
SESSION_COOKIE_NAME = "lablens_operator_session"
CSRF_COOKIE_NAME = "lablens_csrf_token"
OPERATOR_SESSION_TOKEN = os.environ.get("LABLENS_SESSION_TOKEN") or secrets.token_urlsafe(32)
OPERATOR_CSRF_TOKEN = os.environ.get("LABLENS_CSRF_TOKEN") or secrets.token_urlsafe(24)
ALLOWED_HOSTS = {"localhost", "127.0.0.1", "testserver", "0.0.0.0"}

app = FastAPI(
    title="LabLens — Robot Episode Investigator",
    version="1.1.0",
    description="Evidence-first robotics episode review with bounded 2-call Gemini verification via Google Cloud ADC.",
)

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


def verify_operator_session(
    request: Request,
    x_lablens_session: Optional[str] = Header(default=None),
) -> str:
    """Enforce operator session authentication on protected routes (Gate G11)."""
    cookie_token = request.cookies.get(SESSION_COOKIE_NAME)
    token = x_lablens_session or cookie_token
    if not token or not secrets.compare_digest(token, OPERATOR_SESSION_TOKEN):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: valid operator session cookie or X-LabLens-Session header is required.",
        )
    return token


def verify_mutating_csrf_and_origin(
    request: Request,
    _session: str = Depends(verify_operator_session),
    x_lablens_csrf: Optional[str] = Header(default=None),
) -> None:
    """Enforce Origin/Referer same-origin check and CSRF token on state-changing/model routes (Gate G11)."""
    origin = request.headers.get("origin")
    referer = request.headers.get("referer")

    req_host = (request.url.hostname or "").lower()
    fwd_host = (request.headers.get("x-forwarded-host") or "").split(",")[0].strip().split(":")[0].lower()
    valid_hosts = set(ALLOWED_HOSTS)
    if req_host:
        valid_hosts.add(req_host)
    if fwd_host:
        valid_hosts.add(fwd_host)

    for hdr_val in (origin, referer):
        if hdr_val:
            parsed = urlparse(hdr_val)
            host = (parsed.hostname or "").lower()
            if host not in valid_hosts:
                raise HTTPException(
                    status_code=403,
                    detail=f"Forbidden cross-origin request from '{hdr_val}'.",
                )

    if not x_lablens_csrf or not secrets.compare_digest(x_lablens_csrf, OPERATOR_CSRF_TOKEN):
        raise HTTPException(
            status_code=403,
            detail="Forbidden: missing or invalid X-LabLens-CSRF token.",
        )


@app.get("/", response_class=HTMLResponse)
async def serve_index(response: Response) -> HTMLResponse:
    """Serve the LabLens SPA and initialize operator session & CSRF cookies."""
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>LabLens frontend is initializing...</h1>", status_code=200)
    html_content = index_path.read_text(encoding="utf-8")
    resp = HTMLResponse(content=html_content, status_code=200)
    resp.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=OPERATOR_SESSION_TOKEN,
        httponly=True,
        samesite="lax",
    )
    resp.set_cookie(
        key=CSRF_COOKIE_NAME,
        value=OPERATOR_CSRF_TOKEN,
        httponly=False,
        samesite="lax",
    )
    return resp


@app.get("/api/health")
async def health_check() -> Dict[str, Any]:
    """Cloud Run readiness and health probe (no credentials or session tokens exposed)."""
    cat = registry.get_catalog_response()
    return {
        "status": "ok",
        "service": "lablens",
        "version": "1.1.0",
        "reported_prepared_episodes": cat.counts.reported_prepared,
        "registered_episodes": cat.counts.registered,
        "validated_episodes": cat.counts.validated,
        "analyzed_episodes": cat.counts.analyzed,
        "auth_mode": "adc",
        "model_id": DEFAULT_MODEL_ID,
    }


@app.get("/api/session")
async def get_operator_session(response: Response) -> Dict[str, Any]:
    """Initialize operator session cookies and return CSRF token + ADC status (no secrets exposed)."""
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=OPERATOR_SESSION_TOKEN,
        httponly=True,
        samesite="lax",
    )
    response.set_cookie(
        key=CSRF_COOKIE_NAME,
        value=OPERATOR_CSRF_TOKEN,
        httponly=False,
        samesite="lax",
    )
    adc_info = check_gemini_auth_status()
    return {
        "authenticated": True,
        "session_token": OPERATOR_SESSION_TOKEN,
        "csrf_token": OPERATOR_CSRF_TOKEN,
        "auth_mode": "adc",
        "adc_status": adc_info,
        "model_id": DEFAULT_MODEL_ID,
        "active_episode_id": manager._active_episode_id,
        "active_investigation_id": manager._active_investigation_id,
    }


@app.get("/api/catalog", response_model=CatalogResponse)
async def get_catalog(
    _session: str = Depends(verify_operator_session),
) -> CatalogResponse:
    """Return the episode catalog with the 4 separate counters:
    reported_prepared, registered, validated, analyzed (Gate G01).
    """
    return registry.get_catalog_response()


@app.post("/api/catalog/validate-all", response_model=CatalogResponse)
async def validate_all_episodes(
    _csrf: None = Depends(verify_mutating_csrf_and_origin),
) -> CatalogResponse:
    """Explicit operator action to run source preflight validation across all registered episodes."""
    for ep_id in registry.list_episode_ids():
        registry.validate_episode(ep_id)
    return registry.get_catalog_response()


@app.get("/api/episodes/{episode_id}/manifest", response_model=EpisodeManifest)
async def get_episode_manifest(
    episode_id: str,
    _session: str = Depends(verify_operator_session),
) -> EpisodeManifest:
    """Lazy-validate the selected episode on demand and return its full EpisodeManifest."""
    try:
        manifest = registry.get_manifest(episode_id, validate_if_needed=True)
        manager.set_active_episode(episode_id)
        return manifest
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/episodes/{episode_id}/trajectory")
async def get_episode_trajectory(
    episode_id: str,
    start_sample: Optional[int] = Query(default=None, ge=0),
    end_sample: Optional[int] = Query(default=None, ge=0),
    _session: str = Depends(verify_operator_session),
) -> Dict[str, Any]:
    """Return trajectory CSV rows and eligible state-minus-action difference series for plotting and CSV inspection."""
    try:
        manifest = registry.get_manifest(episode_id, validate_if_needed=True)
        df = registry.get_dataframe(episode_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    total_rows = len(df)
    s_idx = 0 if start_sample is None else max(0, min(start_sample, total_rows - 1))
    e_idx = (total_rows - 1) if end_sample is None else max(s_idx, min(end_sample, total_rows - 1))

    sub_df = df.iloc[s_idx : e_idx + 1]
    columns = list(sub_df.columns)
    series: Dict[str, List[float]] = {}
    for col in columns:
        if col in ("frame_index", "episode_index", "index", "task_index"):
            series[col] = [int(v) for v in sub_df[col].tolist()]
        else:
            series[col] = [round(float(v), 4) for v in sub_df[col].tolist()]

    # Also compute eligible difference series (state.<joint> - action.<joint>)
    diff_series: Dict[str, List[float]] = {}
    for dp in manifest.difference_pairs:
        if dp.eligible and dp.state_column in sub_df.columns and dp.action_column in sub_df.columns:
            d_vals = (sub_df[dp.state_column] - sub_df[dp.action_column]).tolist()
            diff_series[dp.pair_id] = [round(float(v), 4) for v in d_vals]

    return {
        "episode_id": episode_id,
        "episode_revision": manifest.episode_revision,
        "total_rows": total_rows,
        "start_sample": s_idx,
        "end_sample": e_idx,
        "row_count": len(sub_df),
        "primary_time_column": manifest.timing_semantics.primary_time_column,
        "fps": manifest.timing_semantics.nominal_fps,
        "nominal_fps": manifest.timing_semantics.nominal_fps,
        "source_kind": manifest.source_kind,
        "action_semantics": manifest.action_semantics.model_dump(),
        "state_semantics": manifest.state_semantics.model_dump(),
        "difference_pairs": [dp.model_dump() for dp in manifest.difference_pairs],
        "columns": columns,
        "series": series,
        "difference_series": diff_series,
    }


@app.get("/api/episodes/{episode_id}/video/{camera_id}")
async def stream_episode_video(
    episode_id: str,
    camera_id: str,
    request: Request,
    codec: Literal["h264", "av1"] = Query(default="h264"),
    _session: str = Depends(verify_operator_session),
) -> Response:
    """Stream the registered camera video (top.mp4 or wrist.mp4) with HTTP Range support for smooth scrubbing."""
    try:
        manifest = registry.get_manifest(episode_id, validate_if_needed=True)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    cam_spec = next((c for c in manifest.cameras if c.camera_id == camera_id), None)
    if not cam_spec:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found for '{episode_id}'.")

    video_path = Path(
        cam_spec.av1_original_path if (codec == "av1" and cam_spec.av1_original_path) else cam_spec.path
    )
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video file missing: {video_path.name}")

    file_size = video_path.stat().st_size
    range_header = request.headers.get("range")

    if not range_header:
        return FileResponse(
            path=str(video_path),
            media_type="video/mp4",
            headers={"Accept-Ranges": "bytes", "Content-Length": str(file_size)},
        )

    # Parse bytes=start-end
    try:
        range_val = range_header.strip().lower()
        if not range_val.startswith("bytes="):
            raise ValueError("Invalid range unit")
        start_str, end_str = range_val[6:].split("-", 1)
        start = int(start_str) if start_str else 0
        end = int(end_str) if end_str else (file_size - 1)
        start = max(0, min(start, file_size - 1))
        end = max(start, min(end, file_size - 1))
    except Exception:
        return FileResponse(path=str(video_path), media_type="video/mp4")

    chunk_len = end - start + 1

    def iter_chunk():
        with video_path.open("rb") as f:
            f.seek(start)
            remaining = chunk_len
            while remaining > 0:
                data = f.read(min(65536, remaining))
                if not data:
                    break
                remaining -= len(data)
                yield data

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(chunk_len),
        "Cache-Control": "private, max-age=3600",
    }
    return StreamingResponse(iter_chunk(), status_code=206, media_type="video/mp4", headers=headers)


@app.get("/api/episodes/{episode_id}/frames/{camera_id}/{sample_index}")
async def get_decoded_frame_image(
    episode_id: str,
    camera_id: str,
    sample_index: int,
    tier: Literal["overview", "detail"] = Query(default="detail"),
    codec: Literal["h264", "av1"] = Query(default="h264"),
    _session: str = Depends(verify_operator_session),
) -> Response:
    """Decode and return the exact original camera frame JPEG via PyAV along with FrameRef headers (Gate G03)."""
    try:
        frame_ref, img_bytes = toolkit.decode_single_frame(
            episode_id=episode_id,
            camera_id=camera_id,
            sample_index=sample_index,
            resolution_tier=tier,
            use_av1_original=(codec == "av1"),
        )
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    headers = {
        "X-LabLens-Frame-Id": frame_ref.frame_id,
        "X-LabLens-Episode-Revision": frame_ref.episode_revision,
        "X-LabLens-Sample-Index": str(frame_ref.sample_index),
        "X-LabLens-Time-S": str(frame_ref.time_s),
        "X-LabLens-PTS": str(frame_ref.pts if frame_ref.pts is not None else ""),
        "X-LabLens-Frame-SHA256": frame_ref.decoded_frame_sha256,
        "Cache-Control": "private, max-age=3600",
    }
    return Response(content=img_bytes, media_type="image/jpeg", headers=headers)


@app.get("/api/episodes/{episode_id}/frames/{camera_id}/{sample_index}/meta", response_model=FrameRef)
async def get_decoded_frame_meta(
    episode_id: str,
    camera_id: str,
    sample_index: int,
    tier: Literal["overview", "detail"] = Query(default="detail"),
    codec: Literal["h264", "av1"] = Query(default="h264"),
    _session: str = Depends(verify_operator_session),
) -> FrameRef:
    """Return the exact FrameRef provenance metadata for a decoded camera frame."""
    try:
        frame_ref, _ = toolkit.decode_single_frame(
            episode_id=episode_id,
            camera_id=camera_id,
            sample_index=sample_index,
            resolution_tier=tier,
            use_av1_original=(codec == "av1"),
        )
        return frame_ref
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/tools/query-signal-window")
async def api_query_signal_window(
    args: QuerySignalWindowArgs,
    _csrf: None = Depends(verify_mutating_csrf_and_origin),
) -> Dict[str, Any]:
    """Execute deterministic T1 (query_signal_window) directly for interactive verification."""
    try:
        res, receipt = toolkit.execute_query_signal_window(args, investigation_id="operator_tool_check")
        return {"result": res.model_dump(), "receipt": receipt.model_dump()}
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/tools/get-frame-bundle")
async def api_get_frame_bundle(
    args: GetFrameBundleArgs,
    _csrf: None = Depends(verify_mutating_csrf_and_origin),
) -> Dict[str, Any]:
    """Execute deterministic T3 (get_frame_bundle) directly for interactive verification."""
    try:
        res, receipt, _ = toolkit.execute_get_frame_bundle(args, investigation_id="operator_tool_check")
        return {"result": res.model_dump(), "receipt": receipt.model_dump()}
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/episodes/{episode_id}/saved-report", response_model=FinalReport)
async def get_episode_saved_report(
    episode_id: str,
    _session: str = Depends(verify_operator_session),
) -> FinalReport:
    """Return a real-tool-backed Saved Analysis report for the episode, clearly labeled analysis_mode='saved_analysis'."""
    try:
        return manager.get_or_create_saved_report_for_episode(episode_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/investigations", response_model=Investigation)
async def create_investigation_endpoint(
    req: InvestigationCreateRequest,
    _csrf: None = Depends(verify_mutating_csrf_and_origin),
) -> Investigation:
    """Create and authorize a single-episode investigation (requires explicit_authorization=True)."""
    try:
        return manager.create_investigation(req)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/investigations/{investigation_id}/run", response_model=Investigation)
async def run_investigation_endpoint(
    investigation_id: str,
    _csrf: None = Depends(verify_mutating_csrf_and_origin),
) -> Investigation:
    """Execute the bounded 2-call Gemini investigation loop via Google Cloud ADC."""
    try:
        return manager.run_full_investigation(investigation_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/investigations/{investigation_id}/stop", response_model=Investigation)
async def stop_investigation_endpoint(
    investigation_id: str,
    _csrf: None = Depends(verify_mutating_csrf_and_origin),
) -> Investigation:
    """Stop an active investigation so late results never overwrite a newly selected episode (Gate G09)."""
    try:
        return manager.stop_investigation(investigation_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/investigations/{investigation_id}", response_model=Investigation)
async def get_investigation_endpoint(
    investigation_id: str,
    _session: str = Depends(verify_operator_session),
) -> Investigation:
    """Retrieve investigation status, tool receipts, model receipts, and final report."""
    try:
        return manager.get_investigation(investigation_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _strip_local_paths(obj: Any) -> Any:
    """Recursively strip any local host or container filesystem paths so exported JSON never leaks machine paths."""
    if isinstance(obj, dict):
        cleaned: Dict[str, Any] = {}
        for k, v in obj.items():
            if k in ("path", "av1_original_path", "source_path"):
                if isinstance(v, str) and v:
                    cleaned["filename"] = Path(v).name
                continue
            cleaned[k] = _strip_local_paths(v)
        return cleaned
    if isinstance(obj, list):
        return [_strip_local_paths(x) for x in obj]
    if isinstance(obj, str):
        root_fwd = str(PROJECT_ROOT).replace("\\", "/")
        root_bwd = str(PROJECT_ROOT)
        prefixes = (
            f"{root_fwd}/",
            f"{root_bwd}\\",
            root_fwd,
            root_bwd,
            "d:/gdm/",
            "d:\\gdm\\",
            "D:/gdm/",
            "D:\\gdm\\",
            "d:/gdm",
            "d:\\gdm",
            "D:/gdm",
            "D:\\gdm",
            "/app/",
        )
        for prefix in prefixes:
            if prefix and prefix in obj:
                obj = obj.replace(prefix, "")
        return obj
    return obj


def _build_sanitized_export_bundle(
    manifest: EpisodeManifest,
    report: FinalReport,
    investigation: Optional[Investigation] = None,
) -> Dict[str, Any]:
    """Build a sanitized JSON export bundle with zero secrets, cookies, or raw media bytes (Gate G11)."""
    # Strip local host absolute paths from source metadata to avoid leaking private filesystem paths
    sanitized_cameras = [
        {
            "camera_id": c.camera_id,
            "filename": Path(c.path).name,
            "sha256": c.sha256,
            "codec": c.codec,
            "width": c.width,
            "height": c.height,
            "fps": c.fps_nominal,
            "frame_count": c.frame_count,
            "duration_s": round(c.frame_count / c.fps_nominal, 2) if c.fps_nominal else 0.0,
        }
        for c in manifest.cameras
    ]

    raw_bundle = {
        "export_schema": "lablens.export.v1_1",
        "exported_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "auth_mode": "adc",
        "episode_id": manifest.episode_id,
        "episode_revision": manifest.episode_revision,
        "source_kind": manifest.source_kind,
        "task_text": manifest.task_text,
        "policy_or_session_id": manifest.policy_or_session_id,
        "outcome_annotation": manifest.outcome_annotation.model_dump(),
        "action_semantics": manifest.action_semantics.model_dump(),
        "state_semantics": manifest.state_semantics.model_dump(),
        "timing_semantics": manifest.timing_semantics.model_dump(),
        "trajectory_digest": {
            "filename": Path(manifest.trajectory.path).name,
            "sha256": manifest.trajectory.sha256,
            "row_count": manifest.trajectory.row_count,
            "columns": manifest.trajectory.columns,
        },
        "cameras": sanitized_cameras,
        "analysis_mode": report.analysis_mode,
        "processing_status": report.processing_status,
        "annotation_policy": report.annotation_policy,
        "question": report.question,
        "focus_window": {
            "start_sample": report.focus_start_sample,
            "end_sample": report.focus_end_sample,
        },
        "direct_answer": report.direct_answer,
        "computed_facts": [
            cf.model_dump(exclude={"source_ref"}) for cf in report.computed_facts
        ],
        "numeric_findings": [nf.model_dump() for nf in report.numeric_findings],
        "visual_findings": [vf.model_dump() for vf in report.visual_findings],
        "delivered_frames": [
            {
                "evidence_id": fe.evidence_id,
                "camera": fe.camera,
                "sample_index": fe.sample_index,
                "time_s": fe.time_s,
                "pts": fe.frame_ref.pts,
                "source_video_sha256": fe.frame_ref.source_video_sha256,
                "decoded_frame_sha256": fe.frame_ref.decoded_frame_sha256,
                "resolution_tier": fe.frame_ref.resolution_tier,
            }
            for fe in report.frame_evidence
        ],
        "events": [ev.model_dump() for ev in report.events],
        "hypotheses": [h.model_dump() for h in report.hypotheses],
        "unresolved_physical_causes": report.unresolved_physical_causes,
        "recommended_manual_checks": report.recommended_manual_checks,
        "limitations": report.limitations,
        "semantics_and_annotation_note": report.semantics_and_annotation_note,
        "evidence_graph": report.evidence_graph.model_dump(),
        "tool_receipts": [tr.model_dump() for tr in report.tool_receipts],
        "model_receipts": [mr.model_dump() for mr in report.model_receipts],
        "budget_usage": report.budget_usage.model_dump(),
        "export_notice": (
            "Sanitized reference evidence bundle. Excludes credentials, session cookies, "
            "and raw video/image binaries."
        ),
    }
    return _strip_local_paths(raw_bundle)


@app.get("/api/episodes/{episode_id}/export")
async def export_episode_report(
    episode_id: str,
    _session: str = Depends(verify_operator_session),
) -> JSONResponse:
    """Export the active or saved report for an episode as a sanitized JSON evidence bundle (Gate G11)."""
    try:
        manifest = registry.get_manifest(episode_id, validate_if_needed=True)
        # Prefer latest completed live investigation for this episode if present, else saved report
        invs = manager.list_investigations(episode_id=episode_id)
        live_completed = next(
            (i for i in invs if i.status == "completed" and i.final_report), None
        )
        if live_completed and live_completed.final_report:
            report = live_completed.final_report
            bundle = _build_sanitized_export_bundle(manifest, report, live_completed)
        else:
            report = manager.get_or_create_saved_report_for_episode(episode_id)
            bundle = _build_sanitized_export_bundle(manifest, report, None)

        filename = f"lablens_evidence_{episode_id}_{manifest.episode_revision}.json"
        return JSONResponse(
            content=bundle,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/investigations/{investigation_id}/export")
async def export_investigation_report(
    investigation_id: str,
    _session: str = Depends(verify_operator_session),
) -> JSONResponse:
    """Export a specific investigation's evidence bundle (Gate G11)."""
    try:
        inv = manager.get_investigation(investigation_id)
        manifest = registry.get_manifest(inv.episode_id, validate_if_needed=True)
        report = inv.final_report or manager.get_or_create_saved_report_for_episode(inv.episode_id)
        bundle = _build_sanitized_export_bundle(manifest, report, inv)
        filename = f"lablens_investigation_{inv.investigation_id}.json"
        return JSONResponse(
            content=bundle,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
