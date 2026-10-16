"""LabLens v1.1 Deterministic Evidence Tools (T1 & T3).

Implements the exact two P0 read-only tools specified in:
- 01_PRD.md
- 04_DATA_TOOLS_AND_EVIDENCE_CONTRACT.md §6
- 05_ACCEPTANCE_AND_VALIDATION.md (Gates G02, G03, G04)
"""
from __future__ import annotations

import datetime
import hashlib
import io
import time
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import av
import numpy as np
import pandas as pd
from PIL import Image

from backend.registry import EpisodeRegistry, UNITS_BY_JOINT
from backend.schemas import (
    DifferencePairMetrics,
    DifferencePairSpec,
    EpisodeManifest,
    FrameEvidence,
    FrameRef,
    GetFrameBundleArgs,
    GetFrameBundleResult,
    NumericalEvidence,
    QuerySignalWindowArgs,
    QuerySignalWindowResult,
    SignalSummaryMetrics,
    SourceRef,
    ToolReceipt,
)


def compute_evenly_spaced_samples(start_sample: int, end_sample: int, requested_k: int) -> List[int]:
    """Compute K = min(requested_k, N) evenly spaced sample indices in [start_sample, end_sample].
    Includes both endpoints when K > 1; returns [start_sample] when K == 1.
    Removes duplicate indices while preserving ascending order.
    """
    if end_sample < start_sample:
        raise ValueError("end_sample must be >= start_sample")
    interval_count = end_sample - start_sample + 1
    k = min(max(1, requested_k), interval_count)
    if k == 1:
        return [start_sample]
    raw_indices = [
        int(round(start_sample + i * (end_sample - start_sample) / (k - 1)))
        for i in range(k)
    ]
    deduped: List[int] = []
    for idx in raw_indices:
        if not deduped or deduped[-1] != idx:
            deduped.append(idx)
    return deduped


def _infer_unit_for_signal(signal_name: str, manifest: EpisodeManifest) -> str:
    if "." in signal_name:
        _, joint = signal_name.split(".", 1)
        return manifest.state_semantics.units_by_joint.get(joint, UNITS_BY_JOINT.get(joint, "calibrated_units"))
    if signal_name.endswith("_s"):
        return "s"
    return "index"


def _to_relative_posix_path(raw_path: str | Path) -> str:
    """Convert any local host path into a sanitized workspace-relative POSIX path."""
    p = Path(raw_path)
    workspace_root = Path(__file__).resolve().parent.parent
    try:
        return p.resolve().relative_to(workspace_root).as_posix()
    except Exception:
        parts = p.parts
        for idx, part in enumerate(parts):
            if part == "lablens_so101_episodes":
                return "/".join(parts[idx:])
        return p.name


class EvidenceToolkit:
    """Executes T1 (query_signal_window) and T3 (get_frame_bundle) with strict provenance."""

    def __init__(self, registry: EpisodeRegistry, frame_cache_dir: Optional[Path] = None) -> None:
        self.registry = registry
        self.frame_cache_dir = (frame_cache_dir or (registry.storage_root / "frames")).resolve()
        self.frame_cache_dir.mkdir(parents=True, exist_ok=True)

    def execute_query_signal_window(
        self,
        args: QuerySignalWindowArgs,
        investigation_id: str = "adhoc",
    ) -> Tuple[QuerySignalWindowResult, ToolReceipt]:
        """T1: Deterministic full-resolution CSV signal & difference summary over [start_sample, end_sample]."""
        t0 = time.perf_counter()
        started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

        manifest = self.registry.get_manifest(args.episode_id, validate_if_needed=True)
        if args.episode_revision and args.episode_revision != manifest.episode_revision:
            raise ValueError(
                f"Stale episode_revision '{args.episode_revision}' != current '{manifest.episode_revision}'"
            )

        df = self.registry.get_dataframe(args.episode_id)
        total_rows = len(df)
        if args.start_sample < 0 or args.end_sample >= total_rows or args.end_sample < args.start_sample:
            raise ValueError(
                f"Invalid sample bounds [{args.start_sample}, {args.end_sample}] for episode "
                f"'{args.episode_id}' with {total_rows} rows (0..{total_rows - 1})."
            )

        # Validate signals exist
        for sig in args.signals:
            if sig not in df.columns:
                raise ValueError(f"Requested signal '{sig}' does not exist in episode '{args.episode_id}'.")

        window_df = df.iloc[args.start_sample : args.end_sample + 1]
        row_count = len(window_df)
        time_col = manifest.timing_semantics.primary_time_column
        start_time_s = float(window_df[time_col].iloc[0])
        end_time_s = float(window_df[time_col].iloc[-1])

        signal_summaries: List[SignalSummaryMetrics] = []
        numerical_evidence: List[NumericalEvidence] = []

        for sig in args.signals:
            arr = window_df[sig].to_numpy(dtype=float)
            if not np.isfinite(arr).all():
                raise ValueError(f"Non-finite values in signal '{sig}' over [{args.start_sample}, {args.end_sample}].")

            start_val = float(arr[0])
            end_val = float(arr[-1])
            min_val = float(np.min(arr))
            max_val = float(np.max(arr))
            mean_val = float(np.mean(arr))
            delta_val = float(end_val - start_val)

            excursions = np.abs(arr - start_val)
            max_excursion_idx = int(np.argmax(excursions))
            abs_delta_max = float(excursions[max_excursion_idx])
            peak_sample = int(window_df["frame_index"].iloc[max_excursion_idx])
            peak_time_s = float(window_df[time_col].iloc[max_excursion_idx])

            unit = _infer_unit_for_signal(sig, manifest)
            metrics = SignalSummaryMetrics(
                signal=sig,
                unit=unit,
                start_sample=args.start_sample,
                end_sample=args.end_sample,
                sample_count=row_count,
                start_value=round(start_val, 4),
                end_value=round(end_val, 4),
                min_value=round(min_val, 4),
                max_value=round(max_val, 4),
                mean_value=round(mean_val, 4),
                delta=round(delta_val, 4),
                abs_delta_max=round(abs_delta_max, 4),
                peak_sample=peak_sample,
                peak_time_s=round(peak_time_s, 4),
            )
            signal_summaries.append(metrics)

            safe_sig_id = sig.replace(".", "_")
            ev_id = f"ev_num_{args.episode_id}_{safe_sig_id}_{args.start_sample}_{args.end_sample}"
            if sig.startswith("action."):
                caveat = manifest.action_semantics.postprocessing_note
            else:
                caveat = manifest.state_semantics.meaning

            summary_text = (
                f"{sig} [{args.start_sample}..{args.end_sample}]: "
                f"first={metrics.start_value:.2f}, last={metrics.end_value:.2f}, "
                f"min={metrics.min_value:.2f}, max={metrics.max_value:.2f}, "
                f"mean={metrics.mean_value:.2f}, Δ={metrics.delta:+.2f} {unit} "
                f"(peak excursion {metrics.abs_delta_max:.2f} at sample {metrics.peak_sample})."
            )

            numerical_evidence.append(
                NumericalEvidence(
                    evidence_id=ev_id,
                    episode_id=args.episode_id,
                    episode_revision=manifest.episode_revision,
                    kind="signal_summary",
                    label=f"{sig} ({args.start_sample}–{args.end_sample})",
                    summary_text=summary_text,
                    start_sample=args.start_sample,
                    end_sample=args.end_sample,
                    start_time_s=round(start_time_s, 4),
                    end_time_s=round(end_time_s, 4),
                    unit=unit,
                    signal_metrics=metrics,
                    difference_metrics=None,
                    caveat=caveat,
                    source_ref=SourceRef(
                        episode_id=args.episode_id,
                        episode_revision=manifest.episode_revision,
                        artifact_kind="trajectory_csv",
                        path=_to_relative_posix_path(manifest.trajectory.path),
                        sha256=manifest.trajectory.sha256,
                        row_start=args.start_sample,
                        row_end=args.end_sample,
                        columns=["frame_index", time_col, sig],
                        formula=None,
                    ),
                )
            )

        # Evaluate difference pairs
        difference_summaries: List[DifferencePairMetrics] = []
        rejected_pairs: List[str] = []
        eligible_map: Dict[str, DifferencePairSpec] = {
            dp.pair_id: dp for dp in manifest.difference_pairs if dp.eligible
        }
        # Also allow lookup by formula or joint name
        by_joint_map: Dict[str, DifferencePairSpec] = {
            dp.joint: dp for dp in manifest.difference_pairs if dp.eligible
        }

        requested_pair_specs: List[DifferencePairSpec] = []
        explicit_requests: List[str] = []
        if args.difference_pair_id:
            explicit_requests.append(args.difference_pair_id)
        if args.requested_difference_pairs:
            for p in args.requested_difference_pairs:
                if p not in explicit_requests:
                    explicit_requests.append(p)

        if explicit_requests:
            for req_pair in explicit_requests:
                spec = eligible_map.get(req_pair) or by_joint_map.get(req_pair)
                if not spec:
                    rejected_pairs.append(req_pair)
                    raise ValueError(
                        f"Ineligible or incompatible difference pair '{req_pair}' rejected before computation."
                    )
                if spec.state_column not in args.signals or spec.action_column not in args.signals:
                    rejected_pairs.append(req_pair)
                    raise ValueError(
                        f"Difference pair '{spec.pair_id}' requires both '{spec.state_column}' and "
                        f"'{spec.action_column}' in requested signals."
                    )
                requested_pair_specs.append(spec)
        elif args.include_State_action_diff:
            for dp in manifest.difference_pairs:
                if dp.eligible and dp.state_column in args.signals and dp.action_column in args.signals:
                    requested_pair_specs.append(dp)

        for dp in requested_pair_specs:
            s_arr = window_df[dp.state_column].to_numpy(dtype=float)
            a_arr = window_df[dp.action_column].to_numpy(dtype=float)
            diff_arr = s_arr - a_arr
            abs_diff_arr = np.abs(diff_arr)

            mean_signed = float(np.mean(diff_arr))
            mean_abs = float(np.mean(abs_diff_arr))
            max_abs_idx = int(np.argmax(abs_diff_arr))
            max_abs = float(abs_diff_arr[max_abs_idx])
            min_signed = float(np.min(diff_arr))
            max_signed = float(np.max(diff_arr))
            peak_sample = int(window_df["frame_index"].iloc[max_abs_idx])
            peak_time_s = float(window_df[time_col].iloc[max_abs_idx])

            d_metrics = DifferencePairMetrics(
                pair_id=dp.pair_id,
                joint=dp.joint,
                state_column=dp.state_column,
                action_column=dp.action_column,
                formula=dp.formula,
                unit=dp.unit,
                start_sample=args.start_sample,
                end_sample=args.end_sample,
                sample_count=row_count,
                mean_signed_diff=round(mean_signed, 4),
                mean_abs_diff=round(mean_abs, 4),
                max_abs_diff=round(max_abs, 4),
                min_signed_diff=round(min_signed, 4),
                max_signed_diff=round(max_signed, 4),
                peak_sample=peak_sample,
                peak_time_s=round(peak_time_s, 4),
                relationship=dp.relationship,
                caveat=dp.caveat,
            )
            difference_summaries.append(d_metrics)

            ev_id = f"ev_num_{args.episode_id}_{dp.pair_id}_{args.start_sample}_{args.end_sample}"
            summary_text = (
                f"{dp.formula} [{args.start_sample}..{args.end_sample}]: "
                f"mean_diff={d_metrics.mean_signed_diff:+.2f}, "
                f"mean_abs_diff={d_metrics.mean_abs_diff:.2f}, "
                f"max_abs_diff={d_metrics.max_abs_diff:.2f} {dp.unit} "
                f"(min={d_metrics.min_signed_diff:+.2f}, max={d_metrics.max_signed_diff:+.2f}, "
                f"peak at sample {d_metrics.peak_sample} / {d_metrics.peak_time_s:.2f}s)."
            )

            numerical_evidence.append(
                NumericalEvidence(
                    evidence_id=ev_id,
                    episode_id=args.episode_id,
                    episode_revision=manifest.episode_revision,
                    kind="difference_pair",
                    label=f"Δ {dp.joint} (state - action)",
                    summary_text=summary_text,
                    start_sample=args.start_sample,
                    end_sample=args.end_sample,
                    start_time_s=round(start_time_s, 4),
                    end_time_s=round(end_time_s, 4),
                    unit=dp.unit,
                    signal_metrics=None,
                    difference_metrics=d_metrics,
                    caveat=dp.caveat,
                    source_ref=SourceRef(
                        episode_id=args.episode_id,
                        episode_revision=manifest.episode_revision,
                        artifact_kind="trajectory_csv",
                        path=_to_relative_posix_path(manifest.trajectory.path),
                        sha256=manifest.trajectory.sha256,
                        row_start=args.start_sample,
                        row_end=args.end_sample,
                        columns=["frame_index", time_col, dp.state_column, dp.action_column],
                        formula=dp.formula,
                    ),
                )
            )

        result = QuerySignalWindowResult(
            episode_id=args.episode_id,
            episode_revision=manifest.episode_revision,
            start_sample=args.start_sample,
            end_sample=args.end_sample,
            start_time_s=round(start_time_s, 4),
            end_time_s=round(end_time_s, 4),
            row_count=row_count,
            signal_summaries=signal_summaries,
            difference_summaries=difference_summaries,
            rejected_difference_pairs=rejected_pairs,
            numerical_evidence=numerical_evidence,
            semantics_caveat=manifest.action_semantics.postprocessing_note,
        )

        completed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        payload_bytes = len(result.model_dump_json().encode("utf-8"))

        receipt = ToolReceipt(
            receipt_id=f"tr_t1_{investigation_id}_{args.start_sample}_{args.end_sample}",
            investigation_id=investigation_id,
            episode_id=args.episode_id,
            episode_revision=manifest.episode_revision,
            tool_name="query_signal_window",
            tool_version="t1.v1_1",
            status="completed",
            started_at=started_at,
            completed_at=completed_at,
            duration_ms=duration_ms,
            request_args=args.model_dump(),
            source_digests={"trajectory_csv": manifest.trajectory.sha256},
            produced_evidence_ids=[ev.evidence_id for ev in numerical_evidence],
            image_count=0,
            payload_bytes=payload_bytes,
            note=f"Computed {len(signal_summaries)} signal summaries and {len(difference_summaries)} difference pairs across {row_count} full-resolution rows.",
        )

        return result, receipt

    def decode_single_frame(
        self,
        episode_id: str,
        camera_id: str,
        sample_index: int,
        resolution_tier: Literal["overview", "detail"] = "overview",
        use_av1_original: bool = False,
    ) -> Tuple[FrameRef, bytes]:
        """Decode an exact presentation frame from the registered camera video using PyAV."""
        manifest = self.registry.get_manifest(episode_id, validate_if_needed=True)
        total_rows = manifest.trajectory.row_count
        if sample_index < 0 or sample_index >= total_rows:
            raise ValueError(
                f"Requested sample_index {sample_index} out of bounds (0..{total_rows - 1}) for {episode_id}."
            )

        cam_spec = next((c for c in manifest.cameras if c.camera_id == camera_id), None)
        if not cam_spec:
            raise ValueError(f"Camera '{camera_id}' not registered for episode '{episode_id}'.")

        video_path = Path(
            cam_spec.av1_original_path
            if (use_av1_original and cam_spec.av1_original_path)
            else cam_spec.path
        )
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        df = self.registry.get_dataframe(episode_id)
        time_col = manifest.timing_semantics.primary_time_column
        nominal_time_s = float(df[time_col].iloc[sample_index])

        codec_tag = "av1" if (use_av1_original and cam_spec.av1_original_path) else "h264"
        cache_name = (
            f"{episode_id}_{manifest.episode_revision}_{camera_id}_"
            f"f{sample_index:04d}_{resolution_tier}_{codec_tag}.jpg"
        )
        cache_file = self.frame_cache_dir / cache_name
        meta_file = self.frame_cache_dir / f"{cache_name}.json"

        if cache_file.exists() and meta_file.exists():
            img_bytes = cache_file.read_bytes()
            frame_ref = FrameRef.model_validate_json(meta_file.read_text(encoding="utf-8"))
            frame_ref.source_video_path = _to_relative_posix_path(frame_ref.source_video_path)
            return frame_ref, img_bytes

        # Decode exact frame index via PyAV
        decoded_pil: Optional[Image.Image] = None
        decoded_pts: Optional[int] = None
        decoded_time_s: float = nominal_time_s

        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            # Seek near target timestamp if sample_index > 45 to keep decoding fast while remaining exact
            if sample_index > 45 and stream.time_base and stream.average_rate:
                target_sec = max(0.0, (sample_index - 35) / float(stream.average_rate))
                target_pts = int(target_sec / float(stream.time_base))
                container.seek(target_pts, stream=stream, any_frame=False, backward=True)

            for frame in container.decode(video=0):
                if frame.pts is not None and stream.time_base and stream.average_rate:
                    f_idx = int(round(float(frame.pts * stream.time_base) * float(stream.average_rate)))
                else:
                    f_idx = frame.index

                if f_idx == sample_index:
                    decoded_pil = frame.to_image()
                    decoded_pts = frame.pts
                    if frame.time is not None:
                        decoded_time_s = round(float(frame.time), 4)
                    break

        # Fallback sequential decode from start if seek overshot
        if decoded_pil is None:
            with av.open(str(video_path)) as container:
                for idx, frame in enumerate(container.decode(video=0)):
                    if idx == sample_index:
                        decoded_pil = frame.to_image()
                        decoded_pts = frame.pts
                        if frame.time is not None:
                            decoded_time_s = round(float(frame.time), 4)
                        break

        if decoded_pil is None:
            raise RuntimeError(f"Failed to decode sample_index {sample_index} from {video_path}")

        max_edge = 768 if resolution_tier == "overview" else 1280
        w, h = decoded_pil.size
        if max(w, h) > max_edge:
            scale = max_edge / float(max(w, h))
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            decoded_pil = decoded_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
            w, h = decoded_pil.size

        buf = io.BytesIO()
        quality = 85 if resolution_tier == "overview" else 92
        decoded_pil.save(buf, format="JPEG", quality=quality)
        img_bytes = buf.getvalue()
        img_sha = hashlib.sha256(img_bytes).hexdigest()

        frame_id = f"frame_{episode_id}_{camera_id}_{sample_index:04d}"
        image_url = f"/api/episodes/{episode_id}/frames/{camera_id}/{sample_index}?tier={resolution_tier}&codec={codec_tag}"

        frame_ref = FrameRef(
            frame_id=frame_id,
            episode_id=episode_id,
            episode_revision=manifest.episode_revision,
            camera=camera_id,
            camera_id=camera_id,
            sample_index=sample_index,
            presentation_frame_index=sample_index,
            requested_time_s=round(nominal_time_s, 4),
            time_s=round(decoded_time_s, 4),
            pts=decoded_pts,
            timing_map_revision=manifest.timing_semantics.timing_map_revision,
            source_video_path=_to_relative_posix_path(video_path),
            source_video_sha256=cam_spec.sha256,
            decoded_frame_sha256=img_sha,
            width=w,
            height=h,
            resolution_tier=resolution_tier,
            mime_type="image/jpeg",
            byte_size=len(img_bytes),
            image_url=image_url,
        )

        cache_file.write_bytes(img_bytes)
        meta_file.write_text(frame_ref.model_dump_json(indent=2), encoding="utf-8")
        return frame_ref, img_bytes

    def execute_get_frame_bundle(
        self,
        args: GetFrameBundleArgs,
        investigation_id: str = "adhoc",
    ) -> Tuple[GetFrameBundleResult, ToolReceipt, Dict[str, bytes]]:
        """T3: Evenly spaced original camera frame retrieval over [start_sample, end_sample]."""
        t0 = time.perf_counter()
        started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

        manifest = self.registry.get_manifest(args.episode_id, validate_if_needed=True)
        if args.episode_revision and args.episode_revision != manifest.episode_revision:
            raise ValueError(
                f"Stale episode_revision '{args.episode_revision}' != current '{manifest.episode_revision}'"
            )

        total_rows = manifest.trajectory.row_count
        if args.start_sample < 0 or args.end_sample >= total_rows or args.end_sample < args.start_sample:
            raise ValueError(
                f"Invalid sample bounds [{args.start_sample}, {args.end_sample}] for episode "
                f"'{args.episode_id}' with {total_rows} rows (0..{total_rows - 1})."
            )

        # Resolve requested cameras
        if args.camera_ids:
            req_cameras = list(dict.fromkeys(args.camera_ids))
        elif args.camera == "both":
            req_cameras = ["top", "wrist"]
        else:
            req_cameras = [args.camera]

        registered_cam_map = {c.camera_id: c for c in manifest.cameras}
        valid_cameras: List[str] = []
        missing_views: List[str] = []
        for cid in req_cameras:
            if cid in registered_cam_map and Path(registered_cam_map[cid].path).exists():
                valid_cameras.append(cid)
            else:
                missing_views.append(cid)

        if not valid_cameras:
            raise ValueError(f"None of the requested cameras {req_cameras} are available for '{args.episode_id}'.")

        effective_sample_count = min(args.sample_count, max(1, 8 // len(valid_cameras)))
        sampled_indices = compute_evenly_spaced_samples(
            args.start_sample, args.end_sample, effective_sample_count
        )

        frame_evidence_list: List[FrameEvidence] = []
        image_bytes_by_id: Dict[str, bytes] = {}
        total_bytes = 0
        source_digests: Dict[str, str] = {}

        for cid in valid_cameras:
            cam_spec = registered_cam_map[cid]
            source_digests[f"video_{cid}"] = cam_spec.sha256
            for s_idx in sampled_indices:
                frame_ref, img_bytes = self.decode_single_frame(
                    episode_id=args.episode_id,
                    camera_id=cid,
                    sample_index=s_idx,
                    resolution_tier=args.resolution_tier,
                )
                ev_id = f"ev_frame_{args.episode_id}_{cid}_{s_idx:04d}"
                label = f"{cid.upper()} · sample {s_idx} ({frame_ref.time_s:.2f}s)"
                caption = (
                    f"Original {cid} camera frame at sample_index={s_idx}, "
                    f"time={frame_ref.time_s:.2f}s, pts={frame_ref.pts} "
                    f"(window [{args.start_sample}..{args.end_sample}])."
                )
                fev = FrameEvidence(
                    evidence_id=ev_id,
                    episode_id=args.episode_id,
                    episode_revision=manifest.episode_revision,
                    camera=cid,
                    sample_index=s_idx,
                    time_s=frame_ref.time_s,
                    label=label,
                    caption=caption,
                    frame_ref=frame_ref,
                    source_ref=SourceRef(
                        episode_id=args.episode_id,
                        episode_revision=manifest.episode_revision,
                        artifact_kind="video_mp4",
                        path=_to_relative_posix_path(cam_spec.path),
                        sha256=cam_spec.sha256,
                        row_start=s_idx,
                        row_end=s_idx,
                        columns=[cid],
                        formula=None,
                    ),
                )
                frame_evidence_list.append(fev)
                image_bytes_by_id[ev_id] = img_bytes
                total_bytes += len(img_bytes)

        interval_len = args.end_sample - args.start_sample + 1
        unobserved_count = max(0, interval_len - len(sampled_indices))
        sampling_caveat = (
            f"Sampled {len(sampled_indices)} evenly spaced timestamps ({sampled_indices}) "
            f"across {len(valid_cameras)} camera(s) ({', '.join(valid_cameras)}) over "
            f"[{args.start_sample}..{args.end_sample}] ({interval_len} total samples; "
            f"{unobserved_count} intermediate samples not viewed in this bundle). "
            f"Sparse frames do not prove continuous retention or events in unsampled gaps."
        )

        result = GetFrameBundleResult(
            episode_id=args.episode_id,
            episode_revision=manifest.episode_revision,
            camera="both" if len(valid_cameras) > 1 else valid_cameras[0],
            camera_ids=valid_cameras,
            start_sample=args.start_sample,
            end_sample=args.end_sample,
            requested_count_per_camera=args.sample_count,
            delivered_count=len(frame_evidence_list),
            sampled_indices=sampled_indices,
            resolution_tier=args.resolution_tier,
            total_bytes=total_bytes,
            frame_evidence=frame_evidence_list,
            missing_views=missing_views,
            sampling_caveat=sampling_caveat,
        )

        completed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        receipt = ToolReceipt(
            receipt_id=f"tr_t3_{investigation_id}_{args.start_sample}_{args.end_sample}",
            investigation_id=investigation_id,
            episode_id=args.episode_id,
            episode_revision=manifest.episode_revision,
            tool_name="get_frame_bundle",
            tool_version="t3.v1_1",
            status="completed",
            started_at=started_at,
            completed_at=completed_at,
            duration_ms=duration_ms,
            request_args=args.model_dump(),
            source_digests=source_digests,
            produced_evidence_ids=[fe.evidence_id for fe in frame_evidence_list],
            image_count=len(frame_evidence_list),
            payload_bytes=total_bytes,
            note=f"Decoded {len(frame_evidence_list)} original frames ({', '.join(valid_cameras)}) at indices {sampled_indices}.",
        )

        return result, receipt, image_bytes_by_id
