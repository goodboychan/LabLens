"""LabLens v1.1 Episode Registry & Source Preflight Validator.

Manages the 16 prepared SO-101 robot episodes in:
- d:/gdm/lablens_so101_episodes/lablens_so101_episodes
- d:/gdm/lablens_so101_episodes_av1_originals/lablens_so101_episodes

Strictly maintains the four distinct counts:
- reported_prepared (16)
- registered
- validated
- analyzed
"""
from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

import av
import numpy as np
import pandas as pd

from backend.schemas import (
    ActionSemanticsSpec,
    CameraSpec,
    CatalogCounts,
    CatalogEntrySummary,
    CatalogResponse,
    DifferencePairSpec,
    EpisodeCapabilities,
    EpisodeManifest,
    EpisodePermissions,
    EpisodeSourceType,
    OutcomeAnnotation,
    StateSemanticsSpec,
    TimingSemanticsSpec,
    TrajectorySpec,
)

JOINTS: List[str] = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

UNITS_BY_JOINT: Dict[str, str] = {
    "shoulder_pan": "deg (calibrated)",
    "shoulder_lift": "deg (calibrated)",
    "elbow_flex": "deg (calibrated)",
    "wrist_flex": "deg (calibrated)",
    "wrist_roll": "deg (calibrated)",
    "gripper": "gripper_open_pct (0..100 calibrated)",
}


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class EpisodeRegistry:
    """Operator-controlled episode registry and lazy source validator."""

    def __init__(
        self,
        dataset_root: Optional[Path] = None,
        av1_root: Optional[Path] = None,
        storage_root: Optional[Path] = None,
    ) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        default_ds = repo_root / "lablens_so101_episodes" / "lablens_so101_episodes"
        default_av1 = repo_root / "lablens_so101_episodes_av1_originals" / "lablens_so101_episodes"
        self.dataset_root = (dataset_root or default_ds).resolve()
        resolved_av1 = (av1_root or default_av1).resolve()
        self.av1_root = resolved_av1 if resolved_av1.exists() else None
        self.storage_root = (storage_root or (repo_root / "storage")).resolve()
        self.storage_root.mkdir(parents=True, exist_ok=True)

        self.reported_prepared_count: int = 16
        self._manifests: Dict[str, EpisodeManifest] = {}
        self._df_cache: Dict[str, pd.DataFrame] = {}
        self._sha_cache: Dict[str, str] = {}
        self._analyzed_episodes: set[str] = set()
        self._saved_analysis_episodes: set[str] = set()

        self._load_registry()

    def _get_sha256(self, path: Path) -> str:
        key = str(path.resolve())
        if key not in self._sha_cache:
            self._sha_cache[key] = compute_sha256(path)
        return self._sha_cache[key]

    def _load_registry(self) -> None:
        index_path = self.dataset_root / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Dataset index.json not found at {index_path}")

        with index_path.open("r", encoding="utf-8") as f:
            index_data = json.load(f)

        if isinstance(index_data, list):
            episodes_raw = index_data
            self.reported_prepared_count = len(episodes_raw)
        else:
            episodes_raw = index_data.get("episodes", [])
            self.reported_prepared_count = int(index_data.get("episode_count", len(episodes_raw) or 16))

        for item in episodes_raw:
            ep_id = item["episode_id"]
            ep_dir = self.dataset_root / ep_id
            ep_json_path = ep_dir / "episode.json"
            if ep_json_path.exists():
                with ep_json_path.open("r", encoding="utf-8") as ef:
                    ep_meta = json.load(ef)
                # Merge top-level index summary fields if not present
                for k, v in item.items():
                    if k not in ep_meta:
                        ep_meta[k] = v
            else:
                ep_meta = item

            manifest = self._build_manifest_from_meta(ep_id, ep_dir, ep_meta, index_item=item)
            self._manifests[ep_id] = manifest

    def _build_manifest_from_meta(
        self, ep_id: str, ep_dir: Path, meta: dict, index_item: Optional[dict] = None
    ) -> EpisodeManifest:
        idx_source = str((index_item or {}).get("source", "")).lower()
        src_obj = meta.get("source")
        if isinstance(src_obj, dict):
            raw_source = str(src_obj.get("kind") or idx_source or "rollout").lower()
        else:
            raw_source = str(src_obj or idx_source or "rollout").lower()

        if (
            ep_id.startswith("rollout")
            or idx_source == "policy_rollout"
            or raw_source.startswith("policy")
            or "not teleoperation" in raw_source
        ):
            source_type: EpisodeSourceType = "policy_rollout"
        elif (
            ep_id.startswith("teleop")
            or idx_source in ("teleop_demo", "teleoperation")
            or raw_source.startswith("teleop")
        ):
            source_type = "teleoperation"
        else:
            source_type = "unknown"

        # Parse outcome from nested dict or top-level boolean/string
        outcome_Field = meta.get("outcome")
        if isinstance(outcome_Field, dict):
            succ_bool = outcome_Field.get("success")
            raw_outcome = "success" if succ_bool is True else ("failure" if succ_bool is False else "unknown")
            outcome_rule = outcome_Field.get("rule") or outcome_Field.get("note")
            auto_judge = outcome_Field.get("automatic_judge") or {}
            geom_reason = auto_judge.get("reason")
        elif isinstance(outcome_Field, str) and outcome_Field in (
            "success",
            "failure",
            "partial",
            "unknown",
            "not_applicable",
        ):
            raw_outcome = outcome_Field
            outcome_rule = None
            geom_reason = None
        else:
            idx_succ = (index_item or {}).get("success")
            raw_outcome = "success" if idx_succ is True else ("failure" if idx_succ is False else "unknown")
            outcome_rule = None
            geom_reason = None

        if source_type == "teleoperation":
            category_group = "teleop_success"
        elif raw_outcome == "success":
            category_group = "policy_success"
        else:
            category_group = "policy_failure"

        ai_judge_obj = meta.get("ai_judge_end_of_trial")
        if isinstance(ai_judge_obj, dict):
            ai_judge = ai_judge_obj.get("answer")
        else:
            ai_judge = meta.get("ai_judge")

        operator_note = meta.get("operator_note") or meta.get("episode_notes") or (index_item or {}).get("operator_note")
        failure_stage = meta.get("failure_stage") or (geom_reason if raw_outcome == "failure" else None)
        if raw_outcome == "success":
            success_detail = str(operator_note) if operator_note else (meta.get("success_detail") or "—")
        else:
            success_detail = "—"

        notes_parts: List[str] = []
        if operator_note:
            notes_parts.append(str(operator_note))
        if ai_judge:
            notes_parts.append(f"End-of-trial AI judge said {ai_judge}.")
        episode_notes = " ".join(notes_parts) if notes_parts else None

        rationale_parts: List[str] = []
        if outcome_rule:
            rationale_parts.append(f"Rule: {outcome_rule}")
        if operator_note:
            rationale_parts.append(f"Operator note: {operator_note}")
        if ai_judge is not None:
            rationale_parts.append(f"End-of-trial AI judge said {ai_judge}")

        rule_id = outcome_rule or (
            "top_camera_auto_judge_plus_blind_human_rejudge_consensus"
            if source_type == "policy_rollout"
            else "curated_teleoperation_demonstration"
        )

        outcome_annotation = OutcomeAnnotation(
            status=raw_outcome,
            provenance="human",
            annotator_or_rule_id=rule_id,
            criterion=outcome_rule or "Place the cola can upright on the coaster",
            rationale="; ".join(rationale_parts) if rationale_parts else "Supplied dataset annotation",
            success_Detail=success_detail,
            failure_stage=failure_stage,
            episode_notes=episode_notes,
            ai_judge_verdict=str(ai_judge) if ai_judge is not None else None,
            ground_truth_verdict=raw_outcome.upper(),
            disclosed_to_model=True,
        )

        # Trajectory metadata
        csv_path = (ep_dir / "trajectory.csv").resolve()
        traj_meta = meta.get("trajectory") if isinstance(meta.get("trajectory"), dict) else {}
        sync_meta = meta.get("sync") if isinstance(meta.get("sync"), dict) else {}
        row_count = int(
            traj_meta.get("rows")
            or sync_meta.get("trajectory_rows")
            or meta.get("frames")
            or 0
        )
        if csv_path.exists():
            with csv_path.open("r", encoding="utf-8") as cf:
                header_line = cf.readline().strip()
            columns = [c.strip() for c in header_line.split(",") if c.strip()]
            size_bytes = csv_path.stat().st_size
            csv_sha = self._get_sha256(csv_path)
        else:
            columns = []
            size_bytes = 0
            csv_sha = ""

        time_cols = [c for c in columns if c in ("t_wall_s", "t_video_s", "t_s")]
        state_cols = [c for c in columns if c.startswith("state.")]
        action_cols = [c for c in columns if c.startswith("action.")]

        trajectory_spec = TrajectorySpec(
            path=str(csv_path),
            format="csv",
            row_count=row_count,
            columns=columns,
            time_columns=time_cols,
            state_columns=state_cols,
            action_columns=action_cols,
            sha256=csv_sha,
            size_bytes=size_bytes,
        )

        # Cameras
        cameras: List[CameraSpec] = []
        for cam_id, role_desc, default_res in [
            ("top", "Overhead workspace camera (640x480 @ 30 fps)", (640, 480)),
            ("wrist", "End-effector wrist camera (320x200 @ 30 fps)", (320, 200)),
        ]:
            vid_path = (ep_dir / "video" / f"{cam_id}.mp4").resolve()
            av1_path: Optional[Path] = None
            if self.av1_root:
                candidate_av1 = (
                    self.av1_root / ep_id / "video" / f"{cam_id}.mp4"
                ).resolve()
                if not candidate_av1.exists():
                    candidate_av1 = (
                        self.av1_root / ep_id / "video" / f"{cam_id}_av1.mp4"
                    ).resolve()
                if candidate_av1.exists():
                    av1_path = candidate_av1

            vid_sha = self._get_sha256(vid_path) if vid_path.exists() else ""
            vid_size = vid_path.stat().st_size if vid_path.exists() else 0

            cameras.append(
                CameraSpec(
                    camera_id=cam_id,
                    role=role_desc,
                    path=str(vid_path),
                    av1_original_path=str(av1_path) if av1_path else None,
                    codec="h264",
                    av1_codec="av1" if av1_path else None,
                    width=int(default_res[0]),
                    height=int(default_res[1]),
                    fps_nominal=float(meta.get("fps", 30.0)),
                    frame_count=row_count,
                    sha256=vid_sha,
                    size_bytes=vid_size,
                )
            )

        # State, Action, Timing semantics
        state_semantics = StateSemanticsSpec(
            kind="follower_measured_joint_state",
            meaning=(
                "Measured follower arm joint positions read from SO-101 servos at each "
                "sample index in calibrated joint degrees (and 0..100 open scale for gripper)."
            ),
            units_by_joint=UNITS_BY_JOINT,
        )

        if source_type == "policy_rollout":
            action_semantics = ActionSemanticsSpec(
                kind="issued_robot_command",
                relationship="policy_output_After_chunk_blending_clipped_to_motor_limits",
                is_leader_command=False,
                postprocessing_note=(
                    "Policy-issued action values after temporal chunk blending and motor limit "
                    "clipping. These are NOT human teleoperation leader arm commands."
                ),
                meaning=(
                    "Autonomous policy action command issued to follower servos at the logged "
                    "control step (not a leader arm position)."
                ),
            )
            primary_time_col = "t_video_s" if "t_video_s" in time_cols else "t_wall_s"
            secondary_time_col = "t_wall_s" if primary_time_col == "t_video_s" else None
            has_wall = "t_wall_s" in time_cols
            alignment_note = (
                "Row index frame_index (0..N-1) aligns 1:1 with decoded video frame index; "
                "t_video_s is nominal 30 fps video time and t_wall_s is logger wall-clock elapsed seconds. "
                "physical_sync_uncertainty_ms is null (unmeasured)."
            )
        else:
            action_semantics = ActionSemanticsSpec(
                kind="leader_command",
                relationship="leader_target",
                is_leader_command=True,
                postprocessing_note=(
                    "Human teleoperation leader arm target position recorded at 30 Hz during demonstration."
                ),
                meaning=(
                    "Teleoperation leader arm joint target command sent to the follower arm."
                ),
            )
            primary_time_col = "t_s" if "t_s" in time_cols else (time_cols[0] if time_cols else "t_s")
            secondary_time_col = None
            has_wall = False
            alignment_note = (
                "Row index frame_index (0..N-1) aligns 1:1 with decoded video frame index; "
                "t_s is nominal 30 Hz demonstration timestamp. "
                "physical_sync_uncertainty_ms is null (unmeasured)."
            )

        timing_semantics = TimingSemanticsSpec(
            nominal_fps=float(meta.get("fps", 30.0)),
            primary_index_column="frame_index",
            primary_time_column=primary_time_col,
            secondary_time_column=secondary_time_col,
            has_wall_clock=has_wall,
            has_video_pts=True,
            physical_sync_uncertainty_ms=None,
            timing_map_revision="tmap_v1_1_1to1",
            alignment_note=alignment_note,
        )

        difference_pairs: List[DifferencePairSpec] = []
        for joint in JOINTS:
            sc = f"state.{joint}"
            ac = f"action.{joint}"
            if sc in state_cols and ac in action_cols:
                if source_type == "policy_rollout":
                    caveat = (
                        f"Signed difference ({sc} - {ac}) compares measured follower joint state "
                        f"against the policy's issued command (not a leader arm). A nonzero gap "
                        f"quantifies recorded state-vs-action discrepancy in {UNITS_BY_JOINT[joint]} "
                        f"and does NOT prove motor stall, slip, torque saturation, calibration offset, "
                        f"or external contact."
                    )
                else:
                    caveat = (
                        f"Signed difference ({sc} - {ac}) compares measured follower joint state "
                        f"against the teleoperation leader target in {UNITS_BY_JOINT[joint]}. "
                        f"A nonzero gap indicates recorded tracking difference, not a confirmed physical fault."
                    )
                difference_pairs.append(
                    DifferencePairSpec(
                        pair_id=f"diff_{joint}",
                        joint=joint,
                        state_column=sc,
                        action_column=ac,
                        formula=f"{sc} - {ac}",
                        unit=UNITS_BY_JOINT[joint],
                        eligible=True,
                        relationship=action_semantics.relationship,
                        caveat=caveat,
                    )
                )

        # Immutable episode revision derived from source digests + semantics version
        rev_hasher = hashlib.sha256()
        rev_hasher.update(ep_id.encode("utf-8"))
        rev_hasher.update(csv_sha.encode("utf-8"))
        for c in cameras:
            rev_hasher.update(c.sha256.encode("utf-8"))
        rev_hasher.update(action_semantics.kind.encode("utf-8"))
        episode_revision = f"rev_{rev_hasher.hexdigest()[:12]}"

        # Display title
        ckpt_val = meta.get("engine") or meta.get("checkpoint")
        if source_type == "teleoperation":
            display_title = f"Teleop Demo #{ep_id.split('_')[-1]} (Success)"
        else:
            short_code = ep_id.split("_")[1]
            status_cap = "Success" if raw_outcome == "success" else "Failure"
            ckpt = ckpt_val or "policy"
            display_title = f"Rollout {short_code} · {ckpt} ({status_cap})"

        compatible = [
            "teleop_demo_episode_000001",
            "teleop_demo_episode_000047",
            "rollout_T1001_1014_213934",
        ]
        compatible = [c for c in compatible if c != ep_id]

        task_str = meta.get("task", "Place the cup on the coaster")
        capabilities = EpisodeCapabilities(
            permitted_signals=state_cols + action_cols,
            permitted_difference_pairs=[dp.pair_id for dp in difference_pairs],
            available_views=[c.camera_id for c in cameras if c.sha256],
            missing_information=[
                "No motor current / torque telemetry",
                "No tactile / contact / force-torque sensors",
                "No independent 3D object pose or cup-slip sensor",
                "physical_sync_uncertainty_ms is unmeasured (null)",
            ],
        )

        return EpisodeManifest(
            episode_id=ep_id,
            episode_revision=episode_revision,
            display_title=display_title,
            category_group=category_group,
            source_type=source_type,
            source_kind=source_type,
            readiness="registered",
            task_label=task_str,
            task_text=task_str,
            robot_id=meta.get("robot_type", "so101_follower"),
            policy_or_session_id=ckpt_val or ep_id,
            checkpoint_label=ckpt_val,
            checkpoint_note=meta.get("checkpoint_note"),
            outcome_annotation=outcome_annotation,
            cameras=cameras,
            trajectory=trajectory_spec,
            state_semantics=state_semantics,
            action_semantics=action_semantics,
            timing_semantics=timing_semantics,
            difference_pairs=difference_pairs,
            capabilities=capabilities,
            permissions=EpisodePermissions(),
            compatible_comparators=compatible,
            validation_status="unvalidated",
            validation_errors=[],
            has_saved_analysis=ep_id in self._saved_analysis_episodes,
            is_analyzed=ep_id in self._analyzed_episodes,
        )

    def list_episode_ids(self) -> List[str]:
        return list(self._manifests.keys())

    def get_manifest(self, episode_id: str, validate_if_needed: bool = True) -> EpisodeManifest:
        if episode_id not in self._manifests:
            raise KeyError(f"Unknown episode_id '{episode_id}' — not registered in catalog.")
        manifest = self._manifests[episode_id]
        if validate_if_needed and manifest.validation_status == "unvalidated":
            self.validate_episode(episode_id)
        manifest.has_saved_analysis = episode_id in self._saved_analysis_episodes
        manifest.is_analyzed = episode_id in self._analyzed_episodes
        return manifest

    def validate_episode(self, episode_id: str) -> EpisodeManifest:
        """Lazy per-episode source preflight validator (Gate G01, G02, G03)."""
        if episode_id not in self._manifests:
            raise KeyError(f"Unknown episode_id '{episode_id}'")
        manifest = self._manifests[episode_id]
        errors: List[str] = []

        # 1. Validate trajectory CSV existence, readability, monotonic index, finite numbers
        csv_path = Path(manifest.trajectory.path)
        if not csv_path.exists():
            errors.append(f"Trajectory file missing: {csv_path}")
        else:
            try:
                df = self.get_dataframe(episode_id)
                if len(df) == 0:
                    errors.append("Trajectory CSV is empty.")
                elif len(df) != manifest.trajectory.row_count:
                    errors.append(
                        f"Row count mismatch: manifest={manifest.trajectory.row_count}, actual={len(df)}"
                    )
                if "frame_index" not in df.columns:
                    errors.append("Missing required 'frame_index' column.")
                else:
                    idx_series = df["frame_index"]
                    if not idx_series.is_monotonic_increasing:
                        errors.append("frame_index is not monotonic increasing.")
                    if int(idx_series.iloc[0]) != 0 or int(idx_series.iloc[-1]) != len(df) - 1:
                        errors.append("frame_index does not form contiguous 0..N-1 sequence.")

                # Check time monotonicity
                t_col = manifest.timing_semantics.primary_time_column
                if t_col in df.columns and not df[t_col].is_monotonic_increasing:
                    errors.append(f"Primary time column '{t_col}' is not monotonic increasing.")

                # Check finite values on state and action columns
                num_cols = manifest.trajectory.state_columns + manifest.trajectory.action_columns
                if not manifest.trajectory.state_columns:
                    errors.append("No state.* columns found.")
                if not manifest.trajectory.action_columns:
                    errors.append("No action.* columns found.")
                if num_cols:
                    vals = df[num_cols].to_numpy(dtype=float)
                    if not np.isfinite(vals).all():
                        errors.append("Non-finite (NaN/Inf) values detected in trajectory signals.")
            except Exception as exc:
                errors.append(f"Failed to parse trajectory CSV: {exc}")

        # 2. Validate camera MP4 streams via PyAV
        for cam in manifest.cameras:
            vid_path = Path(cam.path)
            if not vid_path.exists():
                errors.append(f"Camera video missing for '{cam.camera_id}': {vid_path}")
                continue
            try:
                with av.open(str(vid_path)) as container:
                    if not container.streams.video:
                        errors.append(f"No video stream in {vid_path}")
                    else:
                        vstream = container.streams.video[0]
                        if vstream.width and vstream.height:
                            cam.width = int(vstream.width)
                            cam.height = int(vstream.height)
                        if vstream.frames and vstream.frames != manifest.trajectory.row_count:
                            errors.append(
                                f"Video frame count mismatch on {cam.camera_id}: "
                                f"video={vstream.frames}, csv={manifest.trajectory.row_count}"
                            )
            except Exception as exc:
                errors.append(f"Failed to open video {vid_path} with PyAV: {exc}")

        # 3. Validate semantics fields
        if manifest.source_type == "policy_rollout" and manifest.action_semantics.is_leader_command:
            errors.append("Policy rollout cannot declare is_leader_command=True.")

        manifest.validation_errors = errors
        if not errors:
            manifest.validation_status = "validated"
            manifest.readiness = "ready"
        else:
            manifest.validation_status = "invalid"
            manifest.readiness = "unavailable"
        manifest.validated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        return manifest

    def get_dataframe(self, episode_id: str) -> pd.DataFrame:
        if episode_id not in self._manifests:
            raise KeyError(f"Unknown episode_id '{episode_id}'")
        if episode_id not in self._df_cache:
            csv_path = Path(self._manifests[episode_id].trajectory.path)
            df = pd.read_csv(csv_path)
            self._df_cache[episode_id] = df
        return self._df_cache[episode_id]

    def mark_analyzed(self, episode_id: str, is_saved: bool = False) -> None:
        if episode_id in self._manifests:
            self._analyzed_episodes.add(episode_id)
            if is_saved:
                self._saved_analysis_episodes.add(episode_id)
            self._manifests[episode_id].is_analyzed = True
            self._manifests[episode_id].has_saved_analysis = (
                episode_id in self._saved_analysis_episodes
            )

    def get_counts(self) -> CatalogCounts:
        manifests = list(self._manifests.values())
        registered = len(manifests)
        validated = sum(1 for m in manifests if m.validation_status == "validated")
        analyzed = len(self._analyzed_episodes)
        policy_all = [m for m in manifests if m.source_type == "policy_rollout"]
        policy_s = [m for m in policy_all if m.outcome_annotation.status == "success"]
        policy_f = [m for m in policy_all if m.outcome_annotation.status == "failure"]
        teleop_all = [m for m in manifests if m.source_type == "teleoperation"]
        teleop_s = [m for m in teleop_all if m.outcome_annotation.status == "success"]

        return CatalogCounts(
            reported_prepared=self.reported_prepared_count,
            registered=registered,
            validated=validated,
            analyzed=analyzed,
            policy_rollout_total=len(policy_all),
            policy_success_count=len(policy_s),
            policy_failure_count=len(policy_f),
            teleop_total=len(teleop_all),
            teleop_success_count=len(teleop_s),
        )

    def get_catalog_response(self) -> CatalogResponse:
        counts = self.get_counts()
        summaries: List[CatalogEntrySummary] = []
        for ep_id, m in self._manifests.items():
            duration_s = round(m.trajectory.row_count / m.timing_semantics.nominal_fps, 2)
            summaries.append(
                CatalogEntrySummary(
                    episode_id=ep_id,
                    episode_revision=m.episode_revision,
                    display_title=m.display_title,
                    category_group=m.category_group,
                    source_type=m.source_type,
                    source_kind=m.source_kind,
                    readiness=m.readiness,
                    task_label=m.task_label,
                    robot_id=m.robot_id,
                    checkpoint_label=m.checkpoint_label,
                    outcome_status=m.outcome_annotation.status,
                    outcome_provenance=m.outcome_annotation.provenance,
                    failure_stage=m.outcome_annotation.failure_stage,
                    success_detail=m.outcome_annotation.success_Detail,
                    episode_notes=m.outcome_annotation.episode_notes,
                    ai_judge_verdict=m.outcome_annotation.ai_judge_verdict,
                    ground_truth_verdict=m.outcome_annotation.ground_truth_verdict,
                    row_count=m.trajectory.row_count,
                    duration_s=duration_s,
                    nominal_fps=m.timing_semantics.nominal_fps,
                    validation_status=m.validation_status,
                    has_saved_analysis=ep_id in self._saved_analysis_episodes,
                    is_analyzed=ep_id in self._analyzed_episodes,
                )
            )
        return CatalogResponse(
            dataset_name="lablens_so101_episodes",
            counts=counts,
            counting_note=(
                "Prepared catalog reports 16 SO-101 pick-and-place episodes "
                "(14 autonomous policy rollouts: 4 success, 10 failure; 2 teleoperation demos). "
                "Reported prepared, registered, source-validated, and analyzed counts are tracked separately."
            ),
            episodes=summaries,
        )
