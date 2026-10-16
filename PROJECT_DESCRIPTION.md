# Project Description — LabLens v1.1

**Title:** LabLens — Robot Episode Investigator  
**Track:** Gemini 4 Argon in hard tech  
*Inspect the rollout. Follow the evidence.*

---

## Problem
Robotics engineers reviewing policy rollouts need to connect a suspected event with the camera and trajectory records behind it. A supplied failure label does not explain when or how the failure unfolded, and a command–observation difference alone does not identify its physical cause.

## Solution
LabLens offers a registered 16-episode catalog and a bounded, one-at-a-time multimodal investigation for SO-101 6-DOF robot arms. Gemini selects relevant signal and camera queries (`Task P`); fixed deterministic tools (`query_signal_window` and `get_frame_bundle`) compute recorded CSV trajectory facts (reporting both real wall-clock `t_wall_s` and nominal 30 fps video time `t_video_s`) and decode original `video/top.mp4` (overhead) and `video/wrist.mp4` (wrist) camera frames via PyAV; a final evidence review (`Task R`) produces structured findings and a clickable semantic evidence map (8–12 meaningful nodes). Source types (`policy_rollout` vs. `teleop_demo`), consensus outcome labels (`top_camera_auto_judge_plus_blind_human_rejudge_consensus`, kept distinct from separate `operator_note` annotations), action semantics, and epistemic uncertainty (`computed_fact`, `visual_observation`, `hypothesis`, `unresolved`) remain explicitly visible throughout the UI and exported reports.

## Google AI and Verification
- **Authentication & SDK**: Uses **Google Cloud Application Default Credentials (ADC)** (`google.auth.default()`) with the `google-genai` SDK (`genai.Client(vertexai=True, credentials=credentials, project=project_id, location=location)`), avoiding static API keys.
- **Model & Bounded Execution**: Configures `gemini-4-argon` (with explicit fallback telemetry to `gemini-2.5-flash` when Argon is unavailable in the target GCP project) in a strict two-call `Task P → T1/T3 Tools → Task R` sequence (`max_model_attempts = 2`, `1` tool round, `2` tool calls).
- **Deterministic Grounding**: Numeric findings cite deterministic `query_signal_window` records (`sample_stride=1`, exact min/max/mean/delta/gap math); visual claims cite exact `FrameRef` objects (`camera_name` in `{"top", "wrist"}`, `sample_index`, `timestamp_s`, `sha256_16`) actually decoded and delivered in the `get_frame_bundle` response. Pydantic v2 schemas (`extra="forbid"`) enforce authority boundaries and reject overclaim phrases (`proven root cause`, `verified physical cause`).

## Dataset and Scope
The workspace contains 16 prepared SO-101 6-DOF episodes (`lablens_so101_episodes/lablens_so101_episodes/`):
- **14 learned ACT policy rollouts** (`4` consensus `success`, `10` consensus `failure`, labeled via overhead `top` camera geometric automatic judge + blind human video re-judge consensus, with separate operator notes) recorded on a Jetson Orin Nano Super.
- **2 teleoperation demonstrations** (`2` curated `success` demonstrations: `teleop_demo_episode_000001` and `teleop_demo_episode_000047` from LeRobot v3 `chans/quantgate-so101-coaster`).
- All 16 episodes are registered (`reported_prepared_count = 16`, `registered_count = 16`) and support lazy per-episode CSV/MP4 validation (`validated_count`) and individual tracking of completed analyses (`analyzed_count`). These 16 episodes are selected evaluation/demonstration cases, not an unbiased policy benchmark. LabLens performs annotation-aware review rather than blind outcome classification.

## Observed Results
- **Tested Representative Episodes (`G12`)**:
  1. **Policy Failure (`rollout_T2012_1014_231614` & `rollout_T1002_1014_214044`)**: In `rollout_T2012_1014_231614` (1,800 rows; `60.00s` video time / `71.09s` wall time), `query_signal_window` and `get_frame_bundle` localize the late post-placement retraction knock-over (`samples 1500..1620`, `t_video=50.00s..54.00s`, `t_wall=59.10s..63.94s`) across decoded `top` and `wrist` frames.
  2. **Policy Success (`rollout_T1001_1014_213934`)**: 535 rows (`17.80s` video time). Verified transient `wrist_flex` and `gripper` tracking lag during grasp/place without falsely inferring a hardware fault.
  3. **Teleoperation Success (`teleop_demo_episode_000001`)**: 254 rows (`8.43s` at 30 Hz). Verified leader–follower command tracking with distinct teleoperation action semantics (`leader_command`, `curated_teleoperation_demonstration`).
- **Release Gates**: All 13 automated release gate tests (`G01`–`G12` in `tests/test_gates.py`) pass (`13 passed`).

## Deployment, Implementation and Data Provenance
- **Container & Cloud Run**: Packaged via root `Dockerfile` (`python:3.12-slim` + `ffmpeg`/PyAV + FastAPI/Uvicorn on port `8080`) with same-origin static UI serving, HTTP-only session cookie + `X-LabLens-CSRF` verification backed by a non-default Cloud Run secret (`LABLENS_OPERATOR_SECRET`), and Vertex AI ADC authentication.
- **Provenance & License**: All backend (`backend/`), frontend (`frontend/`), and test (`tests/`) code was created in this workspace following the LabLens v1.1 specification pack and is licensed under Apache-2.0 (`LICENSE`); dataset recordings in `lablens_so101_episodes/` are licensed under CC BY 4.0.

## Limitations
- Analyzes one episode per operator request (`<= 2` model calls, `<= 1` tool round); no batch diagnosis across all 16 episodes, no live robot control, and no trajectory repair.
- Sampled camera frames (`<= 8` per bundle, `<= 24` unique per investigation) can miss rapid inter-frame events.
- Read-only CSV and video evidence cannot distinguish between mechanical friction, external collision, or policy action saturation without external force/torque telemetry; physical causes remain hypotheses (`cause_confidence = "hypothesis_only"` or `"not_established"`).
