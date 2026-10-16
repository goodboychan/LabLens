# Project Description — LabLens v1.1

**Title:** LabLens — Robot Episode Investigator  
**Track:** Gemini 4 Argon in hard tech  
*Inspect the rollout. Follow the evidence.*

---

## Problem
Robotics engineers reviewing policy rollouts need to connect a suspected event with the camera and trajectory records behind it. A supplied failure label does not explain the failure, and a command–observation difference alone does not identify its cause.

## Solution
LabLens offers a registered 16-episode catalog and a bounded, one-at-a-time multimodal investigation for SO-101 6-DOF robot arms. Gemini selects relevant signal and camera queries (`Task P`); fixed deterministic tools (`query_signal_window` and `get_frame_bundle`) compute recorded CSV trajectory facts and decode original `front.mp4` / `top.mp4` camera frames via PyAV; a final evidence review (`Task R`) produces structured findings and a small clickable semantic evidence map (8–12 meaningful nodes). Source types (`policy_rollout` vs. `teleoperation`), operator-supplied outcome labels (`success` / `failure`), action semantics, and epistemic uncertainty (`computed_fact`, `visual_observation`, `hypothesis`, `unresolved`) remain explicitly visible throughout the UI and exported reports.

## Google AI and Verification
- **Authentication & SDK**: Uses **Google Cloud Application Default Credentials (ADC)** (`google.auth.default()`) with the `google-genai` SDK (`genai.Client(vertexai=True, credentials=credentials, project=project_id, location=location)`), avoiding static API keys.
- **Model & Bounded Execution**: Configures `gemini-4-argon` (with explicit fallback telemetry to `gemini-2.5-flash` when Argon is unavailable in the target GCP project) in a strict two-call `Task P → T1/T3 Tools → Task R` sequence (`max_model_attempts = 2`, `1` tool round, `2` tool calls).
- **Deterministic Grounding**: Numeric findings cite deterministic `query_signal_window` records (`sample_stride=1`, exact min/max/mean/delta/gap math); visual claims cite exact `FrameRef` objects (`camera_name`, `sample_index`, `timestamp_s`, `sha256_16`) actually decoded and delivered in the `get_frame_bundle` response. Pydantic v2 schemas (`extra="forbid"`) enforce authority boundaries and reject overclaim phrases (`proven root cause`, `verified physical cause`).

## Dataset and Scope
The workspace contains 16 prepared SO-101 6-DOF episodes (`lablens_so101_episodes/lablens_so101_episodes/`):
- **14 learned ACT policy rollouts** (`4` operator-labeled `success`, `10` operator-labeled `failure`) recorded on a Jetson Orin Nano Super (`T1001`–`T1003`).
- **2 teleoperation demonstrations** (`2` operator-labeled `success`, `demo_ep000_1001_140028` and `demo_ep013_1001_140846`).
- All 16 episodes are registered (`prepared_count = 16`, `registered_count = 16`) and support lazy per-episode CSV/MP4 validation (`data_validated_count`) and individual tracking of completed analyses (`analyzed_count`). These 16 episodes are selected evaluation/demonstration cases, not an unbiased policy benchmark. LabLens performs annotation-aware review rather than blind outcome classification.

## Observed Results
- **Tested Representative Episodes (`G12`)**:
  1. **Policy Failure (`rollout_T1002_1014_214044`)**: 535 rows (`17.80s` at 30 Hz). `query_signal_window` localized peak `wrist_flex` command–observation gap (`37.86 deg` at `sample_index=248`, `t=8.27s`) and `gripper` gap (`24.81` at `sample_index=210`, `t=7.00s`), paired with decoded `front` and `top` frames across the grasp/lift interval (`samples 165..320`).
  2. **Policy Success (`rollout_T1001_1014_212329`)**: 536 rows (`17.83s` at 30 Hz). Verified transient `wrist_flex` tracking lag (`30.58 deg` at `sample_index=322`) without falsely inferring a hardware fault.
  3. **Teleoperation Success (`demo_ep000_1001_140028`)**: 397 rows (`13.20s` at 30 Hz). Verified leader–follower command tracking (`shoulder_lift` peak gap `45.39 deg` at `sample_index=354`) with distinct teleoperation action semantics (`leader_goal_position`).
  4. **Additional Policy Failure Reference (`rollout_T1001_1014_212604`)**: 535 rows (`17.80s` at 30 Hz) pre-populated as a saved reference report.
- **Release Gates**: All 12 automated release gates (`G01`–`G12` in `tests/test_gates.py`) pass (`12 passed in 28.75s`).

## Deployment, Implementation and Data Provenance
- **Container & Cloud Run**: Packaged via root `Dockerfile` (`python:3.12-slim` + `ffmpeg`/PyAV + FastAPI/Uvicorn on port `8080`) with same-origin static UI serving, HTTP-only session cookie + `X-CSRF-Token` verification, and Vertex AI ADC authentication.
- **Provenance**: All backend (`backend/`), frontend (`frontend/`), and test (`tests/`) code was created from scratch in this workspace following the LabLens v1.1 specification pack (`00_START_HERE.md`–`08_SOURCES_RULES_AND_MIGRATION.md`) and UI layout mockups (`1000016204.svg`, `1000016205.svg`).

## Limitations
- Analyzes one episode per operator request (`<= 2` model calls, `<= 1` tool round); no batch diagnosis across all 16 episodes, no live robot control, and no trajectory repair.
- Sampled camera frames (`<= 8` per bundle, `<= 24` unique per investigation) can miss rapid inter-frame events.
- Read-only CSV and video evidence cannot distinguish between mechanical friction, external collision, or policy action saturation without external force/torque telemetry; physical causes remain hypotheses (`cause_confidence = "hypothesis_only"` or `"not_established"`).
