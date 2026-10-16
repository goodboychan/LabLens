# LabLens v1.1 — Robot Episode Investigator

**Track:** Gemini 4 Argon in Hard Tech  
*Inspect the rollout. Follow the evidence.*

> **Browse 16 prepared episodes. Investigate one at a time. Open the evidence behind each finding.**

LabLens v1.1 is an annotation-aware, multimodal robot episode investigator for reviewing SO-101 6-DOF robot policy rollouts and teleoperation demonstrations. It connects review-worthy trajectory intervals to exact decoded original camera frames (`top` and `wrist`) and deterministic signal statistics through a bounded two-call Gemini investigation loop (`Task P` → `query_signal_window` & `get_frame_bundle` → `Task R`) authenticated via **Google Cloud Application Default Credentials (ADC)**.

---

## 1. Key Capabilities & v1.1 Scope

- **16-Episode Registered Catalog with Lazy Per-Episode Validation**:
  - Tracks **Prepared (16)**, **Registered (16)**, **Data-Validated**, and **Analyzed** counts separately (`G01`).
  - Covers **14 learned ACT policy rollouts** (`4` success, `10` failure — labeled via consensus between the overhead `top` camera geometric automatic judge and blind human video re-judging, with operator notes stored separately) and **2 teleoperation demonstrations** (`2` curated success demonstrations from LeRobot v3 `chans/quantgate-so101-coaster`).
  - Switching episodes cancels any active run and lazily validates only the selected episode without triggering background model calls (`G09`).
- **Per-Episode Action Semantics & Outcome Provenance (`G02`)**:
  - Reads `action_semantics` (`source_type`, `action_meaning`, `state_meaning`, `comparable_pairs`) and `outcome_provenance` (`supplied_outcome`, `annotation_source`, `verification_rule`, `is_ground_truth=false`) per episode from `episode.json`.
  - Rollout outcome labels disclose `top_camera_auto_judge_plus_blind_human_rejudge_consensus` (`outcome.automatic_judge` + `outcome.video_consensus_rejudge`), kept distinct from `operator_note` and `ai_judge_end_of_trial`.
  - Default mode is **annotation-aware review**, not blind outcome classification or autonomous root-cause diagnosis.
- **Two Deterministic Evidence Tools (`G03`, `G04`)**:
  - `T1: query_signal_window`: Computes exact CSV trajectory metrics (`min`, `max`, `mean`, `delta`, `abs_delta`, `max_abs_gap`, `mean_abs_gap`, `peak_gap_sample`, `peak_step_change`, `row_count`, plus real wall-clock `t_wall_s` and nominal `t_video_s`) over `sample_index` with `sample_stride=1` and `NaN`/`Inf` rejection.
  - `T3: get_frame_bundle`: Decodes exact original overhead (`top`) and wrist (`wrist`) camera frames from H.264 MP4 (`video/top.mp4`, `video/wrist.mp4`, or AV1 originals in `lablens_so101_episodes_av1_originals`) using PyAV (`av`), enforcing `<= 8` frames per bundle, `<= 24` unique frames, and `<= 36` total image transmissions.
- **Bounded Two-Call Gemini Runtime via Google Cloud ADC (`G05`)**:
  - Uses `google.auth.default()` + `google-genai` Vertex AI ADC client (`vertexai=True`) instead of static `GEMINI_API_KEY` strings.
  - Enforces a hard cap of `max_model_attempts = 2`, `1` tool round, and `2` tool calls per investigation, recording the exact model ID (`gemini-4-argon` primary with explicit fallback telemetry if unavailable in the project) in every tool receipt.
- **Interactive Dual-Tab UI (`Evidence Map` & `Investigation`) (`G08`)**:
  - **Evidence Map Tab**: Renders an 8–12 meaningful-node SVG semantic evidence graph (`<= 20` nodes, `<= 30` edges, zero orphan nodes) split into structured numeric/camera rows with interactive node inspection and a real-time **Verification Loop (`Task P` → `T1/T3` → `Task R`)** status bar.
  - **Investigation Tab**: Dual synchronized Overhead (`top`) and Wrist (`wrist`) camera players + Combined preview toggle, interactive 6-DOF joint canvas chart (`state` vs `action`, command–observation gap, gripper aperture) with brush range selection (`t_start..t_end`), decoded frame evidence strip, and saved reference reports (`G10`).

---

## 2. Supported Input Contract, Source Semantics & Annotation Policy

Each episode folder under `lablens_so101_episodes/lablens_so101_episodes/<episode_id>/` provides:
- `episode.json`: Episode metadata, `source` (`policy_rollout` or `teleop_demo`), `outcome` (`success`, `automatic_judge`, `video_consensus_rejudge`, `rule`), `operator_note`, `ai_judge_end_of_trial`, camera specs (`top`, `wrist`), and joint definitions (`shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`).
- `trajectory.csv` / `trajectory.parquet`: Canonical `frame_index` (`0..N-1`), real elapsed time `t_wall_s` (rollouts) or `t_s` (teleop), video clock `t_video_s` (`frame_index / 30`), `state.<joint>`, and `action.<joint>`.
- `video/top.mp4`, `video/wrist.mp4`: Individual original camera views (`top` overhead 640×480, `wrist` camera 320×200) used for `FrameRef` extraction, with optional side-by-side combined preview marked strictly as `is_derivative_preview=true`.

**Annotation & Authority Discipline (`G07`)**:
- Supplied outcome labels (`success` / `failure` from overhead auto-judge + blind video re-judge consensus) and separate operator notes are displayed as disclosed context, not self-discovered model classifications.
- Every finding is tagged with `authority_class` (`computed_fact`, `visual_observation`, `hypothesis`, or `unresolved`) and `cause_confidence` (`not_established` or `hypothesis_only`). Forbidden overclaim phrases (`proven root cause`, `verified physical cause`, `hardware fault confirmed`, `autonomous root-cause diagnosis`) are rejected by schema validation.

---

## 3. Authentication (Google Cloud ADC) & Setup

LabLens v1.1 authenticates with Google Cloud Vertex AI / Gemini using **Application Default Credentials (ADC)** (`google.auth.default()`). No `GEMINI_API_KEY` is required.

### Prerequisites
1. Python 3.12+
2. Google Cloud SDK (`gcloud`) authenticated with ADC:
   ```powershell
   gcloud auth application-default login
   ```
3. Optional environment variables:
   - `GOOGLE_CLOUD_PROJECT`: Your Google Cloud project ID (auto-detected from ADC if omitted)
   - `GOOGLE_CLOUD_LOCATION`: Vertex AI region (default: `global` / `us-central1`)
   - `GEMINI_MODEL_ID`: Primary model ID (default: `gemini-4-argon`, with automatic fallback to `gemini-2.5-flash` if Argon is not provisioned in the project)
   - `LABLENS_OPERATOR_SECRET` / `LABLENS_SESSION_TOKEN`: Operator session secret injected via Cloud Run environment variables / Secret Manager (if unset locally, the server generates a cryptographically random per-instance token via `secrets.token_urlsafe(32)`)

### Local Installation & Startup
```powershell
pip install -r requirements.txt
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```
Open `http://127.0.0.1:8000` in your browser.

### Running the Release Gate Tests (`G01`–`G12`)
```powershell
pytest tests/test_gates.py -v
```

---

## 4. Limitations, Data Provenance & License

- **Single-Episode Bounded Scope**: Analyzes one episode per operator-authorized request (`2` model calls, `1` tool round, `2` tools). No batch analysis across all 16 episodes, no autonomous root-cause diagnosis, and no live robot control or trajectory repair.
- **Sampling & Causation Limits**: Decoded frame bundles sample specific intervals (`<= 8` frames per bundle) and can miss transient visual events between sampled frames. Command–observation discrepancies do not by themselves establish physical causation (e.g., mechanical jam vs. policy saturation vs. calibration offset).
- **Provenance**: Application backend, frontend, schema contracts, and automated release gates were built in-event according to the LabLens v1.1 specification pack (`planning/lablens-v1_1/`). Raw SO-101 robot recordings reside under `lablens_so101_episodes/`.
- **License**: Application code is licensed under the [Apache License 2.0](LICENSE). The bundled SO-101 episode recordings under `lablens_so101_episodes/` are licensed under [CC BY 4.0](lablens_so101_episodes/lablens_so101_episodes/README.md).
