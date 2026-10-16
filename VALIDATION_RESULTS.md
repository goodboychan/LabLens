# Validation Results — LabLens v1.1 Release Gates (`G01`–`G12`)

**Test Suite:** `tests/test_gates.py`  
**Command:** `pytest tests/test_gates.py -v`  
**Observed Result:** `12 passed in 28.75s` (100% pass rate)

---

## Release Gate Summary (`05_ACCEPTANCE_AND_VALIDATION.md`)

| Gate ID | Gate Name | Test Function | Status | Verification Details |
|---|---|---|---|---|
| **G01** | Catalog counters & lazy per-episode validation | `test_g01_catalog_counters_and_lazy_validation` | **PASSED** | Verified `prepared_count=16`, `registered_count=16` (`14` policy rollouts: `4` success / `10` failure; `2` teleoperation demos: `2` success). Selecting a single episode validates only that episode (`data_validated_count` increments by 1) without triggering background model calls. |
| **G02** | Per-episode action semantics & outcome provenance | `test_g02_action_semantics_and_outcome_provenance` | **PASSED** | Verified `policy_rollout` (`rollout_T1002_1014_214044`) vs. `teleoperation` (`demo_ep000_1001_140028`) carry distinct `action_semantics` and disclosed `outcome_label_provenance` (`is_verified_ground_truth=False`). |
| **G03** | Original camera decoding & bundle caps | `test_g03_pyav_frame_decode_and_bundle_caps` | **PASSED** | Verified PyAV decodes exact original frames from `front.mp4` and `top.mp4` (`is_derivative_preview=False`), caches JPEG artifacts with SHA-256 prefixes, and enforces `<= 8` total frames per `get_frame_bundle` call while marking `combined.mp4` as a derivative preview (`is_derivative_preview=True`). |
| **G04** | Deterministic `query_signal_window` math | `test_g04_deterministic_signal_window_math` | **PASSED** | Verified exact full-resolution (`sample_stride=1`) computation of `min_value`, `max_value`, `mean_value`, `delta`, `abs_delta`, `max_abs_gap`, `mean_abs_gap`, `peak_gap_sample_index`, and `peak_step_change` against raw `trajectory.csv` rows, plus rejection of incompatible/unknown signal names. |
| **G05** | Bounded two-call Gemini ADC runtime loop | `test_g05_bounded_two_call_gemini_adc_loop` | **PASSED** | Verified Google Cloud ADC authentication (`auth_mode="adc"`, `google.auth.default()`) and strict execution bounds: `model_attempts_used <= 2`, `tool_rounds_used == 1`, `tool_calls_used <= 2`, `unique_frames_delivered <= 24`, `total_image_transmissions <= 36`. |
| **G06** | Schema strictness (`extra="forbid"`) | `test_g06_schema_strictness_extra_forbid` | **PASSED** | Verified Pydantic v2 models (`InspectionPlanResponse`, `EvidenceReviewResponse`, `InvestigationReport`) reject unknown extra fields, enforce maximum tool request counts (`<= 2`), and reject invalid evidence map node/edge references. |
| **G07** | Authority discipline & overclaim rejection | `test_g07_authority_discipline_and_overclaim_rejection` | **PASSED** | Verified `FindingCard` rejects forbidden overclaim phrases (`proven root cause`, `verified physical cause`, `hardware fault confirmed`, `autonomous root-cause diagnosis`) and enforces `cause_confidence` (`not_established` or `hypothesis_only`). |
| **G08** | Clickable semantic evidence map bounds (`8–12` nodes) | `test_g08_evidence_graph_bounds_and_no_orphans` | **PASSED** | Verified generated `EvidenceMapGraph` contains `8..12` meaningful nodes (hard cap `<= 20` nodes, `<= 30` edges) and zero orphan nodes, with valid `SignalWindowRef` and `FrameRef` source pointers. |
| **G09** | Single-episode isolation & stop/switch cancellation | `test_g09_single_episode_stop_and_switch_isolation` | **PASSED** | Verified `POST /api/investigate/stop` and episode switching cleanly cancel/isolate active state without leaking cross-episode `FrameRef` or `SignalWindowRef` records. |
| **G10** | Saved reference reports & no silent fallback | `test_g10_saved_analysis_label_and_no_silent_fallback` | **PASSED** | Verified pre-generated reference reports carry `is_saved_reference=True` and `saved_reference_banner`, and verified failed live runs do not silently substitute a saved reference report as a live result. |
| **G11** | Session cookie, CSRF/Origin check & sanitized export | `test_g11_security_csrf_origin_and_sanitized_export` | **PASSED** | Verified state-mutating POST routes reject missing `X-CSRF-Token` or cross-origin `Origin` headers (`403 Forbidden`), and verified `GET /api/reports/{id}/export` strips internal server paths and credentials. |
| **G12** | Three-category representative coverage | `test_g12_three_category_representative_coverage` | **PASSED** | Verified end-to-end compatibility across all three required source/outcome classes: Policy Failure (`rollout_T1002_1014_214044`), Policy Success (`rollout_T1001_1014_212329`), and Teleoperation Success (`demo_ep000_1001_140028`). |
