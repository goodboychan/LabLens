"""Automated Verification Suite for LabLens v1.1 Release Gates (G01–G12).

Tests all 12 release gates defined in 07_ACCEPTANCE_TESTS_AND_RELEASE_GATES.md:
- G01: Catalog & Lazy Loading (4 distinct counters, no eager 16-episode load)
- G02: Action Semantics & Outcome Provenance (policy_rollout vs teleoperation)
- G03: Video Decode & Frame Provenance (PyAV FrameRef, SHA-256, <=8 frames/call)
- G04: Deterministic Tool Math (query_signal_window full-res CSV accuracy)
- G05: Bounded 2-Call Gemini Runtime & Tool Loop via Google Cloud ADC
- G06: Schema Strictness (extra="forbid" across DTOs)
- G07: Authority Discipline & Separated Findings (no physical cause overclaims)
- G08: Semantic Evidence Map Bounds (<=20 nodes, <=30 edges, 4 node/edge types, no orphans)
- G09: Single-Episode Concurrency & Stop Semantics
- G10: Saved Analysis Provenance & No Silent Live-to-Saved Substitution
- G11: Security, CSRF, Origin & Sanitized Export
- G12: Representative 3-Category Coverage (policy success, policy failure, teleop demo)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.app import OPERATOR_CSRF_TOKEN, OPERATOR_SESSION_TOKEN, app
from backend.gemini_runtime import (
    InvestigationManager,
    check_gemini_auth_status,
    validate_review_authority,
)
from backend.registry import EpisodeRegistry
from backend.schemas import (
    CandidateIntervalItem,
    EvidenceReviewResponse,
    GetFrameBundleArgs,
    InspectionPlanResponse,
    InvestigationCreateRequest,
    NumericFindingReference,
    PlanHypothesisItem,
    QuerySignalWindowArgs,
    ReviewedEventItem,
    ReviewedHypothesisAssessment,
    ReviewedVisualFinding,
)
from backend.tools import EvidenceToolkit


@pytest.fixture()
def fresh_stack(tmp_path: Path):
    """Return an isolated EpisodeRegistry, EvidenceToolkit, and InvestigationManager."""
    reg = EpisodeRegistry(storage_root=tmp_path / "storage")
    tk = EvidenceToolkit(registry=reg)
    mgr = InvestigationManager(registry=reg, toolkit=tk)
    return reg, tk, mgr


@pytest.fixture()
def client():
    """Return a FastAPI TestClient with operator session and CSRF headers helper."""
    return TestClient(app)


def auth_headers(include_csrf: bool = True, origin: str = "http://localhost:8000") -> dict[str, str]:
    headers = {
        "X-LabLens-Session": OPERATOR_SESSION_TOKEN,
        "Origin": origin,
    }
    if include_csrf:
        headers["X-LabLens-CSRF"] = OPERATOR_CSRF_TOKEN
    return headers


# ============================================================================
# GATE G01: Catalog & Lazy Loading (4 Distinct Counters)
# ============================================================================
def test_g01_catalog_counters_and_lazy_validation(tmp_path: Path):
    # 1. Pure registry startup does not eagerly validate all 16 episodes
    reg_only = EpisodeRegistry(storage_root=tmp_path / "reg_only")
    cat0 = reg_only.get_catalog_response()
    assert cat0.counts.reported_prepared == 16
    assert cat0.counts.registered == 16
    assert cat0.counts.validated == 0
    assert cat0.counts.analyzed == 0

    # 2. When InvestigationManager initializes, it only pre-validates the 4 reference episodes
    tk = EvidenceToolkit(registry=reg_only)
    _mgr = InvestigationManager(registry=reg_only, toolkit=tk)
    catalog = reg_only.get_catalog_response()
    counts = catalog.counts

    assert counts.reported_prepared == 16
    assert counts.registered == 16
    assert counts.validated == 4
    assert counts.analyzed == 4
    assert counts.policy_rollout_total == 14
    assert counts.policy_success_count == 4
    assert counts.policy_failure_count == 10
    assert counts.teleop_total == 2
    assert counts.teleop_success_count == 2

    # Pick an episode that is unvalidated at startup
    unvalidated_ep = "rollout_T1006_1014_214600"
    raw_manifest = reg_only.get_manifest(unvalidated_ep, validate_if_needed=False)
    assert raw_manifest.validation_status == "unvalidated"

    # Lazy validate on selection
    validated_manifest = reg_only.get_manifest(unvalidated_ep, validate_if_needed=True)
    assert validated_manifest.validation_status == "validated"
    assert reg_only.get_catalog_response().counts.validated == 5


# ============================================================================
# GATE G02: Action Semantics & Outcome Provenance
# ============================================================================
def test_g02_action_semantics_and_outcome_provenance(fresh_stack):
    reg, _tk, _mgr = fresh_stack

    # Policy rollout failure (T1002)
    m_policy = reg.get_manifest("rollout_T1002_1014_214044", validate_if_needed=True)
    assert m_policy.source_kind == "policy_rollout"
    assert m_policy.action_semantics.kind == "issued_robot_command"
    assert m_policy.action_semantics.kind != "leader_command"
    assert m_policy.outcome_annotation.status == "failure"
    assert m_policy.outcome_annotation.provenance == "human"
    assert "AI judge said YES" in (m_policy.outcome_annotation.episode_notes or "")

    # Teleoperation demo (episode_000001)
    m_teleop = reg.get_manifest("teleop_demo_episode_000001", validate_if_needed=True)
    assert m_teleop.source_kind == "teleoperation"
    assert m_teleop.action_semantics.kind == "leader_command"
    assert m_teleop.outcome_annotation.status == "success"


# ============================================================================
# GATE G03: Video Decode & Frame Provenance (FrameRef & PyAV)
# ============================================================================
def test_g03_pyav_frame_decode_and_bundle_caps(fresh_stack):
    _reg, tk, _mgr = fresh_stack

    frame_ref, jpeg_bytes = tk.decode_single_frame(
        episode_id="rollout_T1002_1014_214044",
        camera_id="top",
        sample_index=420,
        resolution_tier="detail",
    )
    assert jpeg_bytes[:2] == b"\xff\xd8"  # Valid JPEG SOI marker
    assert frame_ref.episode_id == "rollout_T1002_1014_214044"
    assert frame_ref.camera == "top"
    assert frame_ref.sample_index == 420
    assert frame_ref.width > 0 and frame_ref.height > 0
    assert len(frame_ref.source_video_sha256) == 64
    assert len(frame_ref.decoded_frame_sha256) == 64

    # T3 get_frame_bundle enforces <= 8 frames per call (and sample_count <= 6 in schema)
    args = GetFrameBundleArgs(
        episode_id="rollout_T1002_1014_214044",
        camera="both",
        camera_ids=["top", "wrist"],
        start_sample=320,
        end_sample=600,
        sample_count=6,  # 6 timestamps * 2 cameras = 12 requested, capped to <= 8 frames
        resolution_tier="detail",
    )
    res, receipt, imgs = tk.execute_get_frame_bundle(args, investigation_id="test_inv")
    assert receipt.status == "completed"
    assert len(res.frame_evidence) <= 8
    assert len(imgs) == len(res.frame_evidence)


# ============================================================================
# GATE G04: Deterministic Tool Math (query_signal_window)
# ============================================================================
def test_g04_deterministic_signal_window_math(fresh_stack):
    reg, tk, _mgr = fresh_stack
    ep_id = "rollout_T1002_1014_214044"
    df = reg.get_dataframe(ep_id)
    sub = df.iloc[300 : 500 + 1]

    args = QuerySignalWindowArgs(
        episode_id=ep_id,
        start_sample=300,
        end_sample=500,
        signals=["state.gripper", "action.gripper"],
        include_State_action_diff=True,
        requested_difference_pairs=["diff_gripper"],
    )
    res, receipt = tk.execute_query_signal_window(args, investigation_id="test_math")
    assert receipt.status == "completed"
    assert res.row_count == 201

    sig_by_name = {s.signal: s for s in res.signal_summaries}
    expected_min = round(float(sub["state.gripper"].min()), 4)
    expected_max = round(float(sub["state.gripper"].max()), 4)
    expected_mean = round(float(sub["state.gripper"].mean()), 4)
    expected_delta = round(float(sub["state.gripper"].iloc[-1] - sub["state.gripper"].iloc[0]), 4)

    g_metrics = sig_by_name["state.gripper"]
    assert g_metrics.min_value == pytest.approx(expected_min, abs=1e-4)
    assert g_metrics.max_value == pytest.approx(expected_max, abs=1e-4)
    assert g_metrics.mean_value == pytest.approx(expected_mean, abs=1e-4)
    assert g_metrics.delta == pytest.approx(expected_delta, abs=1e-4)

    # Verify state - action difference calculation
    diff_by_id = {d.pair_id: d for d in res.difference_summaries}
    diff_series = sub["state.gripper"] - sub["action.gripper"]
    expected_diff_max = round(float(diff_series.max()), 4)
    d_metrics = diff_by_id["diff_gripper"]
    assert d_metrics.max_signed_diff == pytest.approx(expected_diff_max, abs=1e-4)


# ============================================================================
# GATE G05: Bounded 2-Call Gemini Runtime & Tool Loop via ADC
# ============================================================================
def test_g05_bounded_two_call_gemini_adc_loop(fresh_stack):
    reg, _tk, mgr = fresh_stack

    # Verify ADC status checker reports adc mode
    adc_info = check_gemini_auth_status()
    assert adc_info["auth_mode"] == "adc"
    assert adc_info["project_id"]

    # Reject creation without explicit_authorization=True
    with pytest.raises(PermissionError, match="Explicit operator authorization"):
        mgr.create_investigation(
            InvestigationCreateRequest(
                episode_id="rollout_T1002_1014_214044",
                question="Test unauthorized",
                start_sample=300,
                end_sample=600,
                explicit_authorization=False,
            )
        )

    # Create authorized investigation
    inv = mgr.create_investigation(
        InvestigationCreateRequest(
            episode_id="rollout_T1002_1014_214044",
            question="Why did placement fail?",
            start_sample=320,
            end_sample=620,
            explicit_authorization=True,
        )
    )
    manifest = reg.get_manifest("rollout_T1002_1014_214044", validate_if_needed=True)

    # Mock Gemini ADC Client to verify exact 2-call + 1-tool-round execution
    call_counter = {"count": 0}

    def fake_generate_content(*, model: str, contents: Any, config: Any):
        call_counter["count"] += 1
        envelope = json.loads(contents[0])
        if call_counter["count"] == 1:
            plan = InspectionPlanResponse(
                request_id=envelope["request_id"],
                investigation_id=inv.investigation_id,
                episode_id=manifest.episode_id,
                episode_revision=manifest.episode_revision,
                decision="inspect",
                annotation_disclosure=inv.annotation_disclosure,
                restated_question=inv.question,
                focus_start_sample=320,
                focus_end_sample=620,
                focus_rationale="Inspect cup placement window 320-620.",
                candidate_intervals=[
                    CandidateIntervalItem(
                        interval_id="evt_release",
                        start_sample=420,
                        end_sample=560,
                        selection_origin="model_selected_from_overview",
                        reason="Check gripper opening and wrist pitch.",
                        delivered_frame_ids=[],
                        visibility_and_coverage_limits="Overview frames are sparse.",
                    )
                ],
                candidate_hypotheses=[
                    "Wrist alignment offset before release positioned the cup over the coaster edge."
                ],
                hypotheses=[
                    PlanHypothesisItem(
                        hypothesis_id="hyp_1",
                        statement="Wrist alignment offset before release positioned the cup over the coaster edge.",
                        supplied_evidence_refs=[],
                        evidence_needed="Sampled detail frames around 420-560.",
                    )
                ],
                signal_window_request=QuerySignalWindowArgs(
                    episode_id=manifest.episode_id,
                    episode_revision=manifest.episode_revision,
                    start_sample=320,
                    end_sample=620,
                    signals=["state.gripper", "action.gripper", "state.wrist_flex", "action.wrist_flex"],
                    include_State_action_diff=True,
                    requested_difference_pairs=["diff_gripper", "diff_wrist_flex"],
                ),
                frame_bundle_request=GetFrameBundleArgs(
                    episode_id=manifest.episode_id,
                    episode_revision=manifest.episode_revision,
                    camera="both",
                    camera_ids=["top", "wrist"],
                    start_sample=360,
                    end_sample=600,
                    sample_count=4,
                    resolution_tier="detail",
                ),
            )
            return SimpleNamespace(text=plan.model_dump_json(), usage_metadata=None)
        else:
            # Call #2: EvidenceReviewResponse referencing actual delivered evidence IDs from envelope
            delivered_num = envelope["delivered_numerical_evidence"]
            delivered_frm = envelope["delivered_frame_evidence"]
            num_id = delivered_num[0]["evidence_id"]
            frm_id = delivered_frm[0]["evidence_id"]

            review = EvidenceReviewResponse(
                request_id=envelope["request_id"],
                investigation_id=inv.investigation_id,
                episode_id=manifest.episode_id,
                episode_revision=manifest.episode_revision,
                decision="report",
                annotation_disclosure=inv.annotation_disclosure,
                direct_answer="Sampled frames show the cup rim landing partially on the coaster border while gripper opens.",
                computed_fact_citations=[num_id],
                numeric_findings=[
                    NumericFindingReference(
                        evidence_id=num_id,
                        metric_keys=["min_value", "max_value", "delta"],
                        concise_interpretation="Gripper state and action signals diverge during release.",
                    )
                ],
                events=[
                    ReviewedEventItem(
                        event_id="evt_release",
                        label="Gripper release over coaster",
                        start_sample=420,
                        end_sample=560,
                        selection_origin="model_selected_from_overview",
                        description="Gripper opens while cup sits on coaster edge.",
                        grounded_evidence_ids=[num_id, frm_id],
                    )
                ],
                visual_findings=[
                    ReviewedVisualFinding(
                        finding_id="vf_1",
                        observation="Top and wrist cameras show the cup base tilted on the coaster rim.",
                        sampled_frame_evidence_ids=[frm_id],
                        sample_range_label="Samples 360–600",
                        is_sampled_observation=True,
                    )
                ],
                hypothesis_assessments=[
                    ReviewedHypothesisAssessment(
                        hypothesis_id="hyp_1",
                        statement="Wrist alignment offset before release positioned the cup over the coaster edge.",
                        status="consistent_with_available_evidence",
                        rationale="Consistent with wrist_flex trajectory and sampled release frames.",
                        supporting_evidence_ids=[num_id, frm_id],
                        conflicting_evidence_ids=[],
                        missing_evidence=["Exact contact force cannot be proven from RGB+encoders."],
                    )
                ],
                unresolved_physical_causes=["Sub-frame slip vs contact friction remains unresolved."],
                recommended_manual_checks=["Inspect samples 500-560 frame by frame."],
                limitations=["Sparse frames do not prove continuous retention between samples."],
            )
            return SimpleNamespace(text=review.model_dump_json(), usage_metadata=None)

    fake_client = SimpleNamespace(models=SimpleNamespace(generate_content=fake_generate_content))
    completed = mgr.run_full_investigation(inv.investigation_id, genai_client=fake_client)

    assert completed.status == "completed"
    assert call_counter["count"] == 2
    assert completed.budget_usage.semantic_calls_used == 2
    assert completed.budget_usage.tool_rounds_used == 1
    assert completed.budget_usage.total_image_transmissions <= 24
    assert completed.final_report is not None
    assert completed.final_report.analysis_mode == "live"


# ============================================================================
# GATE G06: Schema Strictness (extra="forbid")
# ============================================================================
def test_g06_schema_strictness_extra_forbid():
    with pytest.raises(ValidationError):
        QuerySignalWindowArgs(
            episode_id="rollout_T1002_1014_214044",
            start_sample=0,
            end_sample=100,
            signals=["state.gripper"],
            unexpected_field="not_allowed",  # type: ignore[call-arg]
        )

    with pytest.raises(ValidationError):
        GetFrameBundleArgs(
            episode_id="rollout_T1002_1014_214044",
            camera="top",
            start_sample=10,
            end_sample=100,
            extra_field=123,  # type: ignore[call-arg]
        )


# ============================================================================
# GATE G07: Authority Discipline & Separated Findings
# ============================================================================
def test_g07_authority_discipline_and_overclaim_rejection(fresh_stack):
    reg, _tk, mgr = fresh_stack
    ep_id = "rollout_T1002_1014_214044"
    manifest = reg.get_manifest(ep_id, validate_if_needed=True)
    saved = mgr.get_or_create_saved_report_for_episode(ep_id)

    # Verify separated findings in FinalReport
    assert len(saved.computed_facts) > 0
    assert len(saved.visual_findings) > 0
    assert all(vf.is_sampled_observation is True for vf in saved.visual_findings)
    assert len(saved.hypotheses) > 0
    assert len(saved.unresolved_physical_causes) > 0

    # Verify overclaim validator rejects physical cause claims in visual/numeric findings
    bad_review = EvidenceReviewResponse(
        request_id="req_test",
        investigation_id="inv_test",
        episode_id=ep_id,
        episode_revision=manifest.episode_revision,
        decision="report",
        annotation_disclosure="disclosed",
        direct_answer="Test overclaim rejection.",
        computed_fact_citations=["num_001"],
        numeric_findings=[
            NumericFindingReference(
                evidence_id="num_001",
                metric_keys=["max_signed_diff"],
                concise_interpretation="The CSV difference proved motor stall at sample 400.",
            )
        ],
        events=[
            ReviewedEventItem(
                event_id="evt_1",
                label="Event",
                start_sample=320,
                end_sample=400,
                selection_origin="user_selected",
                description="Summary",
                grounded_evidence_ids=["num_001"],
            )
        ],
        visual_findings=[
            ReviewedVisualFinding(
                finding_id="vf_1",
                observation="Sampled frame shows gripper open.",
                sampled_frame_evidence_ids=["frm_001"],
                sample_range_label="Samples 320–400",
                is_sampled_observation=True,
            )
        ],
        hypothesis_assessments=[
            ReviewedHypothesisAssessment(
                hypothesis_id="hyp_1",
                statement="Possible stall.",
                status="insufficient_evidence",
                rationale="Unverified.",
                supporting_evidence_ids=["num_001"],
                conflicting_evidence_ids=[],
                missing_evidence=["No torque sensor."],
            )
        ],
        unresolved_physical_causes=["Motor torque limit unverified."],
        recommended_manual_checks=["Check logs."],
        limitations=["RGB and encoders only."],
    )
    violations = validate_review_authority(
        bad_review,
        expected_episode_id=ep_id,
        expected_revision=manifest.episode_revision,
        delivered_numerical_ids={"num_001"},
        delivered_frame_ids={"frm_001"},
    )
    assert any("overclaims physical cause" in v for v in violations)


# ============================================================================
# GATE G08: Semantic Evidence Map Bounds & Zero Orphans
# ============================================================================
def test_g08_evidence_graph_bounds_and_no_orphans(fresh_stack):
    _reg, _tk, mgr = fresh_stack
    for ep_id in [
        "rollout_T1002_1014_214044",
        "rollout_T2012_1014_231614",
        "rollout_T1001_1014_213934",
        "teleop_demo_episode_000001",
    ]:
        report = mgr.get_or_create_saved_report_for_episode(ep_id)
        graph = report.evidence_graph
        assert 1 <= len(graph.nodes) <= 20
        assert len(graph.edges) <= 30

        node_ids = {n.node_id for n in graph.nodes}
        allowed_node_types = {"Episode", "Event", "Hypothesis", "Evidence"}
        allowed_edge_types = {"PART_OF", "GROUNDED_IN", "INVESTIGATES", "BEARS_ON"}

        for n in graph.nodes:
            assert n.node_type in allowed_node_types
        for e in graph.edges:
            assert e.edge_type in allowed_edge_types
            assert e.source in node_ids, f"Orphan source {e.source}"
            assert e.target in node_ids, f"Orphan target {e.target}"


# ============================================================================
# GATE G09: Single-Episode Concurrency & Stop Semantics
# ============================================================================
def test_g09_single_episode_stop_and_switch_isolation(fresh_stack):
    _reg, _tk, mgr = fresh_stack
    inv1 = mgr.create_investigation(
        InvestigationCreateRequest(
            episode_id="rollout_T1002_1014_214044",
            question="First episode run",
            start_sample=240,
            end_sample=600,
            explicit_authorization=True,
        )
    )
    stopped = mgr.stop_investigation(inv1.investigation_id)
    assert stopped.status == "stopped"

    # Attempting to run a stopped investigation returns stopped state without making model calls
    after_run = mgr.run_full_investigation(inv1.investigation_id)
    assert after_run.status == "stopped"
    assert after_run.budget_usage.semantic_calls_used == 0


# ============================================================================
# GATE G10: Saved Analysis Provenance & No Silent Substitution
# ============================================================================
def test_g10_saved_analysis_label_and_no_silent_fallback(fresh_stack):
    _reg, _tk, mgr = fresh_stack
    saved = mgr.get_or_create_saved_report_for_episode("rollout_T1002_1014_214044")
    assert saved.analysis_mode == "saved_analysis"

    # Simulate a live Gemini ADC failure and verify it never silently substitutes saved_analysis
    inv = mgr.create_investigation(
        InvestigationCreateRequest(
            episode_id="rollout_T1002_1014_214044",
            question="Live run that fails ADC/network",
            start_sample=320,
            end_sample=620,
            explicit_authorization=True,
        )
    )

    def failing_generate_content(**_kwargs):
        raise RuntimeError("Simulated ADC or Vertex AI endpoint failure")

    failing_client = SimpleNamespace(
        models=SimpleNamespace(generate_content=failing_generate_content)
    )
    failed_inv = mgr.run_full_investigation(inv.investigation_id, genai_client=failing_client)
    assert failed_inv.status == "failed"
    assert failed_inv.final_report is None
    assert "Simulated ADC" in (failed_inv.error_message or "")


# ============================================================================
# GATE G11: Security, CSRF, Origin & Sanitized Export
# ============================================================================
def test_g11_security_csrf_origin_and_sanitized_export(client: TestClient):
    # 1. Unauthenticated request to /api/catalog must fail with 401
    unauth_resp = client.get("/api/catalog")
    assert unauth_resp.status_code == 401

    # 2. Mutating POST without CSRF header must fail with 403
    no_csrf_resp = client.post(
        "/api/catalog/validate-all",
        headers=auth_headers(include_csrf=False),
    )
    assert no_csrf_resp.status_code == 403

    # 3. Mutating POST from untrusted cross-origin must fail with 403
    bad_origin_resp = client.post(
        "/api/catalog/validate-all",
        headers=auth_headers(include_csrf=True, origin="https://evil.example.com"),
    )
    assert bad_origin_resp.status_code == 403

    # 4. Sanitized export must contain zero secrets or local paths
    export_resp = client.get(
        "/api/episodes/rollout_T1002_1014_214044/export",
        headers=auth_headers(include_csrf=False),
    )
    assert export_resp.status_code == 200
    bundle = export_resp.json()
    raw_json_str = export_resp.text

    assert bundle["export_schema"] == "lablens.export.v1_1"
    assert bundle["auth_mode"] == "adc"
    assert OPERATOR_SESSION_TOKEN not in raw_json_str
    assert OPERATOR_CSRF_TOKEN not in raw_json_str
    assert "d:\\gdm" not in raw_json_str.lower()
    assert "d:/gdm" not in raw_json_str.lower()


# ============================================================================
# GATE G12: Representative 3-Category Coverage
# ============================================================================
def test_g12_three_category_representative_coverage(client: TestClient):
    categories_to_test = [
        ("rollout_T1001_1014_213934", "policy_rollout", "issued_robot_command", "success"),
        ("rollout_T1002_1014_214044", "policy_rollout", "issued_robot_command", "failure"),
        ("rollout_T2012_1014_231614", "policy_rollout", "issued_robot_command", "failure"),
        ("teleop_demo_episode_000001", "teleoperation", "leader_command", "success"),
    ]
    hdrs = auth_headers()
    for ep_id, expected_source, expected_action, expected_outcome in categories_to_test:
        m_resp = client.get(f"/api/episodes/{ep_id}/manifest", headers=hdrs)
        assert m_resp.status_code == 200
        manifest = m_resp.json()
        assert manifest["source_kind"] == expected_source
        assert manifest["action_semantics"]["kind"] == expected_action
        assert manifest["outcome_annotation"]["status"] == expected_outcome

        r_resp = client.get(f"/api/episodes/{ep_id}/saved-report", headers=hdrs)
        assert r_resp.status_code == 200
        report = r_resp.json()
        assert report["episode_id"] == ep_id
        assert len(report["computed_facts"]) > 0
        assert len(report["visual_findings"]) > 0
        assert len(report["frame_evidence"]) > 0
        assert len(report["evidence_graph"]["nodes"]) <= 20


# ============================================================================
# Cloud Run Deployment & Readiness Verification
# ============================================================================
def test_cloud_run_deployment_readiness_and_health(client: TestClient):
    root_dir = Path(__file__).resolve().parent.parent

    # 1. Verify /api/health endpoint for Cloud Run probes
    h_resp = client.get("/api/health")
    assert h_resp.status_code == 200
    h_data = h_resp.json()
    assert h_data["status"] == "ok"
    assert h_data["registered_episodes"] == 16
    assert h_data["reported_prepared_episodes"] == 16
    assert h_data["auth_mode"] == "adc"
    assert OPERATOR_SESSION_TOKEN not in h_resp.text
    assert OPERATOR_CSRF_TOKEN not in h_resp.text

    # 2. Verify Cloud Run X-Forwarded-Host same-origin support & edge cases
    cloud_run_hdrs = auth_headers(include_csrf=True, origin="https://lablens-12345.us-central1.run.app")
    cloud_run_hdrs["X-Forwarded-Host"] = "lablens-12345.us-central1.run.app"
    cr_resp = client.post("/api/catalog/validate-all", headers=cloud_run_hdrs)
    assert cr_resp.status_code == 200

    # Multi-hop X-Forwarded-Host header
    multi_hdrs = auth_headers(include_csrf=True, origin="https://lablens-12345.us-central1.run.app")
    multi_hdrs["X-Forwarded-Host"] = "lablens-12345.us-central1.run.app, 169.254.1.1"
    multi_resp = client.post("/api/catalog/validate-all", headers=multi_hdrs)
    assert multi_resp.status_code == 200

    # Cross-origin must still be rejected even with X-Forwarded-Host present
    evil_cr_hdrs = auth_headers(include_csrf=True, origin="https://evil.example.com")
    evil_cr_hdrs["X-Forwarded-Host"] = "lablens-12345.us-central1.run.app"
    evil_cr_resp = client.post("/api/catalog/validate-all", headers=evil_cr_hdrs)
    assert evil_cr_resp.status_code == 403

    # Container /app/ path sanitization
    from backend.app import _strip_local_paths
    sanitized = _strip_local_paths({"file": "/app/storage/episodes/rollout_T1002_1014_214044/videos/top.mp4"})
    assert "/app/" not in sanitized["file"]
    assert sanitized["file"] == "storage/episodes/rollout_T1002_1014_214044/videos/top.mp4"

    # 3. Verify Dockerfile, requirements.txt, .dockerignore, and .gcloudignore
    dockerfile_text = (root_dir / "Dockerfile").read_text(encoding="utf-8")
    reqs_text = (root_dir / "requirements.txt").read_text(encoding="utf-8")
    dockerignore_text = (root_dir / ".dockerignore").read_text(encoding="utf-8")
    gcloudignore_text = (root_dir / ".gcloudignore").read_text(encoding="utf-8")

    assert "GOOGLE_CLOUD_LOCATION=global" in dockerfile_text
    assert "COPY requirements.txt" in dockerfile_text
    assert "/app/planning" not in dockerfile_text
    for pkg in ("fastapi", "uvicorn", "pydantic", "numpy", "pandas", "av", "Pillow", "python-dotenv", "google-genai", "google-auth"):
        assert pkg in reqs_text
    assert "*.zip" in dockerignore_text
    assert "lablens_so101_episodes_av1_originals/" in dockerignore_text
    assert "*.zip" in gcloudignore_text
    assert "lablens_so101_episodes_av1_originals/" in gcloudignore_text

