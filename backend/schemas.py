"""LabLens v1.1 Canonical Data Contracts and Schemas.

Strictly implements the normative contracts in:
- 01_PRD.md
- 03_GEMINI_RUNTIME_PROMPTS.md
- 04_DATA_TOOLS_AND_EVIDENCE_CONTRACT.md
- 05_ACCEPTANCE_AND_VALIDATION.md
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator


# --- 1. Canonical Literals ---

EpisodeSourceType = Literal[
    "policy_rollout",
    "teleoperation",
    "perturbation_trial",
    "simulation",
    "real_robot_Other",
    "unknown",
]

ReadinessStatus = Literal["registered", "ready", "limited", "unavailable"]

ReportedOutcomeStatus = Literal[
    "success",
    "failure",
    "partial",
    "unknown",
    "not_applicable",
]

OutcomeProvenance = Literal[
    "rule_based",
    "human",
    "model_generated",
    "user_reported_unverified",
    "unknown",
]

AnnotationDisclosureMode = Literal["disclosed", "blind"]

ActionSemanticsKind = Literal[
    "leader_command",
    "policy_raw_output",
    "policy_postprocessed_target",
    "issued_robot_command",
    "unknown",
]

ActionRelationship = Literal[
    "commanded_Next_target",
    "leader_target",
    "delta_command",
    "policy_output_Before_postprocess",
    "policy_output_After_chunk_blending_clipped_to_motor_limits",
    "issued_robot_command",
    "unverified",
]

InvestigationStatus = Literal[
    "idle",
    "authorized",
    "planning",
    "planned",
    "executing_tools",
    "reviewing",
    "completed",
    "limited",
    "partial",
    "stopped",
    "failed",
    "rejected_late",
]

AnalysisMode = Literal["live", "saved_analysis", "synthetic_test"]

NodeType = Literal["Episode", "Event", "Hypothesis", "Evidence"]

EdgeType = Literal["PART_OF", "GROUNDED_IN", "INVESTIGATES", "BEARS_ON"]

HypothesisAssessmentStatus = Literal[
    "consistent_with_available_evidence",
    "challenged_by_available_evidence",
    "insufficient_evidence",
    "supported",
    "conflicted",
    "unresolved",
]


# --- 2. Episode Manifest & Catalog Schemas ---

class OutcomeAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: ReportedOutcomeStatus
    value: ReportedOutcomeStatus = "unknown"
    provenance: OutcomeProvenance
    provenance_kind: OutcomeProvenance = "unknown"
    annotator_or_rule_id: Optional[str] = None
    criterion: Optional[str] = None
    annotation_revision: str = "rev1"
    rationale: Optional[str] = None
    success_Detail: Optional[str] = None
    failure_stage: Optional[str] = None
    episode_notes: Optional[str] = None
    ai_judge_verdict: Optional[str] = None
    ground_truth_verdict: Optional[str] = None
    disclosed_to_model: bool = True

    @model_validator(mode="after")
    def sync_fields(self) -> "OutcomeAnnotation":
        self.value = self.status
        self.provenance_kind = self.provenance
        return self


class CameraSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    camera_id: str
    role: str
    path: str
    av1_original_path: Optional[str] = None
    codec: str = "h264"
    av1_codec: Optional[str] = None
    width: int = 640
    height: int = 480
    fps_nominal: float = 30.0
    frame_count: int
    sha256: str
    size_bytes: int


class TrajectorySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    format: Literal["csv", "parquet"] = "csv"
    row_count: int
    columns: List[str]
    time_columns: List[str]
    state_columns: List[str]
    action_columns: List[str]
    sha256: str
    size_bytes: int


class DifferencePairSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pair_id: str
    joint: str
    state_column: str
    action_column: str
    formula: str
    unit: str
    eligible: bool = True
    relationship: str
    caveat: str


class ActionSemanticsSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: ActionSemanticsKind
    relationship: str
    is_leader_command: bool
    postprocessing_note: str
    meaning: str


class StateSemanticsSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str
    meaning: str
    units_by_joint: Dict[str, str]


class TimingSemanticsSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nominal_fps: float
    primary_index_column: str = "frame_index"
    primary_time_column: str
    secondary_time_column: Optional[str] = None
    has_wall_clock: bool
    has_video_pts: bool = True
    physical_sync_uncertainty_ms: Optional[float] = None
    timing_map_revision: str = "tmap_v1_1_1to1"
    alignment_note: str


class EpisodeCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid")

    permitted_signals: List[str]
    permitted_difference_pairs: List[str]
    available_views: List[str]
    missing_information: List[str]


class EpisodePermissions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    local_inspection_allowed: bool = True
    cloud_model_egress_allowed: bool = True
    raw_media_export_allowed: bool = False
    split_declaration: str = "unassigned_prepared_catalog"


class EpisodeManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["lablens.episode.v1_1"] = "lablens.episode.v1_1"
    episode_id: str
    episode_revision: str
    display_title: str
    category_group: Literal["policy_failure", "policy_success", "teleop_success"]
    source_type: EpisodeSourceType
    source_kind: EpisodeSourceType
    readiness: ReadinessStatus = "registered"
    task_label: str
    task_text: str
    robot_id: str
    policy_or_session_id: Optional[str] = None
    checkpoint_label: Optional[str] = None
    checkpoint_note: Optional[str] = None
    outcome_annotation: OutcomeAnnotation
    cameras: List[CameraSpec]
    trajectory: TrajectorySpec
    state_semantics: StateSemanticsSpec
    action_semantics: ActionSemanticsSpec
    timing_semantics: TimingSemanticsSpec
    difference_pairs: List[DifferencePairSpec]
    capabilities: EpisodeCapabilities
    permissions: EpisodePermissions = Field(default_factory=EpisodePermissions)
    compatible_comparators: List[str] = Field(default_factory=list)
    validation_status: Literal["unvalidated", "validated", "invalid"] = "unvalidated"
    validation_errors: List[str] = Field(default_factory=list)
    validated_at: Optional[str] = None
    has_saved_analysis: bool = False
    is_analyzed: bool = False


class CatalogCounts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reported_prepared: int = 16
    registered: int
    validated: int
    analyzed: int
    policy_rollout_total: int
    policy_success_count: int
    policy_failure_count: int
    teleop_total: int
    teleop_success_count: int


class CatalogEntrySummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str
    episode_revision: str
    display_title: str
    category_group: Literal["policy_failure", "policy_success", "teleop_success"]
    source_type: EpisodeSourceType
    source_kind: EpisodeSourceType
    readiness: ReadinessStatus
    task_label: str
    robot_id: str
    checkpoint_label: Optional[str] = None
    outcome_status: ReportedOutcomeStatus
    outcome_provenance: OutcomeProvenance
    failure_stage: Optional[str] = None
    success_detail: Optional[str] = None
    episode_notes: Optional[str] = None
    ai_judge_verdict: Optional[str] = None
    ground_truth_verdict: Optional[str] = None
    row_count: int
    duration_s: float
    nominal_fps: float
    validation_status: Literal["unvalidated", "validated", "invalid"]
    has_saved_analysis: bool
    is_analyzed: bool


class CatalogResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["lablens.catalog.v1_1"] = "lablens.catalog.v1_1"
    dataset_name: str
    counts: CatalogCounts
    counting_note: str
    episodes: List[CatalogEntrySummary]


# --- 3. Evidence & Source Reference Schemas ---

class SourceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str
    episode_revision: str
    artifact_kind: Literal["trajectory_csv", "video_mp4", "episode_manifest"]
    path: str
    sha256: str
    row_start: Optional[int] = None
    row_end: Optional[int] = None
    columns: List[str] = Field(default_factory=list)
    formula: Optional[str] = None


class FrameRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_id: str
    episode_id: str
    episode_revision: str
    camera: str
    camera_id: str
    sample_index: int
    presentation_frame_index: int
    requested_time_s: Optional[float] = None
    time_s: float
    pts: Optional[int] = None
    timing_map_revision: str = "tmap_v1_1_1to1"
    source_video_path: str
    source_video_sha256: str
    decoded_frame_sha256: str
    width: int
    height: int
    resolution_tier: Literal["overview", "detail"] = "overview"
    mime_type: Literal["image/jpeg", "image/png"] = "image/jpeg"
    byte_size: int
    image_url: str


class SignalSummaryMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    signal: str
    unit: str
    start_sample: int
    end_sample: int
    sample_count: int
    start_value: float
    end_value: float
    min_value: float
    max_value: float
    mean_value: float
    delta: float
    abs_delta_max: float
    peak_sample: int
    peak_time_s: float


class DifferencePairMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pair_id: str
    joint: str
    state_column: str
    action_column: str
    formula: str
    unit: str
    start_sample: int
    end_sample: int
    sample_count: int
    mean_signed_diff: float
    mean_abs_diff: float
    max_abs_diff: float
    min_signed_diff: float
    max_signed_diff: float
    peak_sample: int
    peak_time_s: float
    relationship: str
    caveat: str


class NumericalEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    episode_id: str
    episode_revision: str
    kind: Literal["signal_summary", "difference_pair"]
    label: str
    summary_text: str
    start_sample: int
    end_sample: int
    start_time_s: float
    end_time_s: float
    unit: str
    signal_metrics: Optional[SignalSummaryMetrics] = None
    difference_metrics: Optional[DifferencePairMetrics] = None
    caveat: str
    source_ref: SourceRef


class FrameEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    episode_id: str
    episode_revision: str
    camera: str
    sample_index: int
    time_s: float
    label: str
    caption: str
    frame_ref: FrameRef
    source_ref: SourceRef


# --- 4. Tool Request, Response & Receipt Schemas ---

class QuerySignalWindowArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str
    episode_revision: Optional[str] = None
    start_sample: int = Field(ge=0)
    end_sample: int = Field(ge=0)
    signals: List[str] = Field(min_length=1, max_length=12)
    include_State_action_diff: bool = True
    difference_pair_id: Optional[str] = None
    requested_difference_pairs: Optional[List[str]] = None

    @model_validator(mode="after")
    def check_bounds(self) -> "QuerySignalWindowArgs":
        if self.end_sample < self.start_sample:
            raise ValueError("end_sample must be >= start_sample")
        return self


class QuerySignalWindowResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: Literal["query_signal_window"] = "query_signal_window"
    tool_version: str = "t1.v1_1"
    episode_id: str
    episode_revision: str
    start_sample: int
    end_sample: int
    start_time_s: float
    end_time_s: float
    row_count: int
    signal_summaries: List[SignalSummaryMetrics]
    difference_summaries: List[DifferencePairMetrics]
    rejected_difference_pairs: List[str] = Field(default_factory=list)
    numerical_evidence: List[NumericalEvidence]
    semantics_caveat: str


class GetFrameBundleArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str
    episode_revision: Optional[str] = None
    camera: str = "both"
    camera_ids: Optional[List[str]] = None
    start_sample: int = Field(ge=0)
    end_sample: int = Field(ge=0)
    sample_count: int = Field(default=6, ge=1, le=6)
    resolution_tier: Literal["overview", "detail"] = "overview"

    @model_validator(mode="after")
    def check_bounds(self) -> "GetFrameBundleArgs":
        if self.end_sample < self.start_sample:
            raise ValueError("end_sample must be >= start_sample")
        return self


class GetFrameBundleResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: Literal["get_frame_bundle"] = "get_frame_bundle"
    tool_version: str = "t3.v1_1"
    episode_id: str
    episode_revision: str
    camera: str
    camera_ids: List[str]
    start_sample: int
    end_sample: int
    requested_count_per_camera: int
    delivered_count: int
    sampled_indices: List[int]
    resolution_tier: Literal["overview", "detail"]
    total_bytes: int
    frame_evidence: List[FrameEvidence]
    missing_views: List[str] = Field(default_factory=list)
    sampling_caveat: str


class ToolReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    receipt_id: str
    investigation_id: str
    episode_id: str
    episode_revision: str
    tool_name: Literal["query_signal_window", "get_frame_bundle"]
    tool_version: str
    status: Literal["completed", "rejected_duplicate", "rejected_quota", "failed"]
    started_at: str
    completed_at: str
    duration_ms: float
    request_args: Dict[str, Any]
    source_digests: Dict[str, str]
    produced_evidence_ids: List[str]
    image_count: int = 0
    payload_bytes: int = 0
    note: Optional[str] = None


class ModelReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    receipt_id: str
    investigation_id: str
    episode_id: str
    episode_revision: str
    task_id: Literal["lablens-plan-v1_1", "lablens-review-v1_1"]
    prompt_version: str
    call_number: int = Field(ge=1, le=2)
    model_id: str
    reported_model_id: Optional[str] = None
    sdk_name: str = "google-genai"
    thinking_level: Literal["MINIMAL", "LOW", "MEDIUM", "HIGH"]
    response_schema_name: str
    started_at: str
    completed_at: str
    duration_ms: float
    delivered_image_ids: List[str] = Field(default_factory=list)
    overview_images_sent: int = 0
    detail_images_sent: int = 0
    total_images_sent: int = 0
    request_payload_bytes: int = 0
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    status: Literal["completed", "failed", "simulated_fallback"] = "completed"


# --- 5. Gemini Structured Output Contracts (Task P & Task R) ---

class CandidateIntervalItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    interval_id: str
    start_sample: int = Field(ge=0)
    end_sample: int = Field(ge=0)
    selection_origin: Literal["user_selected", "model_selected_from_overview"]
    reason: str
    delivered_frame_ids: List[str] = Field(default_factory=list)
    visibility_and_coverage_limits: str


class PlanHypothesisItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hypothesis_id: str
    statement: str
    supplied_evidence_refs: List[str] = Field(default_factory=list)
    evidence_needed: str


class InspectionPlanResponse(BaseModel):
    """Structured output contract for Task P (lablens-plan-v1_1)."""
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["lablens.plan.v1_1"] = "lablens.plan.v1_1"
    request_id: str
    investigation_id: str
    episode_id: str
    episode_revision: str
    decision: Literal["inspect", "request_input"] = "inspect"
    annotation_disclosure: AnnotationDisclosureMode
    restated_question: str
    focus_start_sample: int = Field(ge=0)
    focus_end_sample: int = Field(ge=0)
    focus_rationale: str
    candidate_intervals: List[CandidateIntervalItem] = Field(default_factory=list, max_length=2)
    candidate_hypotheses: List[str] = Field(min_length=1, max_length=3)
    hypotheses: List[PlanHypothesisItem] = Field(default_factory=list, max_length=2)
    signal_window_request: QuerySignalWindowArgs
    frame_bundle_request: GetFrameBundleArgs
    unresolved_without_sensors: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list, max_length=3)


class NumericFindingReference(BaseModel):
    """Model-selected reference to server-computed numerical evidence (no model-invented numbers)."""
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    metric_keys: List[str]
    concise_interpretation: str


class ReviewedEventItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    label: str
    start_sample: int
    end_sample: int
    selection_origin: Literal["user_selected", "model_selected_from_overview"] = "user_selected"
    description: str
    uncertainty_note: str = "Sampled interval review suggestion, not exact physical causal onset."
    grounded_evidence_ids: List[str]


class ReviewedVisualFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    finding_id: str
    observation: str
    sampled_frame_evidence_ids: List[str]
    sample_range_label: str
    visibility_and_coverage_limits: str = "Derived from sparse sampled frames; intermediate motion is unobserved."
    is_sampled_observation: bool = True


class ReviewedHypothesisAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hypothesis_id: str
    statement: str
    status: HypothesisAssessmentStatus
    supporting_evidence_ids: List[str] = Field(default_factory=list)
    conflicting_evidence_ids: List[str] = Field(default_factory=list)
    missing_evidence: List[str] = Field(default_factory=list)
    rationale: str


class EvidenceReviewResponse(BaseModel):
    """Structured output contract for Task R (lablens-review-v1_1).
    Strictly contains NO tool-request field.
    """
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["lablens.review.v1_1"] = "lablens.review.v1_1"
    request_id: str
    investigation_id: str
    episode_id: str
    episode_revision: str
    decision: Literal["report", "request_input"] = "report"
    annotation_disclosure: AnnotationDisclosureMode
    direct_answer: str
    computed_fact_citations: List[str]
    numeric_findings: List[NumericFindingReference] = Field(default_factory=list, max_length=6)
    events: List[ReviewedEventItem] = Field(min_length=1, max_length=3)
    visual_findings: List[ReviewedVisualFinding] = Field(min_length=1, max_length=4)
    hypothesis_assessments: List[ReviewedHypothesisAssessment] = Field(min_length=1, max_length=3)
    unresolved_physical_causes: List[str] = Field(min_length=1, max_length=6)
    recommended_manual_checks: List[str] = Field(default_factory=list, max_length=3)
    limitations: List[str] = Field(default_factory=list, max_length=3)
    Semantics_and_annotation_note: str = "Action semantics and supplied outcome annotation disclosed."


# --- 6. Semantic Evidence Map Graph Schemas ---

class EvidenceGraphNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str
    node_type: NodeType
    label: str
    sublabel: str
    status: Optional[str] = None
    episode_id: str
    evidence_id: Optional[str] = None
    event_id: Optional[str] = None
    hypothesis_id: Optional[str] = None
    sample_start: Optional[int] = None
    sample_end: Optional[int] = None
    camera: Optional[str] = None
    inspector_payload: Dict[str, Any] = Field(default_factory=dict)


class EvidenceGraphEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    edge_id: str
    edge_type: EdgeType
    source: str
    target: str
    style: Literal["solid", "dashed"]
    label: str


class EvidenceGraph(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str
    node_count: int = Field(le=20)
    edge_count: int = Field(le=30)
    omitted_items_count: int = 0
    nodes: List[EvidenceGraphNode] = Field(max_length=20)
    edges: List[EvidenceGraphEdge] = Field(max_length=30)


# --- 7. Final Report & Investigation Lifecycle Schemas ---

class BudgetUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    semantic_calls_used: int = 0
    semantic_calls_limit: int = 2
    tool_rounds_used: int = 0
    tool_rounds_limit: int = 1
    tool_requests_used: int = 0
    tool_requests_limit: int = 2
    overview_images_sent: int = 0
    overview_images_limit: int = 12
    detail_images_sent: int = 0
    detail_images_limit: int = 12
    unique_images_sent: int = 0
    unique_images_limit: int = 24
    total_image_transmissions: int = 0
    total_image_transmissions_limit: int = 36
    max_request_bytes_used: int = 0
    max_request_bytes_limit: int = 12 * 1024 * 1024


class FinalReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["lablens.report.v1_1"] = "lablens.report.v1_1"
    report_id: str
    investigation_id: str
    episode_id: str
    episode_revision: str
    analysis_mode: AnalysisMode
    processing_status: Literal["complete", "limited", "partial", "error", "cancelled"] = "complete"
    source_kind: EpisodeSourceType
    annotation_policy: AnnotationDisclosureMode
    created_at: str
    question: str
    focus_start_sample: int
    focus_end_sample: int
    annotation_disclosure: AnnotationDisclosureMode
    outcome_annotation: OutcomeAnnotation
    action_semantics: ActionSemanticsSpec
    timing_semantics: TimingSemanticsSpec
    direct_answer: str
    computed_facts: List[NumericalEvidence]
    numeric_findings: List[NumericFindingReference] = Field(default_factory=list)
    visual_findings: List[ReviewedVisualFinding]
    frame_evidence: List[FrameEvidence]
    overview_frame_evidence: List[FrameEvidence] = Field(default_factory=list)
    events: List[ReviewedEventItem]
    hypotheses: List[ReviewedHypothesisAssessment]
    unresolved_physical_causes: List[str]
    recommended_manual_checks: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    semantics_and_annotation_note: str
    evidence_graph: EvidenceGraph
    plan_response: InspectionPlanResponse
    review_response: EvidenceReviewResponse
    tool_receipts: List[ToolReceipt]
    model_receipts: List[ModelReceipt]
    budget_usage: BudgetUsage


class InvestigationCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str
    question: str = Field(
        default="What happens in the selected window, how do gripper and arm state compare with issued action values, and what physical causes remain unresolved?",
        max_length=4000,
    )
    start_sample: int = Field(default=0, ge=0)
    end_sample: int = Field(default=300, ge=0)
    camera: str = Field(default="both")
    sample_count: int = Field(default=4, ge=1, le=6)
    annotation_disclosure: AnnotationDisclosureMode = "disclosed"
    explicit_authorization: bool = True
    prefer_live_gemini: bool = True
    operation_id: Optional[str] = None

    @model_validator(mode="after")
    def check_bounds(self) -> "InvestigationCreateRequest":
        if self.end_sample < self.start_sample:
            raise ValueError("end_sample must be >= start_sample")
        return self


class Investigation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["lablens.investigation.v1_1"] = "lablens.investigation.v1_1"
    investigation_id: str
    episode_id: str
    episode_revision: str
    status: InvestigationStatus
    analysis_mode: AnalysisMode = "live"
    question: str
    start_sample: int
    end_sample: int
    camera: str
    sample_count: int
    annotation_disclosure: AnnotationDisclosureMode
    explicit_authorization: bool
    created_at: str
    updated_at: str
    stopped_at: Optional[str] = None
    error_message: Optional[str] = None
    budget_usage: BudgetUsage = Field(default_factory=BudgetUsage)
    overview_frame_evidence: List[FrameEvidence] = Field(default_factory=list)
    overview_signal_result: Optional[QuerySignalWindowResult] = None
    plan_response: Optional[InspectionPlanResponse] = None
    signal_window_result: Optional[QuerySignalWindowResult] = None
    frame_bundle_result: Optional[GetFrameBundleResult] = None
    review_response: Optional[EvidenceReviewResponse] = None
    final_report: Optional[FinalReport] = None
    tool_receipts: List[ToolReceipt] = Field(default_factory=list)
    model_receipts: List[ModelReceipt] = Field(default_factory=list)
