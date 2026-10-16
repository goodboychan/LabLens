/**
 * LabLens v1.1 — Robot Episode Investigator Frontend Controller
 * Implements:
 * - Single-episode lazy loading & 4 distinct catalog counters (Gate G01)
 * - Per-episode action semantics & supplied annotation disclosure (Gate G02)
 * - Synchronized dual-camera scrubbing & exact PyAV frame modal (Gate G03)
 * - Full-resolution interactive trajectory + state-minus-action difference canvas (Gate G04)
 * - Explicit operator authorization modal & bounded 2-call ADC Gemini workflow (Gates G05, G06, G09, G10)
 * - Separated 3-column findings: Computed Facts | Visual Interpretations | Unverified Explanations (Gate G07)
 * - Bounded Semantic Evidence Map (compact + full-screen) + accessible node list (Gate G08)
 * - Session + CSRF header attachment & sanitized JSON export (Gate G11)
 */

(function () {
  'use strict';

  const state = {
    sessionToken: null,
    csrfToken: null,
    adcStatus: null,
    catalog: null,
    activeFilter: 'all',
    activeEpisodeId: 'rollout_T2012_1014_231614',
    episodeLoadGeneration: 0,
    manifest: null,
    trajectory: null,
    currentReport: null,
    activeInvestigationId: null,
    isInvestigationRunning: false,
    selectedJoint: 'gripper',
    currentSample: 1560,
    selectedGraphNodeId: null,
    activeNodeTypeFilter: 'all',
    isPlaying: false,
    showAllComputedFacts: false,
  };

  // Default questions tailored to each episode category
  const DEFAULT_QUESTIONS = {
    rollout_T2012_1014_231614:
      'Which interval should I inspect first, and what do the cameras and logs establish?',
    rollout_T1002_1014_214044:
      'Why is rollout_T1002 annotated as a human failure despite a prior AI judge marking YES, and how do gripper/wrist signals behave during placement?',
    rollout_T1001_1014_213934:
      'How do the gripper and wrist trajectories behave during the successful pick-and-place in rollout_T1001?',
    teleop_demo_episode_000001:
      'How do the human leader_command action signals compare against observed follower state in teleop_demo_episode_000001?',
  };

  function getSampleTimes(sampleIdx) {
    const fps =
      (state.manifest &&
        (state.manifest.timing_semantics.nominal_fps || state.manifest.timing_semantics.fps)) ||
      30;
    const videoSec = sampleIdx / fps;
    const wallArr =
      (state.trajectory &&
        (state.trajectory.series['t_wall_s'] ||
          state.trajectory.series[state.trajectory.primary_time_column])) ||
      [];
    const wallSec = wallArr[sampleIdx] !== undefined ? Number(wallArr[sampleIdx]) : videoSec;
    return { videoSec, wallSec, fps };
  }

  function formatVideoAndWallTime(sampleIdx) {
    const { videoSec, wallSec } = getSampleTimes(sampleIdx);
    return `${videoSec.toFixed(2)}s video time (wall ${wallSec.toFixed(2)}s)`;
  }

  function formatRangeVideoAndWallTime(startSample, endSample) {
    const t1 = getSampleTimes(startSample);
    const t2 = getSampleTimes(endSample);
    return `${t1.videoSec.toFixed(2)}s–${t2.videoSec.toFixed(2)}s video time (wall ${t1.wallSec.toFixed(2)}s–${t2.wallSec.toFixed(2)}s)`;
  }

  // Helper for authenticated API calls
  async function apiFetch(url, options = {}) {
    const headers = Object.assign({}, options.headers || {});
    if (state.sessionToken) {
      headers['X-LabLens-Session'] = state.sessionToken;
    }
    if (options.method && options.method !== 'GET' && state.csrfToken) {
      headers['X-LabLens-CSRF'] = state.csrfToken;
    }
    if (options.body && typeof options.body === 'object' && !(options.body instanceof FormData)) {
      headers['Content-Type'] = 'application/json';
      options.body = JSON.stringify(options.body);
    }
    const resp = await fetch(url, Object.assign({}, options, { headers, credentials: 'same-origin' }));
    if (!resp.ok) {
      let detail = `HTTP ${resp.status}`;
      try {
        const errJson = await resp.json();
        detail = errJson.detail || JSON.stringify(errJson);
      } catch (_e) {
        // ignore
      }
      throw new Error(detail);
    }
    return resp.json();
  }

  function escapeHtml(str) {
    if (str === null || str === undefined) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  // =========================================================================
  // 1. Initialization & Session Bootstrap
  // =========================================================================

  async function initApp() {
    bindUIEvents();
    try {
      const sessionData = await apiFetch('/api/session');
      state.sessionToken = sessionData.session_token;
      state.csrfToken = sessionData.csrf_token;
      state.adcStatus = sessionData.adc_status;
      renderAdcStatusBanner();

      await refreshCatalog();
      await selectEpisode(state.activeEpisodeId);
    } catch (err) {
      console.error('Failed to initialize LabLens:', err);
      const ansText = document.getElementById('direct-answer-text');
      if (ansText) {
        ansText.textContent = `Initialization error: ${err.message}`;
      }
    }
  }

  function renderAdcStatusBanner() {
    const banner = document.getElementById('adc-status-banner');
    if (!banner || !state.adcStatus) return;
    const info = state.adcStatus;
    const isAvailable = Boolean(info.adc_available || info.available);
    if (isAvailable) {
      banner.textContent = `ADC Ready (${info.credential_type || 'Application Default Credentials'} · Vertex AI · ${info.location || 'global'})`;
      banner.style.color = '#188038';
    } else {
      banner.textContent = `ADC Mode (Vertex AI · ${info.location || 'global'} · run gcloud auth application-default login for live calls)`;
      banner.style.color = '#B06000';
    }
  }

  // =========================================================================
  // 2. Catalog & 4 Distinct Counters (Gate G01)
  // =========================================================================

  async function refreshCatalog() {
    const catalog = await apiFetch('/api/catalog');
    state.catalog = catalog;
    renderCatalogCounters(catalog.counts);
    renderCatalogList();
  }

  function renderCatalogCounters(counts) {
    if (!counts) return;
    document.getElementById('count-prepared').textContent = counts.reported_prepared;
    document.getElementById('count-registered').textContent = counts.registered;
    document.getElementById('count-validated').textContent = counts.validated;
    document.getElementById('count-analyzed').textContent = counts.analyzed;
    const brk = counts.reported_breakdown || {};
    const pSucc = counts.policy_success_count ?? brk.policy_rollout_success ?? 4;
    const pFail = counts.policy_failure_count ?? brk.policy_rollout_failure ?? 10;
    const tSucc = counts.teleop_success_count ?? brk.teleop_demo_success ?? 2;
    const brkEl = document.getElementById('catalog-breakdown-line');
    if (brkEl) {
      brkEl.textContent = `Reported: ${pSucc} policy success · ${pFail} policy failure · ${tSucc} teleop demo`;
    }
  }

  function matchesFilter(ep, filter) {
    if (filter === 'all') return true;
    const src = ep.source_kind || ep.source_type;
    if (filter === 'policy_failure') {
      return src === 'policy_rollout' && ep.outcome_status === 'failure';
    }
    if (filter === 'policy_success') {
      return src === 'policy_rollout' && ep.outcome_status === 'success';
    }
    if (filter === 'teleop_success') {
      return src === 'teleoperation' || src === 'teleop_demo';
    }
    return true;
  }

  function renderCatalogList() {
    const listEl = document.getElementById('episode-list');
    if (!listEl || !state.catalog) return;
    listEl.innerHTML = '';

    const filtered = state.catalog.episodes.filter((ep) => matchesFilter(ep, state.activeFilter));
    filtered.forEach((ep) => {
      const card = document.createElement('div');
      card.className = 'episode-item-card' + (ep.episode_id === state.activeEpisodeId ? ' selected' : '');
      card.setAttribute('role', 'option');
      card.setAttribute('tabindex', '0');
      card.setAttribute('aria-selected', ep.episode_id === state.activeEpisodeId ? 'true' : 'false');

      const src = ep.source_kind || ep.source_type;
      const outcomeClass = ep.outcome_status === 'success' ? 'badge-teal' : 'badge-coral';
      const sourceBadgeClass = src === 'teleoperation' || src === 'teleop_demo' ? 'badge-amber' : 'badge-blue';
      const valBadge =
        ep.validation_status === 'validated' || ep.validation_status === 'valid'
          ? '<span class="badge-pill badge-teal">validated</span>'
          : '<span class="badge-pill badge-outline">registered</span>';
      const isAnalyzed = ep.is_analyzed ?? ep.analyzed;
      const analyzedBadge = isAnalyzed
        ? `<span class="badge-pill badge-blue">${
            ep.analysis_mode === 'live' || ep.analysis_mode === 'live_gemini' ? 'live' : 'analyzed'
          }</span>`
        : '';

      card.innerHTML = `
        <div class="ep-card-top">
          <span class="ep-id-label">${escapeHtml(ep.episode_id)}</span>
          <span class="badge-pill ${outcomeClass}">${escapeHtml(ep.outcome_status)}</span>
        </div>
        <div class="ep-badges-row">
          <span class="badge-pill ${sourceBadgeClass}">${escapeHtml(src)}</span>
          <span class="badge-pill badge-outline">${escapeHtml(ep.action_kind)}</span>
          ${valBadge}
          ${analyzedBadge}
        </div>
        <p class="ep-notes-preview">${escapeHtml(
          ep.episode_notes || ep.notes || ep.checkpoint_label || ep.display_title || ''
        )}</p>
      `;

      card.addEventListener('click', () => {
        if (ep.episode_id !== state.activeEpisodeId) {
          selectEpisode(ep.episode_id);
        }
      });
      card.addEventListener('keydown', (ev) => {
        if (ev.key === 'Enter' || ev.key === ' ') {
          ev.preventDefault();
          selectEpisode(ep.episode_id);
        }
      });

      listEl.appendChild(card);
    });
  }

  // =========================================================================
  // 3. Single-Episode Lazy Loading & Switching (Gate G01 & G09)
  // =========================================================================

  async function selectEpisode(episodeId) {
    // If an active investigation is running, stop it immediately so late results cannot overwrite (Gate G09)
    if (state.isInvestigationRunning && state.activeInvestigationId) {
      try {
        await apiFetch(`/api/investigations/${state.activeInvestigationId}/stop`, { method: 'POST' });
      } catch (_e) {
        // ignore stop errors
      }
      state.isInvestigationRunning = false;
      state.activeInvestigationId = null;
      document.getElementById('btn-stop-investigation').classList.add('hidden');
    }

    const myGeneration = ++state.episodeLoadGeneration;
    state.activeEpisodeId = episodeId;
    renderCatalogList();

    // Update default question if appropriate
    const qInput = document.getElementById('input-question');
    if (qInput) {
      qInput.value =
        DEFAULT_QUESTIONS[episodeId] ||
        `What happens during the critical manipulation window of ${episodeId}, and how do recorded joint trajectories compare with sampled camera frames?`;
    }

    try {
      const [manifest, trajectory, savedReport] = await Promise.all([
        apiFetch(`/api/episodes/${encodeURIComponent(episodeId)}/manifest`),
        apiFetch(`/api/episodes/${encodeURIComponent(episodeId)}/trajectory`),
        apiFetch(`/api/episodes/${encodeURIComponent(episodeId)}/saved-report`),
      ]);

      // Guard against stale async responses after rapid episode switching
      if (myGeneration !== state.episodeLoadGeneration) return;

      state.manifest = manifest;
      state.trajectory = trajectory;
      state.currentReport = savedReport;

      // Refresh catalog counters because lazy validation or saved report may have incremented counts
      await refreshCatalog();

      applyEpisodeToUI(manifest, trajectory, savedReport);
    } catch (err) {
      console.error('Error loading episode:', err);
      document.getElementById('direct-answer-text').textContent = `Failed to load episode ${episodeId}: ${err.message}`;
    }
  }

  function applyEpisodeToUI(manifest, trajectory, report) {
    const srcKind = manifest.source_kind || manifest.source_type;
    const ruleId =
      manifest.outcome_annotation.annotator_or_rule_id ||
      manifest.outcome_annotation.provenance ||
      'auto+human consensus';
    const shortRuleLabel = ruleId.includes('consensus') ? 'auto+human consensus' : ruleId;

    // Header pills
    document.getElementById('header-episode-pill').textContent = manifest.episode_id;
    document.getElementById('header-semantics-pill').textContent = `${srcKind} · ${manifest.outcome_annotation.status} (${shortRuleLabel})`;

    // Disclosed context strip (Gate G02)
    document.getElementById('disclosure-source-badge').textContent = `Source: ${srcKind} (${
      manifest.checkpoint_label || manifest.policy_or_session_id || manifest.display_title
    })`;
    document.getElementById('disclosure-action-badge').textContent = `Action: ${manifest.action_semantics.kind} (${
      srcKind === 'policy_rollout' ? 'clipped policy output, not human leader' : 'human leader_command'
    })`;
    document.getElementById('disclosure-outcome-badge').textContent = `Supplied outcome: ${manifest.outcome_annotation.status} (rule: ${ruleId})`;
    const opNote =
      manifest.outcome_annotation.episode_notes ||
      (manifest.outcome_annotation.success_Detail !== '—' ? manifest.outcome_annotation.success_Detail : null) ||
      manifest.outcome_annotation.notes ||
      manifest.outcome_annotation.rationale ||
      '—';
    document.getElementById('disclosure-notes-text').textContent = `Operator note: ${opNote}`;
    document.getElementById('legend-action-kind').textContent = manifest.action_semantics.kind;

    // Authorized sample range inputs
    const maxSample = Math.max(0, trajectory.total_rows - 1);
    const startInput = document.getElementById('input-start-sample');
    const endInput = document.getElementById('input-end-sample');
    startInput.max = maxSample;
    endInput.max = maxSample;
    startInput.value = Math.min(maxSample, report.focus_start_sample);
    endInput.value = Math.min(maxSample, report.focus_end_sample);
    updateRangeTimeBadge();

    // Scrubber setup
    const scrubber = document.getElementById('timeline-scrubber');
    scrubber.max = maxSample;
    const midSample = Math.min(
      maxSample,
      Math.round((report.focus_start_sample + report.focus_end_sample) / 2)
    );
    state.currentSample = midSample;
    scrubber.value = midSample;

    // Load dual camera videos (only for the selected episode!)
    const videoTop = document.getElementById('video-top');
    const videoWrist = document.getElementById('video-wrist');
    videoTop.pause();
    videoWrist.pause();
    state.isPlaying = false;
    document.getElementById('btn-play-pause').textContent = '▶ Play';

    videoTop.src = `/api/episodes/${encodeURIComponent(manifest.episode_id)}/video/top`;
    videoWrist.src = `/api/episodes/${encodeURIComponent(manifest.episode_id)}/video/wrist`;

    const fps = manifest.timing_semantics.nominal_fps || manifest.timing_semantics.fps || 30;
    const targetSec = midSample / fps;
    videoTop.onloadedmetadata = () => {
      videoTop.currentTime = Math.min(videoTop.duration || targetSec, targetSec);
    };
    videoWrist.onloadedmetadata = () => {
      videoWrist.currentTime = Math.min(videoWrist.duration || targetSec, targetSec);
    };

    const topSpec = manifest.cameras.find((c) => c.camera_id === 'top');
    const wristSpec = manifest.cameras.find((c) => c.camera_id === 'wrist');
    if (topSpec) {
      document.getElementById('top-cam-meta').textContent = `${topSpec.width}×${topSpec.height} · ${
        topSpec.fps_nominal || topSpec.fps || 30
      } fps · ${topSpec.frame_count}f`;
    }
    if (wristSpec) {
      document.getElementById('wrist-cam-meta').textContent = `${wristSpec.width}×${wristSpec.height} · ${
        wristSpec.fps_nominal || wristSpec.fps || 30
      } fps · ${wristSpec.frame_count}f`;
    }

    renderReportToUI(report);
    setSampleCursor(midSample, false);
  }

  function updateRangeTimeBadge() {
    if (!state.manifest) return;
    const s = parseInt(document.getElementById('input-start-sample').value || '0', 10);
    const e = parseInt(document.getElementById('input-end-sample').value || '0', 10);
    const badge = document.getElementById('range-time-badge');
    if (badge) {
      badge.textContent = formatRangeVideoAndWallTime(s, e);
    }
  }

  // =========================================================================
  // 4. Synchronized Video Scrubbing & Sample Cursor
  // =========================================================================

  function setSampleCursor(sampleIndex, seekVideos = true) {
    if (!state.trajectory || !state.manifest) return;
    const maxSample = Math.max(0, state.trajectory.total_rows - 1);
    const clamped = Math.max(0, Math.min(maxSample, Math.round(sampleIndex)));
    state.currentSample = clamped;

    const fps = state.manifest.timing_semantics.nominal_fps || state.manifest.timing_semantics.fps || 30;
    const timeSec = clamped / fps;
    const timeFormatted = formatVideoAndWallTime(clamped);

    const scrubber = document.getElementById('timeline-scrubber');
    if (scrubber) scrubber.value = clamped;

    document.getElementById('scrubber-readout').textContent = `Sample ${clamped} / ${maxSample} · ${timeFormatted}`;
    document.getElementById('top-sample-pill').textContent = `sample ${clamped} · ${timeFormatted}`;
    document.getElementById('wrist-sample-pill').textContent = `sample ${clamped} · ${timeFormatted}`;

    if (seekVideos && !state.isPlaying) {
      const videoTop = document.getElementById('video-top');
      const videoWrist = document.getElementById('video-wrist');
      if (videoTop && videoTop.readyState >= 1) {
        videoTop.currentTime = Math.min(videoTop.duration || timeSec, timeSec);
      }
      if (videoWrist && videoWrist.readyState >= 1) {
        videoWrist.currentTime = Math.min(videoWrist.duration || timeSec, timeSec);
      }
    }

    updateCursorLiveReadout(clamped);
    drawTrajectoryCanvas();
    renderCsvSliceTable(clamped);
  }

  function updateCursorLiveReadout(sampleIdx) {
    if (!state.trajectory) return;
    const joint = state.selectedJoint;
    const sArr = state.trajectory.series[`state.${joint}`] || [];
    const aArr = state.trajectory.series[`action.${joint}`] || [];
    const dArr =
      state.trajectory.difference_series[`diff_${joint}`] ||
      state.trajectory.difference_series[`diff.${joint}`] ||
      [];
    const sVal = sArr[sampleIdx] !== undefined ? sArr[sampleIdx].toFixed(2) : '—';
    const aVal = aArr[sampleIdx] !== undefined ? aArr[sampleIdx].toFixed(2) : '—';
    const dVal = dArr[sampleIdx] !== undefined ? dArr[sampleIdx].toFixed(2) : '—';
    const pill = document.getElementById('cursor-live-values');
    if (pill) {
      pill.textContent = `state=${sVal} · action=${aVal} · Δ=${dVal}`;
    }
  }

  // =========================================================================
  // 5. Interactive Reference Trajectory & Difference Plot (Canvas)
  // =========================================================================

  function drawTrajectoryCanvas() {
    const canvas = document.getElementById('trajectory-canvas');
    if (!canvas || !state.trajectory) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;

    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#FFFFFF';
    ctx.fillRect(0, 0, W, H);

    const total = state.trajectory.total_rows;
    if (total <= 1) return;

    const padL = 48;
    const padR = 16;
    const padT = 14;
    const padB = 24;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;

    const joint = state.selectedJoint;
    const stateSeries = state.trajectory.series[`state.${joint}`] || [];
    const actionSeries = state.trajectory.series[`action.${joint}`] || [];
    const diffSeries =
      state.trajectory.difference_series[`diff_${joint}`] ||
      state.trajectory.difference_series[`diff.${joint}`] ||
      [];

    // Compute min/max across state, action, diff
    let minVal = Infinity;
    let maxVal = -Infinity;
    for (let i = 0; i < total; i++) {
      const sv = stateSeries[i];
      const av = actionSeries[i];
      const dv = diffSeries[i];
      if (sv !== undefined) {
        if (sv < minVal) minVal = sv;
        if (sv > maxVal) maxVal = sv;
      }
      if (av !== undefined) {
        if (av < minVal) minVal = av;
        if (av > maxVal) maxVal = av;
      }
      if (dv !== undefined) {
        if (dv < minVal) minVal = dv;
        if (dv > maxVal) maxVal = dv;
      }
    }
    if (!isFinite(minVal) || !isFinite(maxVal)) {
      minVal = -50;
      maxVal = 50;
    }
    if (Math.abs(maxVal - minVal) < 1e-3) {
      maxVal += 10;
      minVal -= 10;
    }
    const margin = (maxVal - minVal) * 0.08;
    minVal -= margin;
    maxVal += margin;

    const xOf = (idx) => padL + (idx / (total - 1)) * plotW;
    const yOf = (val) => padT + plotH - ((val - minVal) / (maxVal - minVal)) * plotH;

    // Shaded authorized scan window
    const authStart = parseInt(document.getElementById('input-start-sample').value || '0', 10);
    const authEnd = parseInt(document.getElementById('input-end-sample').value || '0', 10);
    const xStart = xOf(Math.max(0, Math.min(total - 1, authStart)));
    const xEnd = xOf(Math.max(0, Math.min(total - 1, authEnd)));
    ctx.fillStyle = 'rgba(26, 115, 232, 0.07)';
    ctx.fillRect(xStart, padT, Math.max(2, xEnd - xStart), plotH);
    ctx.strokeStyle = 'rgba(26, 115, 232, 0.35)';
    ctx.lineWidth = 1;
    ctx.strokeRect(xStart, padT, Math.max(2, xEnd - xStart), plotH);

    // Highlight localized report focus window if distinct
    if (
      state.currentReport &&
      state.currentReport.focus_start_sample !== undefined &&
      state.currentReport.focus_end_sample !== undefined
    ) {
      const fStart = xOf(Math.max(0, Math.min(total - 1, state.currentReport.focus_start_sample)));
      const fEnd = xOf(Math.max(0, Math.min(total - 1, state.currentReport.focus_end_sample)));
      ctx.fillStyle = 'rgba(249, 171, 0, 0.14)';
      ctx.fillRect(fStart, padT, Math.max(2, fEnd - fStart), plotH);
      ctx.strokeStyle = 'rgba(176, 96, 0, 0.55)';
      ctx.lineWidth = 1.2;
      ctx.strokeRect(fStart, padT, Math.max(2, fEnd - fStart), plotH);
    }

    // Horizontal grid lines
    ctx.strokeStyle = '#E2E8F0';
    ctx.lineWidth = 1;
    ctx.fillStyle = '#475569';
    ctx.font = '11px Consolas, monospace';
    for (let g = 0; g <= 4; g++) {
      const gv = minVal + ((maxVal - minVal) * g) / 4;
      const gy = yOf(gv);
      ctx.beginPath();
      ctx.moveTo(padL, gy);
      ctx.lineTo(padL + plotW, gy);
      ctx.stroke();
      ctx.fillText(gv.toFixed(1), 4, gy + 3);
    }

    // Zero reference line if in range
    if (minVal < 0 && maxVal > 0) {
      const yZero = yOf(0);
      ctx.strokeStyle = '#94A3B8';
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(padL, yZero);
      ctx.lineTo(padL + plotW, yZero);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Helper to draw series line
    function drawSeries(arr, color, width, dashed = false) {
      if (!arr || !arr.length) return;
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.setLineDash(dashed ? [4, 3] : []);
      ctx.beginPath();
      for (let i = 0; i < arr.length; i++) {
        const px = xOf(i);
        const py = yOf(arr[i]);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Draw difference (Amber), action (Google Blue), and state (Google Green)
    drawSeries(diffSeries, '#B06000', 1.5, true);
    drawSeries(actionSeries, '#1A73E8', 1.8, false);
    drawSeries(stateSeries, '#188038', 2.2, false);

    // Draw event markers from current report
    if (state.currentReport && Array.isArray(state.currentReport.events)) {
      state.currentReport.events.forEach((ev) => {
        const ex = xOf(Math.max(0, Math.min(total - 1, ev.start_sample)));
        ctx.strokeStyle = 'rgba(24, 128, 56, 0.45)';
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(ex, padT);
        ctx.lineTo(ex, padT + plotH);
        ctx.stroke();
      });
    }

    // Current sample cursor line (Google Red)
    const curX = xOf(state.currentSample);
    ctx.strokeStyle = '#D93025';
    ctx.lineWidth = 2.0;
    ctx.beginPath();
    ctx.moveTo(curX, padT);
    ctx.lineTo(curX, padT + plotH);
    ctx.stroke();

    // Sample axis ticks
    ctx.fillStyle = '#334155';
    ctx.font = '11px Consolas, monospace';
    const tickCount = 6;
    for (let t = 0; t <= tickCount; t++) {
      const sIdx = Math.round(((total - 1) * t) / tickCount);
      const tx = xOf(sIdx);
      ctx.fillText(`#${sIdx}`, tx - 14, H - 6);
    }
  }

  // =========================================================================
  // 6. Render Report, 3 Findings Cards, and Semantic Evidence Map
  // =========================================================================

  function renderReportToUI(report) {
    if (!report) return;

    const firstModelRec = (report.model_receipts || [])[0] || {};
    const modelId =
      firstModelRec.reported_model_id ||
      firstModelRec.model_id ||
      'gemini-4-argon';
    const subheaderModelEl = document.getElementById('subheader-model-id');
    if (subheaderModelEl) {
      subheaderModelEl.textContent = modelId;
    }

    // Top status strip & mode badge
    const modeBanner = document.getElementById('analysis-mode-banner');
    const reportModeBadge = document.getElementById('report-mode-badge');
    if (report.analysis_mode === 'live' || report.analysis_mode === 'live_gemini') {
      modeBanner.textContent = `Mode: Live Gemini (${modelId})`;
      modeBanner.className = 'mode-pill mode-live';
      reportModeBadge.textContent = `LIVE GEMINI ANALYSIS (${modelId})`;
      reportModeBadge.className = 'badge-pill badge-teal';
    } else {
      modeBanner.textContent = 'Mode: Saved analysis';
      modeBanner.className = 'mode-pill mode-saved';
      reportModeBadge.textContent = 'SAVED REFERENCE ANALYSIS';
      reportModeBadge.className = 'badge-pill badge-blue';
    }

    const annPolicy = report.annotation_disclosure || report.annotation_policy || 'disclosed';
    document.getElementById('annotation-policy-banner').textContent = `Annotation policy: ${annPolicy}`;
    const b = report.budget_usage || {};
    const callsUsed = b.semantic_calls_used ?? b.gemini_calls_used ?? 0;
    const maxCalls = b.semantic_calls_limit ?? b.max_gemini_calls ?? 2;
    const roundsUsed = b.tool_rounds_used ?? 1;
    const maxRounds = b.tool_rounds_limit ?? b.max_tool_rounds ?? 1;
    const framesUsed = b.total_image_transmissions ?? b.delivered_frames_used ?? 0;
    const maxFrames = b.unique_images_limit ?? b.max_total_frames ?? 24;
    document.getElementById('budget-strip-banner').textContent = `Budget: ${callsUsed}/${maxCalls} calls · ${roundsUsed}/${maxRounds} tool round · ${framesUsed}/${maxFrames} frames`;

    document.getElementById('report-focus-badge').textContent = `Focus samples: ${report.focus_start_sample}–${report.focus_end_sample} · ${formatRangeVideoAndWallTime(
      report.focus_start_sample,
      report.focus_end_sample
    )}`;
    document.getElementById('report-receipts-badge').textContent = `Model: ${modelId} · ${report.tool_receipts.length} tool receipts · ${report.model_receipts.length} model receipts`;
    document.getElementById('direct-answer-text').textContent = report.direct_answer;

    // 1. Computed Facts (Gate G07) — deduplicated by evidence_id (P0.5) & collapsible >3 (P1.2)
    const factsList = document.getElementById('computed-facts-list');
    factsList.innerHTML = '';

    const mergedFactsMap = new Map();
    (report.computed_facts || []).forEach((cf) => {
      const evId = cf.evidence_id || `cf_${mergedFactsMap.size + 1}`;
      const cfText =
        cf.summary_text ||
        `${cf.label || cf.signal_or_pair || ''}: ${cf.value ?? ''} ${cf.unit || ''}`.trim();
      mergedFactsMap.set(evId, {
        evidence_id: evId,
        text: cfText,
        start_sample: cf.start_sample ?? report.focus_start_sample,
        end_sample: cf.end_sample ?? report.focus_end_sample,
        metrics_label: cf.signal_or_pair || '',
      });
    });

    (report.numeric_findings || []).forEach((nf) => {
      const evId =
        nf.evidence_id || (nf.evidence_ids && nf.evidence_ids[0]) || `nf_${mergedFactsMap.size + 1}`;
      const nfText = nf.concise_interpretation || nf.claim_text || '';
      const nfMetrics = (nf.metric_keys || nf.metric_names || []).join(', ');
      if (mergedFactsMap.has(evId)) {
        const existing = mergedFactsMap.get(evId);
        if (nfText && nfText.length >= existing.text.length) {
          existing.text = nfText;
        }
        if (nfMetrics && !existing.metrics_label) {
          existing.metrics_label = nfMetrics;
        }
      } else {
        const sr = nf.sample_range || [];
        mergedFactsMap.set(evId, {
          evidence_id: evId,
          text: nfText,
          start_sample: sr[0] ?? report.focus_start_sample,
          end_sample: sr[1] ?? report.focus_end_sample,
          metrics_label: nfMetrics || 'computed metrics',
        });
      }
    });

    const mergedFacts = Array.from(mergedFactsMap.values());
    mergedFacts.forEach((fact, idx) => {
      const item = document.createElement('div');
      item.className =
        'finding-item' + (!state.showAllComputedFacts && idx >= 3 ? ' hidden' : '');
      item.dataset.factIndex = String(idx);
      const sStart = fact.start_sample;
      const sEnd = fact.end_sample;
      const metaLabel = fact.metrics_label
        ? `${fact.metrics_label} · samples ${sStart}–${sEnd}`
        : `samples ${sStart}–${sEnd}`;
      item.innerHTML = `
        <p class="finding-item-text">${escapeHtml(fact.text)}</p>
        <div class="finding-item-meta">
          <span class="badge-pill badge-blue">${escapeHtml(fact.evidence_id)}</span>
          <span>${escapeHtml(metaLabel)}</span>
          <span>Calculation checked</span>
        </div>
      `;
      item.addEventListener('click', () => {
        setSampleCursor(Math.round((sStart + sEnd) / 2), true);
        selectGraphNode(`node_ev_${fact.evidence_id}`);
      });
      factsList.appendChild(item);
    });

    const toggleFactsBtn = document.getElementById('btn-toggle-computed-facts');
    if (toggleFactsBtn) {
      if (mergedFacts.length > 3) {
        toggleFactsBtn.classList.remove('hidden');
        toggleFactsBtn.textContent = state.showAllComputedFacts
          ? 'Show first 3'
          : `Show all (${mergedFacts.length})`;
      } else {
        toggleFactsBtn.classList.add('hidden');
      }
    }

    // 2. Visual Interpretations (Gate G07)
    const visualList = document.getElementById('visual-findings-list');
    visualList.innerHTML = '';
    (report.visual_findings || []).forEach((vf) => {
      const item = document.createElement('div');
      item.className = 'finding-item';
      const vfText = vf.observation || vf.claim_text || '';
      const vfEvIds = vf.sampled_frame_evidence_ids || vf.evidence_ids || [];
      const vfRange = vf.sample_range_label || `samples ${(vf.sample_indices || []).join(', ')}`;
      item.innerHTML = `
        <p class="finding-item-text">${escapeHtml(vfText)}</p>
        <div class="finding-item-meta">
          <span class="badge-pill badge-teal">${escapeHtml(vfEvIds[0] || vf.finding_id || 'frame')}</span>
          <span>${escapeHtml(vfRange)}</span>
          <span>Sampled visual interpretation</span>
        </div>
      `;
      item.addEventListener('click', () => {
        if (vfEvIds[0]) {
          selectGraphNode(`node_ev_${vfEvIds[0]}`);
        }
      });
      visualList.appendChild(item);
    });

    // 3. Unverified Explanations & Hypotheses (Gate G07)
    const unverifiedList = document.getElementById('unverified-explanations-list');
    unverifiedList.innerHTML = '';
    (report.hypotheses || []).forEach((hyp) => {
      const item = document.createElement('div');
      item.className = 'finding-item';
      const missingText = (hyp.missing_evidence || hyp.cannot_verify_from_sources || []).join('; ');
      item.innerHTML = `
        <p class="finding-item-text"><strong>[${escapeHtml(hyp.status.toUpperCase())}]</strong> ${escapeHtml(hyp.statement)}</p>
        <div class="finding-item-meta">
          <span class="badge-pill badge-amber">${escapeHtml(hyp.hypothesis_id)}</span>
          <span>Cause unresolved · Missing: ${escapeHtml(missingText || 'force/torque sensor')}</span>
        </div>
      `;
      item.addEventListener('click', () => {
        selectGraphNode(`node_hyp_${hyp.hypothesis_id}`);
      });
      unverifiedList.appendChild(item);
    });

    (report.unresolved_physical_causes || []).forEach((cause) => {
      const item = document.createElement('div');
      item.className = 'finding-item';
      item.innerHTML = `
        <p class="finding-item-text">${escapeHtml(cause)}</p>
        <div class="finding-item-meta">
          <span class="badge-pill badge-amber">Unresolved physical cause</span>
        </div>
      `;
      unverifiedList.appendChild(item);
    });

    // Render Semantic Evidence Graph (both compact and full-screen views)
    renderEvidenceGraphs(report.evidence_graph);

    // Update Verification Loop bottom bar (Tab 2)
    updateVerificationLoopBar(report);

    // Select first Evidence or Event node by default
    const defaultNode =
      report.evidence_graph.nodes.find((n) => n.node_type === 'Evidence') ||
      report.evidence_graph.nodes[0];
    if (defaultNode) {
      selectGraphNode(defaultNode.node_id, false);
    }
  }

  // =========================================================================
  // 7. Semantic Evidence Graph Renderer (Compact + Full View + Accessible List)
  // =========================================================================

  function isNumericalEvidenceNode(n) {
    return (
      n &&
      n.node_type === 'Evidence' &&
      (n.status === 'computed_record' ||
        String(n.node_id || '').includes('ev_num') ||
        String(n.evidence_id || '').includes('ev_num'))
    );
  }

  function computeGraphPositions(nodes, width, height, isCompact) {
    const positions = {};
    const byType = {
      Episode: [],
      Event: [],
      Hypothesis: [],
      NumEvidence: [],
      FrameEvidence: [],
    };
    nodes.forEach((n) => {
      if (n.node_type === 'Episode') byType.Episode.push(n);
      else if (n.node_type === 'Event') byType.Event.push(n);
      else if (n.node_type === 'Hypothesis') byType.Hypothesis.push(n);
      else if (isNumericalEvidenceNode(n)) byType.NumEvidence.push(n);
      else byType.FrameEvidence.push(n);
    });

    const numEvList = isCompact ? byType.NumEvidence.slice(0, 4) : byType.NumEvidence.slice(0, 5);
    const frameEvList = byType.FrameEvidence.slice(0, 4);

    const rows = isCompact
      ? [
          { list: byType.Episode, y: 38, w: 196, h: 42 },
          { list: byType.Event, y: 112, w: 162, h: 44 },
          { list: byType.Hypothesis, y: 194, w: 166, h: 48 },
          { list: numEvList, y: 286, w: 118, h: 46 },
          { list: frameEvList, y: 376, w: 118, h: 46 },
        ]
      : [
          { list: byType.Episode, y: 62, w: 280, h: 58 },
          { list: byType.Event, y: 178, w: 280, h: 66 },
          { list: byType.Hypothesis, y: 306, w: 300, h: 74 },
          { list: numEvList, y: 448, w: 204, h: 68 },
          { list: frameEvList, y: 576, w: 224, h: 68 },
        ];

    rows.forEach((row) => {
      const count = row.list.length;
      if (!count) return;
      if (count === 1) {
        positions[row.list[0].node_id] = {
          x: isCompact && row.y < 60 ? Math.min(width / 2, width - 210) : width / 2,
          y: row.y,
          w: row.w,
          h: row.h,
        };
        return;
      }
      const padX = row.w / 2 + (isCompact ? 10 : 24);
      const usableW = Math.max(10, width - 2 * padX);
      row.list.forEach((node, idx) => {
        positions[node.node_id] = {
          x: padX + (usableW * idx) / (count - 1),
          y: row.y,
          w: row.w,
          h: row.h,
        };
      });
    });

    return positions;
  }

  function getNodePalette(nodeOrType) {
    const nodeType = typeof nodeOrType === 'string' ? nodeOrType : nodeOrType.node_type;
    const isNumEv = typeof nodeOrType === 'object' && isNumericalEvidenceNode(nodeOrType);
    switch (nodeType) {
      case 'Episode':
        return { fill: '#E8F0FE', stroke: '#1A73E8', badge: '#174EA6', text: '#0F172A' };
      case 'Event':
        return { fill: '#E6F4EA', stroke: '#188038', badge: '#137333', text: '#0F172A' };
      case 'Hypothesis':
        return { fill: '#FEF7E0', stroke: '#D97706', badge: '#B06000', text: '#0F172A' };
      case 'Evidence':
      default:
        if (isNumEv) {
          return { fill: '#E8F0FE', stroke: '#1A73E8', badge: '#174EA6', text: '#0F172A' };
        }
        return { fill: '#E6F4EA', stroke: '#188038', badge: '#137333', text: '#0F172A' };
    }
  }

  function wrapSvgLabelLines(label, maxCharsPerLine) {
    const clean = String(label || '').trim();
    if (clean.length <= maxCharsPerLine) return [clean];
    const words = clean.split(/\s+/);
    let line1 = '';
    let idx = 0;
    while (idx < words.length) {
      const next = line1 ? `${line1} ${words[idx]}` : words[idx];
      if (next.length <= maxCharsPerLine || !line1) {
        line1 = next.slice(0, maxCharsPerLine);
        idx++;
      } else {
        break;
      }
    }
    const rest = words.slice(idx).join(' ');
    if (!rest) return [line1];
    const line2 =
      rest.length > maxCharsPerLine ? rest.slice(0, maxCharsPerLine - 1) + '…' : rest;
    return [line1, line2];
  }

  function renderEvidenceGraphs(graph) {
    if (!graph) return;
    document.getElementById('graph-counts-pill').textContent = `${graph.nodes.length} nodes · ${graph.edges.length} edges`;

    renderSingleSvgGraph(
      graph,
      'compact-graph-edges',
      'compact-graph-nodes',
      540,
      440,
      true
    );
    renderSingleSvgGraph(
      graph,
      'large-graph-edges',
      'large-graph-nodes',
      1160,
      660,
      false
    );
    renderAccessibleNodeList(graph);
  }

  function renderSingleSvgGraph(graph, edgesGroupId, nodesGroupId, width, height, isCompact) {
    const edgesGroup = document.getElementById(edgesGroupId);
    const nodesGroup = document.getElementById(nodesGroupId);
    if (!edgesGroup || !nodesGroup) return;
    edgesGroup.innerHTML = '';
    nodesGroup.innerHTML = '';

    const visibleNodes = graph.nodes.filter(
      (n) => state.activeNodeTypeFilter === 'all' || n.node_type === state.activeNodeTypeFilter || n.node_type === 'Episode'
    );
    const positions = computeGraphPositions(visibleNodes, width, height, isCompact);
    const positionedNodes = visibleNodes.filter((n) => Boolean(positions[n.node_id]));
    const visibleIds = new Set(positionedNodes.map((n) => n.node_id));

    const ns = 'http://www.w3.org/2000/svg';
    const solidMarker = isCompact ? 'url(#arrow-solid-compact)' : 'url(#arrow-solid-large)';
    const dashedMarker = isCompact ? 'url(#arrow-dashed-compact)' : 'url(#arrow-dashed-large)';

    // Draw edges
    graph.edges.forEach((edge) => {
      const srcId = edge.source || edge.source_id;
      const tgtId = edge.target || edge.target_id;
      if (!visibleIds.has(srcId) || !visibleIds.has(tgtId)) return;
      const p1 = positions[srcId];
      const p2 = positions[tgtId];
      if (!p1 || !p2) return;

      const isDashed =
        edge.style === 'dashed' || edge.edge_type === 'INVESTIGATES' || edge.edge_type === 'BEARS_ON';
      const isConnectedToSelected =
        srcId === state.selectedGraphNodeId || tgtId === state.selectedGraphNodeId;

      const line = document.createElementNS(ns, 'path');
      const startY = p1.y < p2.y ? p1.y + p1.h / 2 : p1.y - p1.h / 2;
      const endY = p1.y < p2.y ? p2.y - p2.h / 2 : p2.y + p2.h / 2;
      const midY = (startY + endY) / 2;
      const d = `M ${p1.x} ${startY} C ${p1.x} ${midY}, ${p2.x} ${midY}, ${p2.x} ${endY}`;

      line.setAttribute('d', d);
      line.setAttribute('fill', 'none');
      line.setAttribute('stroke', isDashed ? '#D97706' : '#475569');
      line.setAttribute(
        'stroke-width',
        isConnectedToSelected ? (isCompact ? '2.2' : '2.6') : isCompact ? '1.3' : '1.6'
      );
      line.setAttribute('stroke-opacity', isConnectedToSelected ? '0.95' : '0.45');
      if (isDashed) {
        line.setAttribute('stroke-dasharray', '5 4');
        line.setAttribute('marker-end', dashedMarker);
      } else {
        line.setAttribute('marker-end', solidMarker);
      }
      edgesGroup.appendChild(line);
    });

    // Draw nodes
    positionedNodes.forEach((node) => {
      const pos = positions[node.node_id];
      if (!pos) return;
      const pal = getNodePalette(node);

      const g = document.createElementNS(ns, 'g');
      g.setAttribute(
        'class',
        'svg-node-group' + (node.node_id === state.selectedGraphNodeId ? ' selected' : '')
      );
      g.setAttribute('data-node-id', node.node_id);
      g.setAttribute('tabindex', '0');
      g.setAttribute('role', 'button');
      g.setAttribute('aria-label', `${node.node_type}: ${node.label}`);

      const rx = pos.x - pos.w / 2;
      const ry = pos.y - pos.h / 2;

      const rect = document.createElementNS(ns, 'rect');
      rect.setAttribute('class', 'node-card-rect');
      rect.setAttribute('x', String(rx));
      rect.setAttribute('y', String(ry));
      rect.setAttribute('width', String(pos.w));
      rect.setAttribute('height', String(pos.h));
      rect.setAttribute('rx', isCompact ? '8' : '10');
      rect.setAttribute('fill', pal.fill);
      rect.setAttribute('stroke', pal.stroke);
      rect.setAttribute('stroke-width', node.node_id === state.selectedGraphNodeId ? '2.8' : '1.5');
      g.appendChild(rect);

      // Type eyebrow text
      const typeTxt = document.createElementNS(ns, 'text');
      typeTxt.setAttribute('x', String(rx + 8));
      typeTxt.setAttribute('y', String(ry + (isCompact ? 12 : 16)));
      typeTxt.setAttribute('fill', pal.badge);
      typeTxt.setAttribute('font-size', isCompact ? '8.5' : '10.5');
      typeTxt.setAttribute('font-weight', '700');
      typeTxt.setAttribute('font-family', 'Consolas, monospace');
      const typeLabel =
        node.node_type === 'Evidence'
          ? isNumericalEvidenceNode(node)
            ? 'CSV EVIDENCE'
            : 'FRAME EVIDENCE'
          : node.node_type.toUpperCase();
      typeTxt.textContent = typeLabel;
      g.appendChild(typeTxt);

      // Main label text wrapped across up to 2 lines
      const maxChars = isCompact ? Math.max(19, Math.floor((pos.w - 12) / 5.0)) : Math.max(26, Math.floor((pos.w - 16) / 6.0));
      const lines = wrapSvgLabelLines(node.label, maxChars);
      const mainTxt = document.createElementNS(ns, 'text');
      mainTxt.setAttribute('x', String(rx + 8));
      mainTxt.setAttribute('y', String(ry + (isCompact ? 25 : 33)));
      mainTxt.setAttribute('fill', pal.text);
      mainTxt.setAttribute('font-size', isCompact ? '9.5' : '12');
      mainTxt.setAttribute('font-weight', '600');
      mainTxt.setAttribute('font-family', 'Segoe UI, sans-serif');

      lines.forEach((ln, lIdx) => {
        const tspan = document.createElementNS(ns, 'tspan');
        tspan.setAttribute('x', String(rx + 8));
        if (lIdx > 0) {
          tspan.setAttribute('dy', isCompact ? '11' : '14');
        }
        tspan.textContent = ln;
        mainTxt.appendChild(tspan);
      });
      g.appendChild(mainTxt);

      const nStart = node.sample_start ?? node.start_sample;
      const nEnd = node.sample_end ?? node.end_sample ?? nStart;
      if (!isCompact && nStart !== null && nStart !== undefined) {
        const subTxt = document.createElementNS(ns, 'text');
        subTxt.setAttribute('x', String(rx + 8));
        subTxt.setAttribute('y', String(ry + pos.h - 7));
        subTxt.setAttribute('fill', '#475569');
        subTxt.setAttribute('font-size', '10');
        subTxt.setAttribute('font-family', 'Consolas, monospace');
        subTxt.textContent = `samples ${nStart}–${nEnd}`;
        g.appendChild(subTxt);
      }

      g.addEventListener('click', () => selectGraphNode(node.node_id, true));
      g.addEventListener('keydown', (ev) => {
        if (ev.key === 'Enter' || ev.key === ' ') {
          ev.preventDefault();
          selectGraphNode(node.node_id, true);
        }
      });

      nodesGroup.appendChild(g);
    });

    // In-canvas SVG Legend Box in top-right corner (P1.1)
    const legW = isCompact ? 156 : 210;
    const legH = isCompact ? 38 : 46;
    const legX = width - legW - 10;
    const legY = 8;
    const legG = document.createElementNS(ns, 'g');
    legG.setAttribute('aria-label', 'Edge style legend');

    const legRect = document.createElementNS(ns, 'rect');
    legRect.setAttribute('x', String(legX));
    legRect.setAttribute('y', String(legY));
    legRect.setAttribute('width', String(legW));
    legRect.setAttribute('height', String(legH));
    legRect.setAttribute('rx', '6');
    legRect.setAttribute('fill', 'rgba(255, 255, 255, 0.94)');
    legRect.setAttribute('stroke', '#CBD5E1');
    legRect.setAttribute('stroke-width', '1');
    legG.appendChild(legRect);

    const l1 = document.createElementNS(ns, 'line');
    l1.setAttribute('x1', String(legX + 8));
    l1.setAttribute('y1', String(legY + (isCompact ? 13 : 16)));
    l1.setAttribute('x2', String(legX + 28));
    l1.setAttribute('y2', String(legY + (isCompact ? 13 : 16)));
    l1.setAttribute('stroke', '#475569');
    l1.setAttribute('stroke-width', '2');
    legG.appendChild(l1);

    const t1 = document.createElementNS(ns, 'text');
    t1.setAttribute('x', String(legX + 34));
    t1.setAttribute('y', String(legY + (isCompact ? 16 : 20)));
    t1.setAttribute('fill', '#334155');
    t1.setAttribute('font-size', isCompact ? '9' : '11');
    t1.setAttribute('font-family', 'Segoe UI, sans-serif');
    t1.textContent = 'Solid: source relation';
    legG.appendChild(t1);

    const l2 = document.createElementNS(ns, 'line');
    l2.setAttribute('x1', String(legX + 8));
    l2.setAttribute('y1', String(legY + (isCompact ? 27 : 33)));
    l2.setAttribute('x2', String(legX + 28));
    l2.setAttribute('y2', String(legY + (isCompact ? 27 : 33)));
    l2.setAttribute('stroke', '#D97706');
    l2.setAttribute('stroke-width', '2');
    l2.setAttribute('stroke-dasharray', '4 3');
    legG.appendChild(l2);

    const t2 = document.createElementNS(ns, 'text');
    t2.setAttribute('x', String(legX + 34));
    t2.setAttribute('y', String(legY + (isCompact ? 30 : 37)));
    t2.setAttribute('fill', '#334155');
    t2.setAttribute('font-size', isCompact ? '9' : '11');
    t2.setAttribute('font-family', 'Segoe UI, sans-serif');
    t2.textContent = 'Dashed: interpretation';
    legG.appendChild(t2);

    nodesGroup.appendChild(legG);
  }

  function renderAccessibleNodeList(graph) {
    const container = document.getElementById('accessible-node-list');
    if (!container || !graph) return;
    container.innerHTML = '';
    graph.nodes.forEach((node) => {
      const nStart = node.sample_start ?? node.start_sample;
      const btn = document.createElement('button');
      btn.className = 'acc-node-btn' + (node.node_id === state.selectedGraphNodeId ? ' selected' : '');
      btn.innerHTML = `
        <span><strong>[${escapeHtml(node.node_type)}]</strong> ${escapeHtml(node.label)}</span>
        <span class="badge-pill badge-outline">${
          nStart !== null && nStart !== undefined ? `#${nStart}` : 'root'
        }</span>
      `;
      btn.addEventListener('click', () => selectGraphNode(node.node_id, true));
      container.appendChild(btn);
    });
  }

  // =========================================================================
  // 8. Source Inspector & Node Selection (Gate G03 & G08)
  // =========================================================================

  function selectGraphNode(nodeId, scrubToNode = true) {
    if (!state.currentReport || !state.currentReport.evidence_graph) return;
    const graph = state.currentReport.evidence_graph;
    const node =
      graph.nodes.find((n) => n.node_id === nodeId) ||
      graph.nodes.find((n) => n.evidence_id && `node_${n.evidence_id}` === nodeId);
    if (!node) return;

    state.selectedGraphNodeId = node.node_id;
    renderEvidenceGraphs(graph);

    // Update Compact Source Inspector (Tab 1)
    document.getElementById('inspector-node-title').textContent = node.label;
    document.getElementById('inspector-node-type-badge').textContent = node.node_type;

    const nStart = node.sample_start ?? node.start_sample;
    const nEnd = node.sample_end ?? node.end_sample ?? nStart;
    const sampleTarget = nStart !== null && nStart !== undefined ? nStart : state.currentSample;

    if (scrubToNode && nStart !== null && nStart !== undefined) {
      setSampleCursor(nStart, true);
    }

    const evPayload = (node.inspector_payload && node.inspector_payload.evidence) || {};
    const nodeSummary =
      node.summary ||
      evPayload.summary_text ||
      evPayload.caption ||
      (node.inspector_payload &&
        (node.inspector_payload.summary_text ||
          node.inspector_payload.caption ||
          node.inspector_payload.description ||
          node.inspector_payload.rationale ||
          node.inspector_payload.statement)) ||
      node.sublabel ||
      '';

    const firstModelRec = (state.currentReport.model_receipts || [])[0] || {};
    const modelId =
      firstModelRec.reported_model_id ||
      firstModelRec.model_id ||
      'gemini-4-argon';
    const authorityText =
      node.status === 'computed_record' || isNumericalEvidenceNode(node)
        ? 'Calculation checked (T1 CSV)'
        : node.node_type === 'Hypothesis'
        ? `Unverified hypothesis (${modelId})`
        : `Sampled frame / event (${modelId})`;

    const metaGrid = document.getElementById('inspector-provenance-meta');
    metaGrid.innerHTML = `
      <div class="meta-kv"><span class="meta-k">NODE ID</span><span class="meta-v">${escapeHtml(node.node_id)}</span></div>
      <div class="meta-kv"><span class="meta-k">TYPE</span><span class="meta-v">${escapeHtml(node.node_type)}</span></div>
      <div class="meta-kv"><span class="meta-k">SAMPLE RANGE</span><span class="meta-v">${
        nStart ?? 0
      }–${nEnd ?? 0}</span></div>
      <div class="meta-kv"><span class="meta-k">AUTHORITY / MODEL</span><span class="meta-v">${escapeHtml(
        authorityText
      )}</span></div>
    `;

    const summaryEl = document.getElementById('inspector-summary');
    if (summaryEl) {
      summaryEl.textContent = nodeSummary || node.label;
    }

    // Check if node corresponds to a FrameEvidence or has associated sample
    const frameBox = document.getElementById('inspector-frame-box');
    const frameImg = document.getElementById('inspector-frame-img');
    const frameCaption = document.getElementById('inspector-frame-caption');

    const matchingFrameEv = (state.currentReport.frame_evidence || []).find(
      (fe) =>
        fe.evidence_id === node.evidence_id ||
        `node_ev_${fe.evidence_id}` === node.node_id ||
        fe.sample_index === sampleTarget
    );

    const camToPreview = matchingFrameEv ? matchingFrameEv.camera : node.camera || 'top';
    const previewSample = matchingFrameEv ? matchingFrameEv.sample_index : sampleTarget;

    if (state.manifest) {
      frameBox.classList.remove('hidden');
      const frameUrl = `/api/episodes/${encodeURIComponent(
        state.manifest.episode_id
      )}/frames/${camToPreview}/${previewSample}?tier=detail`;
      frameImg.src = frameUrl;
      frameCaption.textContent = `${camToPreview}.mp4 · sample #${previewSample} · ${formatVideoAndWallTime(
        previewSample
      )}`;
      frameBox.dataset.camera = camToPreview;
      frameBox.dataset.sample = String(previewSample);
    }

    // Update Large Inspector in Tab 2 (Evidence Map View)
    updateLargeEvidenceInspector(node, previewSample, nodeSummary);
  }

  function updateLargeEvidenceInspector(node, sampleIdx, nodeSummary) {
    if (!state.manifest || !state.currentReport) return;
    document.getElementById('large-inspector-title').textContent = `${node.node_type}: ${node.label}`;
    document.getElementById('large-inspector-badge').textContent = node.node_id;
    document.getElementById('large-inspector-summary').textContent = nodeSummary || node.sublabel || '';

    const epId = state.manifest.episode_id;
    const clampedSample = Math.max(
      0,
      Math.min((state.trajectory ? state.trajectory.total_rows : 1) - 1, sampleIdx)
    );

    document.getElementById('large-top-img').src = `/api/episodes/${encodeURIComponent(
      epId
    )}/frames/top/${clampedSample}?tier=detail`;
    document.getElementById('large-wrist-img').src = `/api/episodes/${encodeURIComponent(
      epId
    )}/frames/wrist/${clampedSample}?tier=detail`;
    document.getElementById('large-top-pts').textContent = `sample #${clampedSample} · ${formatVideoAndWallTime(
      clampedSample
    )}`;
    document.getElementById('large-wrist-pts').textContent = `sample #${clampedSample} · ${formatVideoAndWallTime(
      clampedSample
    )}`;

    const provBox = document.getElementById('large-provenance-box');
    const firstModelRec = (state.currentReport.model_receipts || [])[0] || {};
    const modelId =
      firstModelRec.reported_model_id ||
      firstModelRec.model_id ||
      'gemini-4-argon';
    const trList = (state.currentReport.tool_receipts || [])
      .map((tr) => `${tr.tool_name} (${tr.status}, ${tr.duration_ms}ms, rev=${tr.episode_revision})`)
      .join(' | ');
    provBox.innerHTML = `
      <div><strong>Model ID:</strong> <code>${escapeHtml(modelId)}</code> (${escapeHtml(state.currentReport.analysis_mode)})</div>
      <div><strong>Episode Revision:</strong> <code>${escapeHtml(state.manifest.episode_revision)}</code></div>
      <div><strong>Trajectory CSV SHA-256:</strong> <code>${escapeHtml(state.manifest.trajectory.sha256.slice(0, 24))}…</code></div>
      <div><strong>Action Semantics:</strong> <code>${escapeHtml(state.manifest.action_semantics.kind)}</code></div>
      <div><strong>Tool Receipts:</strong> ${escapeHtml(trList || 'None')}</div>
    `;
  }

  function renderCsvSliceTable(centerSample) {
    if (!state.trajectory) return;
    const tbody = document.getElementById('inspector-csv-tbody');
    if (!tbody) return;

    const joint = state.selectedJoint;
    document.getElementById('csv-th-state').textContent = `state.${joint}`;
    document.getElementById('csv-th-action').textContent = `action.${joint}`;
    document.getElementById('csv-th-diff').textContent = `Δ.${joint}`;
    document.getElementById('inspector-csv-range-label').textContent = `Rows around sample #${centerSample}`;

    const total = state.trajectory.total_rows;
    const startRow = Math.max(0, centerSample - 4);
    const endRow = Math.min(total - 1, centerSample + 4);

    const sSeries = state.trajectory.series[`state.${joint}`] || [];
    const aSeries = state.trajectory.series[`action.${joint}`] || [];
    const dSeries =
      state.trajectory.difference_series[`diff_${joint}`] ||
      state.trajectory.difference_series[`diff.${joint}`] ||
      [];

    tbody.innerHTML = '';
    for (let r = startRow; r <= endRow; r++) {
      const tr = document.createElement('tr');
      if (r === centerSample) tr.className = 'active-row';
      const { videoSec, wallSec } = getSampleTimes(r);
      const sVal = sSeries[r] !== undefined ? Number(sSeries[r]).toFixed(2) : '—';
      const aVal = aSeries[r] !== undefined ? Number(aSeries[r]).toFixed(2) : '—';
      const dVal = dSeries[r] !== undefined ? Number(dSeries[r]).toFixed(2) : '—';

      tr.innerHTML = `
        <td>#${r}</td>
        <td>${videoSec.toFixed(2)}s / ${wallSec.toFixed(2)}s</td>
        <td>${sVal}</td>
        <td>${aVal}</td>
        <td>${dVal}</td>
      `;
      tr.style.cursor = 'pointer';
      tr.addEventListener('click', () => setSampleCursor(r, true));
      tbody.appendChild(tr);
    }
  }

  function updateVerificationLoopBar(report) {
    if (!report) return;
    const s1 = document.getElementById('step-01-detail');
    const s2 = document.getElementById('step-02-detail');
    const s3 = document.getElementById('step-03-detail');
    const s4 = document.getElementById('step-04-detail');
    const summaryBadge = document.getElementById('loop-status-summary');
    const annPolicy = report.annotation_disclosure || report.annotation_policy || 'disclosed';
    const firstModelRec = (report.model_receipts || [])[0] || {};
    const modelId =
      firstModelRec.reported_model_id ||
      firstModelRec.model_id ||
      'gemini-4-argon';
    if (summaryBadge) {
      summaryBadge.textContent = `${modelId} · 2 semantic calls max · 1 tool round · No hidden retries`;
    }
    if (s1) {
      s1.textContent = `Episode ${report.episode_id} · samples ${report.focus_start_sample}–${report.focus_end_sample} · annotation ${annPolicy}.`;
    }
    if (s2) {
      const planRec = (report.model_receipts || []).find((m) => (m.call_number ?? m.call_index) === 1);
      const planModel = planRec ? planRec.reported_model_id || planRec.model_id : modelId;
      s2.textContent = planRec
        ? `Call #1 (${planModel}): ${planRec.status} in ${planRec.duration_ms ?? planRec.latency_ms}ms.`
        : `Saved reference inspection plan (${report.focus_start_sample}–${report.focus_end_sample}).`;
    }
    if (s3) {
      s3.textContent = `Executed ${report.tool_receipts.length} tools · ${report.computed_facts.length} computed facts · ${report.frame_evidence.length} decoded frames.`;
    }
    if (s4) {
      const revRec = (report.model_receipts || []).find((m) => (m.call_number ?? m.call_index) === 2);
      const revModel = revRec ? revRec.reported_model_id || revRec.model_id : modelId;
      s4.textContent = revRec
        ? `Call #2 (${revModel}): ${revRec.status} in ${revRec.duration_ms ?? revRec.latency_ms}ms.`
        : `Mode: ${report.analysis_mode} · ${report.visual_findings.length} visual findings · ${report.hypotheses.length} hypotheses.`;
    }
  }

  // =========================================================================
  // 9. Exact PyAV Decoded Frame Modal (Gate G03)
  // =========================================================================

  async function openExactFrameModal(cameraId, sampleIndex) {
    if (!state.manifest) return;
    const modal = document.getElementById('modal-frame-viewer');
    const imgEl = document.getElementById('modal-exact-frame-img');
    const metaEl = document.getElementById('modal-frame-ref-details');

    const epId = state.manifest.episode_id;
    imgEl.src = `/api/episodes/${encodeURIComponent(epId)}/frames/${cameraId}/${sampleIndex}?tier=detail`;
    metaEl.innerHTML = '<div>Decoding exact frame via PyAV...</div>';
    if (typeof modal.showModal === 'function') modal.showModal();

    try {
      const ref = await apiFetch(
        `/api/episodes/${encodeURIComponent(epId)}/frames/${cameraId}/${sampleIndex}/meta?tier=detail`
      );
      metaEl.innerHTML = `
        <div class="meta-kv"><span class="meta-k">FRAME ID</span><span class="meta-v">${escapeHtml(ref.frame_id)}</span></div>
        <div class="meta-kv"><span class="meta-k">EPISODE REVISION</span><span class="meta-v">${escapeHtml(ref.episode_revision)}</span></div>
        <div class="meta-kv"><span class="meta-k">CAMERA / SAMPLE</span><span class="meta-v">${escapeHtml(ref.camera)} · sample #${ref.sample_index}</span></div>
        <div class="meta-kv"><span class="meta-k">TIMESTAMP / PTS</span><span class="meta-v">${ref.time_s}s · PTS=${ref.pts ?? 'N/A'}</span></div>
        <div class="meta-kv"><span class="meta-k">DIMENSIONS</span><span class="meta-v">${ref.width}×${ref.height} (${escapeHtml(ref.resolution_tier)})</span></div>
        <div class="meta-kv"><span class="meta-k">SOURCE VIDEO SHA-256</span><span class="meta-v">${escapeHtml(ref.source_video_sha256)}</span></div>
        <div class="meta-kv"><span class="meta-k">DECODED FRAME SHA-256</span><span class="meta-v">${escapeHtml(ref.decoded_frame_sha256)}</span></div>
      `;
    } catch (err) {
      metaEl.innerHTML = `<div class="text-amber">Error fetching FrameRef metadata: ${escapeHtml(err.message)}</div>`;
    }
  }

  // =========================================================================
  // 10. Explicit Operator Authorization & Live ADC Gemini Run (Gates G05, G09, G10)
  // =========================================================================

  function openAuthorizationModal() {
    if (!state.manifest) return;
    const modal = document.getElementById('modal-authorize');
    const summaryEl = document.getElementById('auth-modal-summary');
    const startSample = parseInt(document.getElementById('input-start-sample').value || '0', 10);
    const endSample = parseInt(document.getElementById('input-end-sample').value || '0', 10);
    const adcInfo = state.adcStatus || {};
    const isAdcAvailable = Boolean(adcInfo.adc_available || adcInfo.available);

    summaryEl.innerHTML = `
      <div class="meta-kv"><span class="meta-k">EPISODE ID</span><span class="meta-v">${escapeHtml(state.manifest.episode_id)}</span></div>
      <div class="meta-kv"><span class="meta-k">SAMPLE WINDOW</span><span class="meta-v">${startSample} to ${endSample} (${escapeHtml(formatRangeVideoAndWallTime(startSample, endSample))})</span></div>
      <div class="meta-kv"><span class="meta-k">CALL &amp; TOOL BUDGET</span><span class="meta-v">Max 2 Gemini calls · 1 tool round (T1+T3)</span></div>
      <div class="meta-kv"><span class="meta-k">FRAME BUDGET</span><span class="meta-v">≤8 overview + ≤8 detail (≤24 total)</span></div>
      <div class="meta-kv"><span class="meta-k">AUTH MODE</span><span class="meta-v">Google Cloud ADC (Vertex AI · ${escapeHtml(adcInfo.location || 'global')})</span></div>
      <div class="meta-kv"><span class="meta-k">ADC CREDENTIAL STATUS</span><span class="meta-v">${
        isAdcAvailable
          ? 'Active (' + escapeHtml(adcInfo.credential_type || 'ADC') + ')'
          : 'Not initialized locally'
      }</span></div>
    `;
    if (typeof modal.showModal === 'function') modal.showModal();
  }

  async function executeAuthorizedGeminiRun() {
    const modal = document.getElementById('modal-authorize');
    const chk = document.getElementById('chk-explicit-auth');
    if (!chk || !chk.checked) {
      alert('Explicit operator authorization checkbox must be checked to proceed.');
      return;
    }
    if (typeof modal.close === 'function') modal.close();

    const epId = state.activeEpisodeId;
    const question = document.getElementById('input-question').value.trim();
    const startSample = parseInt(document.getElementById('input-start-sample').value || '0', 10);
    const endSample = parseInt(document.getElementById('input-end-sample').value || '0', 10);

    const stopBtn = document.getElementById('btn-stop-investigation');
    stopBtn.classList.remove('hidden');
    state.isInvestigationRunning = true;

    document.getElementById('direct-answer-text').textContent =
      'Running bounded 2-call Gemini investigation via Google Cloud ADC...';

    try {
      const inv = await apiFetch('/api/investigations', {
        method: 'POST',
        body: {
          episode_id: epId,
          question: question,
          start_sample: startSample,
          end_sample: endSample,
          explicit_authorization: true,
          annotation_disclosure: 'disclosed',
        },
      });
      state.activeInvestigationId = inv.investigation_id;

      const completedInv = await apiFetch(`/api/investigations/${inv.investigation_id}/run`, {
        method: 'POST',
      });

      state.isInvestigationRunning = false;
      stopBtn.classList.add('hidden');

      if (completedInv.episode_id !== state.activeEpisodeId) {
        // Episode changed while running; ignore stale result (Gate G09)
        return;
      }

      if (completedInv.status === 'completed' && completedInv.final_report) {
        state.currentReport = completedInv.final_report;
        renderReportToUI(completedInv.final_report);
        await refreshCatalog();
      } else {
        // Live call failed (e.g., ADC not logged in) — never silently substitute saved analysis! (Gate G10)
        const modeBanner = document.getElementById('analysis-mode-banner');
        modeBanner.textContent = `Live Run Failed (${completedInv.status})`;
        modeBanner.className = 'mode-pill mode-failed';
        document.getElementById('report-mode-badge').textContent = 'LIVE GEMINI FAILED (NO SILENT FALLBACK)';
        document.getElementById('report-mode-badge').className = 'badge-pill badge-coral';
        document.getElementById('direct-answer-text').textContent =
          `Live Gemini ADC execution stopped with status '${completedInv.status}': ${completedInv.error_message || 'Unknown error'}. ` +
          `Per Gate G10, LabLens does not silently replace failed live calls with saved analysis. Click "Load Saved Analysis" if you wish to view the pre-computed reference report.`;
      }
    } catch (err) {
      state.isInvestigationRunning = false;
      stopBtn.classList.add('hidden');
      document.getElementById('direct-answer-text').textContent = `Investigation error: ${err.message}`;
    }
  }

  // =========================================================================
  // 11. Interactive Deterministic Tool Execution (Run T1 + T3)
  // =========================================================================

  async function runDeterministicToolsOnRange() {
    if (!state.manifest) return;
    const epId = state.manifest.episode_id;
    const startSample = parseInt(document.getElementById('input-start-sample').value || '0', 10);
    const endSample = parseInt(document.getElementById('input-end-sample').value || '0', 10);
    const midSample = Math.round((startSample + endSample) / 2);

    try {
      const [t1Resp, t3Resp] = await Promise.all([
        apiFetch('/api/tools/query-signal-window', {
          method: 'POST',
          body: {
            episode_id: epId,
            start_sample: startSample,
            end_sample: endSample,
            signals: [
              'state.gripper',
              'action.gripper',
              'state.wrist_flex',
              'action.wrist_flex',
              'state.elbow_flex',
              'action.elbow_flex',
            ],
            include_State_action_diff: true,
            requested_difference_pairs: ['diff_gripper', 'diff_wrist_flex', 'diff_elbow_flex'],
          },
        }),
        apiFetch('/api/tools/get-frame-bundle', {
          method: 'POST',
          body: {
            episode_id: epId,
            camera: 'both',
            camera_ids: ['top', 'wrist'],
            start_sample: startSample,
            end_sample: endSample,
            sample_count: 3,
            resolution_tier: 'detail',
          },
        }),
      ]);

      const facts = t1Resp.result.numerical_evidence || [];
      const frames = t3Resp.result.frame_evidence || [];
      document.getElementById('direct-answer-text').textContent =
        `Deterministic T1+T3 execution complete on samples ${startSample}–${endSample}: computed ${facts.length} full-resolution CSV facts (${t1Resp.receipt.duration_ms}ms) and decoded ${frames.length} exact PyAV frames (${t3Resp.receipt.duration_ms}ms).`;
      setSampleCursor(midSample, true);
    } catch (err) {
      document.getElementById('direct-answer-text').textContent = `Tool execution failed: ${err.message}`;
    }
  }

  // =========================================================================
  // 12. UI Event Listeners
  // =========================================================================

  function bindUIEvents() {
    // Primary workspace tabs (Investigation vs Evidence Map)
    document.querySelectorAll('.main-tab-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        const targetTab = btn.dataset.tab;
        document.querySelectorAll('.main-tab-btn').forEach((b) => {
          const active = b === btn;
          b.classList.toggle('active', active);
          b.setAttribute('aria-selected', active ? 'true' : 'false');
        });
        document.getElementById('panel-investigation').classList.toggle('active', targetTab === 'investigation');
        document.getElementById('panel-evidence-map').classList.toggle('active', targetTab === 'evidence-map');
      });
    });

    document.getElementById('btn-expand-map-tab').addEventListener('click', () => {
      document.getElementById('tab-btn-evidence-map').click();
    });

    // Catalog filters
    document.querySelectorAll('.filter-chip').forEach((chip) => {
      chip.addEventListener('click', () => {
        document.querySelectorAll('.filter-chip').forEach((c) => c.classList.remove('active'));
        chip.classList.add('active');
        state.activeFilter = chip.dataset.filter;
        renderCatalogList();
      });
    });

    // Validate All (16) button
    document.getElementById('btn-validate-all').addEventListener('click', async () => {
      const btn = document.getElementById('btn-validate-all');
      btn.textContent = 'Validating 16...';
      try {
        const catalog = await apiFetch('/api/catalog/validate-all', { method: 'POST' });
        state.catalog = catalog;
        renderCatalogCounters(catalog.counts);
        renderCatalogList();
        btn.textContent = `Validated (${catalog.counts.validated}/16)`;
      } catch (err) {
        btn.textContent = 'Validate All (16)';
        alert(`Validation error: ${err.message}`);
      }
    });

    // Export JSON button
    document.getElementById('btn-export-json').addEventListener('click', () => {
      if (!state.activeEpisodeId) return;
      window.location.href = `/api/episodes/${encodeURIComponent(state.activeEpisodeId)}/export`;
    });

    // Joint selector chips
    document.querySelectorAll('.joint-chip').forEach((chip) => {
      chip.addEventListener('click', () => {
        document.querySelectorAll('.joint-chip').forEach((c) => c.classList.remove('active'));
        chip.classList.add('active');
        state.selectedJoint = chip.dataset.joint;
        document.querySelectorAll('.joint-name-lbl').forEach((el) => {
          el.textContent = state.selectedJoint;
        });
        setSampleCursor(state.currentSample, false);
      });
    });

    // Scrubber & transport controls
    const scrubber = document.getElementById('timeline-scrubber');
    scrubber.addEventListener('input', () => {
      setSampleCursor(parseInt(scrubber.value, 10), true);
    });

    document.getElementById('btn-prev-frame').addEventListener('click', () => {
      setSampleCursor(state.currentSample - 1, true);
    });
    document.getElementById('btn-next-frame').addEventListener('click', () => {
      setSampleCursor(state.currentSample + 1, true);
    });

    document.getElementById('btn-set-range-start').addEventListener('click', () => {
      document.getElementById('input-start-sample').value = state.currentSample;
      updateRangeTimeBadge();
      drawTrajectoryCanvas();
    });
    document.getElementById('btn-set-range-end').addEventListener('click', () => {
      document.getElementById('input-end-sample').value = state.currentSample;
      updateRangeTimeBadge();
      drawTrajectoryCanvas();
    });

    document.getElementById('input-start-sample').addEventListener('change', () => {
      updateRangeTimeBadge();
      drawTrajectoryCanvas();
    });
    document.getElementById('input-end-sample').addEventListener('change', () => {
      updateRangeTimeBadge();
      drawTrajectoryCanvas();
    });

    // Play/Pause synchronized dual video
    const videoTop = document.getElementById('video-top');
    const videoWrist = document.getElementById('video-wrist');
    const playBtn = document.getElementById('btn-play-pause');

    playBtn.addEventListener('click', () => {
      if (state.isPlaying) {
        videoTop.pause();
        videoWrist.pause();
        state.isPlaying = false;
        playBtn.textContent = '▶ Play';
      } else {
        state.isPlaying = true;
        playBtn.textContent = '⏸ Pause';
        videoTop.play().catch(() => {});
        videoWrist.play().catch(() => {});
      }
    });

    videoTop.addEventListener('timeupdate', () => {
      if (state.isPlaying && state.manifest) {
        const fps = state.manifest.timing_semantics.fps || 30;
        const sample = Math.round(videoTop.currentTime * fps);
        setSampleCursor(sample, false);
      }
    });

    videoTop.addEventListener('ended', () => {
      state.isPlaying = false;
      playBtn.textContent = '▶ Play';
    });

    // Trajectory canvas interactive click/drag scrubbing
    const canvas = document.getElementById('trajectory-canvas');
    let isDraggingCanvas = false;
    function scrubFromCanvasEvent(ev) {
      if (!state.trajectory) return;
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const x = (ev.clientX - rect.left) * scaleX;
      const padL = 46;
      const padR = 16;
      const plotW = canvas.width - padL - padR;
      const ratio = Math.max(0, Math.min(1, (x - padL) / plotW));
      const sample = Math.round(ratio * (state.trajectory.total_rows - 1));
      setSampleCursor(sample, true);
    }
    canvas.addEventListener('mousedown', (ev) => {
      isDraggingCanvas = true;
      scrubFromCanvasEvent(ev);
    });
    window.addEventListener('mousemove', (ev) => {
      if (isDraggingCanvas) scrubFromCanvasEvent(ev);
    });
    window.addEventListener('mouseup', () => {
      isDraggingCanvas = false;
    });

    // Open exact frame buttons
    document.getElementById('btn-open-exact-top').addEventListener('click', () => {
      openExactFrameModal('top', state.currentSample);
    });
    document.getElementById('btn-open-exact-wrist').addEventListener('click', () => {
      openExactFrameModal('wrist', state.currentSample);
    });
    document.getElementById('btn-inspector-open-modal').addEventListener('click', () => {
      const box = document.getElementById('inspector-frame-box');
      const cam = box.dataset.camera || 'top';
      const smp = parseInt(box.dataset.sample || String(state.currentSample), 10);
      openExactFrameModal(cam, smp);
    });
    document.getElementById('btn-close-frame-modal').addEventListener('click', () => {
      const modal = document.getElementById('modal-frame-viewer');
      if (typeof modal.close === 'function') modal.close();
    });

    // Accessible list toggle
    document.getElementById('btn-toggle-accessible-list').addEventListener('click', () => {
      const list = document.getElementById('accessible-node-list');
      const btn = document.getElementById('btn-toggle-accessible-list');
      const isHidden = list.classList.toggle('hidden');
      btn.setAttribute('aria-expanded', isHidden ? 'false' : 'true');
    });

    // Computed Facts show all / show top 4 toggle
    const toggleFactsBtn = document.getElementById('btn-toggle-computed-facts');
    if (toggleFactsBtn) {
      toggleFactsBtn.addEventListener('click', () => {
        state.showAllComputedFacts = !state.showAllComputedFacts;
        if (state.currentReport) {
          renderReportToUI(state.currentReport);
        }
      });
    }

    // Secondary dev utilities toggle (Validate All, Export JSON, Run T1+T3)
    const toggleDevBtn = document.getElementById('btn-toggle-dev-tools');
    if (toggleDevBtn) {
      toggleDevBtn.addEventListener('click', () => {
        const shown = document.body.classList.toggle('show-dev-utilities');
        toggleDevBtn.textContent = shown ? 'Hide Utilities' : 'Utilities';
      });
    }

    // Evidence Map node type filters
    document.querySelectorAll('.ev-type-filter').forEach((btn) => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.ev-type-filter').forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        state.activeNodeTypeFilter = btn.dataset.nodetype;
        if (state.currentReport) {
          renderEvidenceGraphs(state.currentReport.evidence_graph);
        }
      });
    });

    // Inspect with Gemini & Authorization modal
    document.getElementById('btn-inspect-gemini').addEventListener('click', openAuthorizationModal);
    document.getElementById('btn-close-auth-modal').addEventListener('click', () => {
      document.getElementById('modal-authorize').close();
    });
    document.getElementById('btn-cancel-auth').addEventListener('click', () => {
      document.getElementById('modal-authorize').close();
    });
    document.getElementById('btn-confirm-auth-run').addEventListener('click', executeAuthorizedGeminiRun);

    // Load Saved Analysis button
    document.getElementById('btn-load-saved').addEventListener('click', async () => {
      if (!state.activeEpisodeId) return;
      const saved = await apiFetch(
        `/api/episodes/${encodeURIComponent(state.activeEpisodeId)}/saved-report`
      );
      state.currentReport = saved;
      renderReportToUI(saved);
    });

    // Run T1+T3 button
    document.getElementById('btn-run-tools-only').addEventListener('click', runDeterministicToolsOnRange);

    // Stop Investigation button
    document.getElementById('btn-stop-investigation').addEventListener('click', async () => {
      if (!state.activeInvestigationId) return;
      try {
        await apiFetch(`/api/investigations/${state.activeInvestigationId}/stop`, { method: 'POST' });
      } catch (_e) {
        // ignore
      }
      state.isInvestigationRunning = false;
      document.getElementById('btn-stop-investigation').classList.add('hidden');
      document.getElementById('direct-answer-text').textContent = 'Investigation stopped by operator.';
    });
  }

  document.addEventListener('DOMContentLoaded', initApp);
})();
