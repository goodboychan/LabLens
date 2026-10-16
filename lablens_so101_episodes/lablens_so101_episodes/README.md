# LabLens episode set: SO-101 cola can to coaster (QuantGate recordings)

Recorded real-robot data for analysis only. Nothing here runs a robot. **Internal use until the QuantGate public
release (2026-10-26).**

## What is in it

- **14 policy rollouts**: a learned ACT policy driving a real SO-101 from a Jetson Orin Nano Super, inside
  QuantGate's blinded trials.
  - The engine (precision, fine-tuned or not) and the condition differ by episode.
  - There are 4 successes and 10 failures of different kinds.
- **2 teleoperated demonstrations**: a person driving the leader arm. They come from the LeRobot v3 training set
  `chans/quantgate-so101-coaster`, and both are successes.
- **Do not describe a teleop demo as policy performance**, or the other way round. `episode.json → source` says which
  one each episode is.

`index.csv` lists every episode with its block, start position, gripper configuration, engine, outcome and the
operator's note.

## Per episode

```
<episode>/video/top.mp4        overhead camera 640x480, H.264 (Gemini-ready)
<episode>/video/wrist.mp4      wrist camera 320x200, H.264
<episode>/video/*_av1.mp4      original AV1 recordings (rollouts only; in the separate *_av1_originals.zip)
<episode>/trajectory.parquet   one row per control step (same data in trajectory.csv)
<episode>/episode.json         source, condition, engine, outcome labels, operator note, AI-judge answer, sync rules
```

### Trajectory columns

| Column | Meaning |
|---|---|
| `frame_index` | Control step `i`. Video frame `i` is the image observed at this step |
| `t_wall_s` (rollouts) / `t_s` (teleop) | Real time since the episode start, in seconds |
| `t_video_s` (rollouts) | `frame_index / 30`, the video's own clock. It is **not** real time; see below |
| `state.<joint>` | Measured follower joint position at the step |
| `action.<joint>` | Command at the step. For rollouts this is the policy output after chunk blending, clipped to the motor limits. For teleop it is the leader arm position |

- Joints: `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll` are in **degrees**.
- `gripper` is a **0–100 position**. It is not force and not contact.
- Everything is in joint space. There is no end-effector pose; computing one needs SO-101 kinematics.

## Synchronisation (read this before comparing video and logs)

- **Rollouts**: every video has exactly as many frames as the trajectory has rows (`episode.json → sync`). Frame `i`
  is the camera image observed at step `i`; `state[i]` was read at that step, and `action[i]` was sent right after it.
- **The clocks differ**:
  - Videos are encoded at a nominal 30 fps.
  - Control steps are not evenly spaced. The median is about 33 ms, but some steps take 150–200 ms (for example
    around action-chunk switches).
  - So frame `i` plays at `i / 30` s while it really happened at `t_wall_s[i]`. Use the frame index to pair video and
    log, and `t_wall_s` for durations.
- Chunk timing: `episode.json → trajectory.chunk_timing` lists the inference time and round trip of each 100-step
  chunk request.
- **Teleop demos**: LeRobot timestamps. Frame `i` is at `t_s = i / 30`.

## Labels and how far to trust them

| Field | What it is | Limits |
|---|---|---|
| `outcome.success` | The study's locked outcome: can standing on the coaster at the end, gripper released, no human contact | End state only |
| `outcome.automatic_judge` | Geometric judge on the overhead video, run live | Can be fooled when the can is off-centre; see the re-judge |
| `outcome.video_consensus_rejudge` | The study's re-judge from the saved video (§11 consensus circle). It agrees with the block totals | End state only |
| `operator_note` | What the operator saw, written at the time | Some failures were not seen directly; the operator was blind to the engine |
| `ai_judge_end_of_trial` | Cosmos-Reason2's answer on the final frame | Fine-tune blocks (T*) use the validated C1 prompt (`Answer: YES/NO`; live check 21/23 failures caught). Stress v1 used the first one-word setup (caught 1/21 in the re-evaluation). G6 and the positive control have no live answer |

Cases worth looking at:

- **AI judge said YES on a failure**: T1-002 (half on the coaster) and T0-018 (pushed off).
- **Placed, then knocked over afterwards**: T2-012, P0-021, G6-186. The end state alone hides the moment of the
  error.
- **Never grasped**: T2-014.
- **Dropped while carrying**: P0-016 (FP32), S0-005 (gripper pads removed).
- **Same position, different policy**: T1-001 (fine-tuned, success) vs T1-002 and T1-006 (old policy, failures).
  - These are recorded runs, not a controlled replay. Do not claim that swapping the model fixes a given episode.

## Engines

| Code | Meaning |
|---|---|
| `VC-000` | FP32, old policy |
| `VC-002` / `VC-020` | INT8 (ModelOpt, per-channel, 16 / 1024 calibration frames), old policy |
| `VC-025` | FP16, old policy |
| `FT-FP16` | FP16, policy fine-tuned on 50 demos at −4…+4 cm |
| `PC-B` | INT8 deliberately miscalibrated on black frames (a positive control) |

Source: the QuantGate repository (`docs/finetune_position_preregistration.md`, `docs/sessions/*operator_notes.md`).
Built by `scripts/make_lablens_bundle.py`. License: CC BY 4.0 (same as the datasets).
