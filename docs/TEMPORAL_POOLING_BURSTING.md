# Temporal Pooling And Bursting

This note explains the Layer 1 bursting issue seen while investigating
temporal pooling in `chat_htm`, what appears to cause it, where to look in the
code, and how to test it.

## Short Summary

The symptom is: enabling temporal pooling can increase column bursting in Layer
1 instead of smoothly improving temporal stability.

Current status: this is improved but not fully fixed. TP proximal learning is
now constrained by local distal evidence, and TP can strengthen real proximal
permanences for predicted columns. We still see some Layer 1 bursting with
temporal pooling enabled, so the remaining work is tuning and diagnosis rather
than declaring the issue solved.

Headless true-burst diagnostics show that the current repeated-sequence failure
mode is mostly **no previous prediction** for newly bursting Layer 1 columns.
That means the burst gate is not ignoring valid predictive segment evidence;
proximal TP can make columns win before distal sequence memory predicts them.
The reference delayed config uses a single runtime override with
active-predict reinforcement plus a moderate post-active bridge while leaving
the riskiest predicted-non-active path disabled.
Layer 1 also uses `sequence_memory.cells_per_column: 6` and
`sequence_memory.activation_threshold: 4`. Four cells per column left familiar
branching transitions under-represented; threshold `6` left familiar transitions
under-predicted, while threshold `3` produced too much multi-cell predictive
activity.

The main reason is that a column only avoids bursting when a cell was both:

- predictive at `t-1`
- backed by an active sequence segment at `t-1`

Temporal pooling interferes with that because it updates the same proximal
permanence tensor already used by spatial learning, so TP can make columns win
inhibition before distal sequence structure is strong enough to support
non-bursting predictions.

The current calculator applies local safeguards:

- TP proximal reinforcement only applies to columns that had active-predict
  cells in the previous temporal-pooler distal update, or columns that are
  currently segment-backed predictive but did not win inhibition.
- A one-step later-input rule reinforces a recently active-predict column only
  when it is still segment-backed predictive and failed to win inhibition on the
  next timestep.
- `temporal_pooling.spatial_permanence_inc` now strengthens real proximal
  permanences for those supported columns instead of adding a temporary overlap
  bonus.
- TP proximal reinforcement is split into local scale knobs for active-predict
  winners and the post-active bridge. The direct predicted-non-winner proximal
  path was removed because it made tuning harder and tended to create
  unpredicted winners.
- TP distal reinforcement now rewards synapses targeting the same `prev2`
  learning-cell context used to select a temporal-pooling segment.

## Implemented Fixes

Several local fixes are now in place.

1. **Persistence was removed.**

   Temporal pooling used to have an optional persistence mode. It kept a cell
   predictive for a few timesteps after its active-predict streak ended, and
   copied the cell's active segment timestamp forward so the burst gate accepted
   the prediction. Those predictions had no support from current distal input,
   and the GUI showed predictive cells with no segment over threshold.

   The temporal pooler now reads `predict_cells_time` and `active_segs_time`
   but never writes them. Only `PredictCellsCalculator` sets predictive state.

2. **TP proximal learning is trust-gated.**

   `TemporalPoolerCalculator::update_proximal()` is now the runtime proximal
   TP update. It consumes active-predict support recorded by the preceding
   `update_distal()` call, plus current predictive-cell and active-segment
   state, before changing proximal permanence.

   This keeps high `temporal_pooling.spatial_permanence_inc` values from being
   a generic spatial-learning shortcut. TP proximal learning only reinforces
   columns with local distal evidence: active-predict support, current
   segment-backed prediction, or the stricter post-active bridge condition.

Both fixes use local per-cell/per-column state already available in the layer.
No global burst-rate rule or layer-wide normalization was added.

3. **Active-predict support reinforces proximal permanence.**

   `TemporalPoolerCalculator` records which columns had active-predict cells
   during its distal update. After that same update, `HTMLayer::step_once()` asks
   the temporal pooler to run `update_proximal()` and reinforce active proximal
   inputs for those columns using `temporal_pooling.spatial_permanence_inc`.

   This means temporal pooling improves the real proximal support that future
   overlap calculations use. It is deliberately not a new synapse or long-lived
   trace; the bookkeeping is derived from the same active-predict condition that
   TP distal learning uses to reinforce or create segments.

   The same reinforcement path also handles two non-winner cases:

   - **Predicted but not active:** if a column is predictive at the current
     timestep and has an active segment timestamp, but did not win inhibition,
     TP gives the column's currently active proximal inputs one local permanence
     increment. This lets distal TP predictions become future proximal overlap
     instead of waiting for the column to win inhibition first.
   - **Recently active-predict and still predicted:** if a column had
     active-predict support on the previous timestep, is still segment-backed
     predictive on the current timestep, and did not win inhibition, TP gives
     the currently active proximal inputs one additional local increment. This
     lets a correctly predicted activation grow proximal support for inputs just
     after that activation, but only while distal evidence still says the column
     belongs.

   The later-input rule deliberately requires both previous active-predict
   support and current segment-backed prediction. Without the current prediction
   check, TP would smear proximal permanence onto whatever happened to follow a
   correct activation. Without the previous active-predict check, the rule would
   be the generic predicted-but-not-active case only.

4. **TP distal matching and reinforcement use the same context.**

   `TemporalPoolerCalculator::update_distal()` selects a best-matching segment
   by counting synapses whose targets are in the `prev2` learning-cell context.
   It now reinforces synapses targeting that same `prev2` context.

   Previously, a segment could be selected because it matched `prev2`, but then
   reinforced according to cells active at the current timestep. That mismatch
   made it hard for TP-created distal synapses to become connected and later
   drive earlier/later predictive state.

## Why Bursting Happens

For newly active columns, the burst/no-burst decision is made in:

- `htm_flow/src/sequence_pooler/active_cells/active_cells.cpp`

The key rule is:

- if no cell in the column was predictive at `t-1` and had an active segment at
  `t-1`, the column bursts

Important supporting code paths:

- `htm_flow/src/htm_layer.cpp`
  - step order is: overlap -> inhibition -> spatial learning -> active cells ->
    predict cells -> sequence learning -> temporal pooler
  - TP runs after normal sequence memory work and writes back into shared state
  - TP distal records active-predict support, then TP proximal reinforces active
    proximal inputs for those supported columns so the effect appears in later
    overlap calculations
- `htm_flow/src/temporal_pooler/temporal_pooler.cpp`
  - `update_proximal()` changes `col_syn_perm_`
  - TP proximal reinforcement now requires active-predict support from the
    previous TP distal update, or current segment-backed predictive support for
    a column that did not win inhibition
  - `active_predict_proximal_scale` and `post_active_proximal_scale` control
    how much each local support path contributes
  - the later-input rule uses the previous distal update's active-predict
    support plus current segment-backed prediction before reinforcing a
    non-winning column's current active proximal inputs
  - `update_distal()` changes `distal_synapses_`
  - reused TP distal segments are matched and reinforced against `prev2`
    learning-cell context
- `htm_flow/src/sequence_pooler/predict_cells/predict_cells.cpp`
  - `predict_cells_time_` and `active_segs_time_` are produced only here,
    from connected distal support

## Important Investigation Notes

- Temporal pooling is not a separate clean layer of logic. It shares learning
  tensors with the normal layer algorithm.
- The burst spike observed when turning TP on later was useful as a diagnosis,
  but the real goal is a static config that works from the start.
- At the moment, the safest configuration is to keep TP proximal learning weak
  and active-predict-gated, using `temporal_pooling.spatial_permanence_inc` as
  the main "more TP / less TP" proximal knob.
- The current proximal reinforcement model is intentionally local:
  - active-predict winners reinforce the input that made them win
  - segment-backed predictive non-winners get current-input reinforcement
  - recently active-predict, still-predictive non-winners get a later-input
    reinforcement step
- Because TP proximal is now trust-gated, stronger TP settings can be explored
  with less risk of immediately reintroducing Layer 1 burst spikes.
- If `new_true_burst_causes` reports `no_prev_prediction`, proximal pressure is activating
  columns that distal sequence memory did not predict on the previous timestep.
  Back off `spatial_permanence_inc` and `post_active_proximal_scale` before
  changing active-cells bursting logic.
- If delayed TP causes new winners to burst, first reduce
  `post_active_proximal_scale`. That path teaches columns shortly after a
  correct activation, so it is the first local knob to check before lowering all
  TP learning.

## Current Config Direction

The delayed-runtime reference config is:

- `configs/word_rows_2layer_delayed_temporal_pooling_text.yaml`
- `configs/overrides/word_rows_2layer_enable_temporal_pooling.yaml`

Layer 1 temporal pooling is enabled at timestep 1000 with:

- `spatial_permanence_inc: 0.12`
- `active_predict_proximal_scale: 0.25`
- `post_active_proximal_scale: 0.5`
- moderate TP distal learning

Layer 1 also uses six cells per column and sequence-memory
`activation_threshold: 4` so learned distal segments can predict familiar
word-row transitions without becoming overly broad.

Generic `HTMLayerConfig` defaults now also bias toward safer TP startup
(`temp_sequence_permanence_inc > temp_spatial_permanence_inc`).

This is meant to avoid temporary selection effects that make columns win without
improving their real proximal support. TP proximal learning now changes the same
permanences that future overlap calculations inspect, but only for columns that
had active-predict support or segment-backed predictive support. Runtime logs
now report only the number of reinforced active proximal inputs. If that count
is non-zero but active columns do not broaden over time,
`temporal_pooling.spatial_permanence_inc` may be too small relative to
`connected_perm` and normal proximal decay. If a small set of columns dominates
every timestep, the value may be too large or the temporal support gates may be
too permissive.

## How To Test

The main regression coverage is in:

- `tests/integration/test_text_htm.cpp`

Relevant tests:

- `TextHTMIntegration.RuntimeEnableTemporalPoolingDoesNotSpikeLayer1Bursting`
  - keeps the original diagnostic scenario where TP is applied at runtime
- `TextHTMIntegration.AlwaysOnTemporalPoolingDoesNotRaiseLayer1BurstingVsNoTP`
  - compares always-on TP against a TP-off control
- `TextHTMIntegration.StrongTemporalPoolingDoesNotRaiseLayer1BurstingVsNoTP`
  - verifies that a much stronger Layer 1 TP setup does not automatically bring
    back the old burst failure mode

Calculator-level coverage lives in:

- `htm_flow/test/unit/test_temporal_pooler.cpp`
  - checks active-predict-gated TP proximal updates
  - checks predictive non-winner and post-active predictive non-winner proximal
    reinforcement

Run them with:

```bash
cmake --build build --target chat_htm_tests
./build/chat_htm_tests --gtest_filter=TextHTMIntegration.RuntimeEnableTemporalPoolingDoesNotSpikeLayer1Bursting:TextHTMIntegration.AlwaysOnTemporalPoolingDoesNotRaiseLayer1BurstingVsNoTP
```

For stronger TP tuning experiments, also run:

```bash
./build/chat_htm_tests --gtest_filter=TextHTMIntegration.StrongTemporalPoolingDoesNotRaiseLayer1BurstingVsNoTP
```

## What To Look At During Debugging

When investigating this issue again, check:

- whether Layer 1 active columns are increasing without a matching increase in
  predictive coverage
- whether bursting rises because TP proximal learning is bypassing the intended
  local temporal-support gate
- whether distal learning rates and thresholds make it too hard for TP-created
  synapses to become useful before new columns start winning
- whether TP distal segment creation/reuse is actually producing connected
  synapses that meet `activation_threshold` in `PredictCellsCalculator`
- whether TP proximal reinforcement is firing outside the old saturated winner
  set; add temporary instrumentation around `update_proximal()` if the aggregate
  `reinforced_inputs` log is not enough
- whether `temporal_pooling.spatial_permanence_inc` is too small to overcome
  normal proximal decay or so large that active-predict-supported columns become
  overly sticky

## Future Work

Future improvements should focus on making temporal learning stronger without
reintroducing burst spikes while keeping the state model easy to inspect.

The likely next step is to compare weak and moderate active-predict-gated
proximal reinforcement settings. Avoid adding another persistent synapse-like
trace unless this simpler local permanence update proves insufficient.

Do not treat the current implementation as a final bursting fix. It narrows when
TP proximal learning is allowed to change shared proximal permanence, but config
tuning still needs to balance stronger temporal pooling against remaining Layer
1 bursting.
